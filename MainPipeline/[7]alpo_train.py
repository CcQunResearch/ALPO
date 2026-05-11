import json
import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class ALPODataset(Dataset):
    def __init__(self, data_path: str):
        self.records = []
        with open(data_path, 'r', encoding='utf-8') as file:
            if data_path.endswith('.jsonl'):
                for line in file:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
            else:
                self.records = json.load(file)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


@dataclass
class ALPODataCollator:
    tokenizer: AutoTokenizer
    cutoff_len: int = 8192

    def _encode_pair(self, prompt: str, completion: str):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not completion_ids:
            completion_ids = [self.tokenizer.eos_token_id]

        if len(completion_ids) >= self.cutoff_len:
            completion_ids = completion_ids[: self.cutoff_len - 1]
        prompt_budget = self.cutoff_len - len(completion_ids)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, labels

    def _pad(self, values: List[List[int]], pad_value: int):
        max_len = max(len(value) for value in values)
        return torch.tensor(
            [value + [pad_value] * (max_len - len(value)) for value in values],
            dtype=torch.long,
        )

    def __call__(self, features: List[Dict]):
        chosen = [self._encode_pair(item['prompt'], item['chosen']) for item in features]
        rejected = [self._encode_pair(item['prompt'], item['rejected']) for item in features]
        pad_id = self.tokenizer.pad_token_id

        return {
            'chosen_input_ids': self._pad([item[0] for item in chosen], pad_id),
            'chosen_attention_mask': self._pad([item[1] for item in chosen], 0),
            'chosen_labels': self._pad([item[2] for item in chosen], -100),
            'rejected_input_ids': self._pad([item[0] for item in rejected], pad_id),
            'rejected_attention_mask': self._pad([item[1] for item in rejected], 0),
            'rejected_labels': self._pad([item[2] for item in rejected], -100),
            'weights': torch.tensor([float(item.get('weight', 1.0)) for item in features], dtype=torch.float),
            'betas': torch.tensor([float(item.get('beta', 1.0)) for item in features], dtype=torch.float),
        }


def batch_logps(logits, labels, average_log_prob=False):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].clone()
    loss_mask = shift_labels != -100
    shift_labels[shift_labels == -100] = 0
    per_token_logps = torch.gather(
        shift_logits.log_softmax(-1),
        dim=2,
        index=shift_labels.unsqueeze(2),
    ).squeeze(2)
    sequence_logps = (per_token_logps * loss_mask).sum(dim=1)
    if average_log_prob:
        sequence_logps = sequence_logps / loss_mask.sum(dim=1).clamp_min(1)
    return sequence_logps


class ALPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        ref_model=None,
        beta_scale=1.0,
        average_log_prob=False,
        reference_free=False,
        disable_shuffling=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta_scale = beta_scale
        self.average_log_prob = average_log_prob
        self.reference_free = reference_free
        self.disable_shuffling = disable_shuffling
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

    def get_train_dataloader(self):
        if not self.disable_shuffling or self.args.world_size > 1:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _model_logps(self, model, input_ids, attention_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return batch_logps(outputs.logits, labels, average_log_prob=self.average_log_prob)

    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop('weights').to(model.device)
        betas = inputs.pop('betas').to(model.device) * self.beta_scale

        chosen_logps = self._model_logps(
            model,
            inputs['chosen_input_ids'],
            inputs['chosen_attention_mask'],
            inputs['chosen_labels'],
        )
        rejected_logps = self._model_logps(
            model,
            inputs['rejected_input_ids'],
            inputs['rejected_attention_mask'],
            inputs['rejected_labels'],
        )

        if self.reference_free:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)
        else:
            if self.ref_model is None:
                raise ValueError('ref_model is required when reference_free is false')
            with torch.no_grad():
                ref_chosen_logps = self._model_logps(
                    self.ref_model,
                    inputs['chosen_input_ids'],
                    inputs['chosen_attention_mask'],
                    inputs['chosen_labels'],
                )
                ref_rejected_logps = self._model_logps(
                    self.ref_model,
                    inputs['rejected_input_ids'],
                    inputs['rejected_attention_mask'],
                    inputs['rejected_labels'],
                )

        chosen_rewards = chosen_logps - ref_chosen_logps
        rejected_rewards = rejected_logps - ref_rejected_logps
        logits = betas * (chosen_rewards - rejected_rewards)
        losses = -F.logsigmoid(logits) * weights
        loss = losses.sum() / weights.sum().clamp_min(1e-6)
        return (loss, {'chosen_rewards': chosen_rewards, 'rejected_rewards': rejected_rewards}) if return_outputs else loss


def maybe_apply_lora(model, config):
    if config.get('alpo_finetuning_type', config.get('dpo_finetuning_type', 'full')) != 'lora':
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError('LoRA ALPO training requires the peft package.') from exc

    lora_config = LoraConfig(
        r=config.get('alpo_lora_rank', 8),
        lora_alpha=config.get('alpo_lora_alpha', 16),
        lora_dropout=config.get('alpo_lora_dropout', 0.05),
        target_modules=config.get('alpo_lora_target', 'all-linear'),
        task_type='CAUSAL_LM',
    )
    return get_peft_model(model, lora_config)


def resolve_paths(config, dirname):
    sft_model = config['sft_model']
    trpe_model = config['trpe_model']
    lang = config['lang']
    model_path = config['model_path']
    sft_proportion = config['sft_proportion']
    finetuning_type = config.get('alpo_finetuning_type', config.get('dpo_finetuning_type', 'full'))

    sft_name = 'sft_default' if sft_proportion == 1.0 else f'sft_{sft_proportion}'
    model_name_or_path = config.get(
        'alpo_model_name_or_path',
        osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, sft_name),
    )
    ref_model_name_or_path = config.get('alpo_ref_model_name_or_path', model_name_or_path)
    output_dir = config.get(
        'alpo_output_dir',
        osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, f'alpo_{sft_proportion}_{finetuning_type}'),
    )
    dataset_path = config.get(
        'alpo_dataset_path',
        osp.join(dirname, '..', 'LLaMAFactory', 'data', f'alpo_train_{sft_model}_{trpe_model}_{lang}_{sft_proportion}.jsonl'),
    )
    return model_name_or_path, ref_model_name_or_path, output_dir, dataset_path


if __name__ == '__main__':
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpus', '0'))

    model_name_or_path, ref_model_name_or_path, output_dir, dataset_path = resolve_paths(config, dirname)
    if not osp.exists(dataset_path):
        raise FileNotFoundError(f'ALPO dataset not found: {dataset_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if config.get('alpo_bf16', True) else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = maybe_apply_lora(model, config)
    if config.get('alpo_gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    reference_free = config.get('alpo_reference_free', False)
    ref_model = None
    if not reference_free:
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        ref_model.config.use_cache = False

    train_dataset = ALPODataset(dataset_path)
    data_collator = ALPODataCollator(
        tokenizer=tokenizer,
        cutoff_len=config.get('alpo_cutoff_len', 8192),
    )

    deepspeed_path = config.get('alpo_deepspeed')
    if deepspeed_path:
        if not osp.isabs(deepspeed_path):
            deepspeed_path = osp.join(dirname, deepspeed_path)
        if not osp.exists(deepspeed_path):
            deepspeed_path = None

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=config.get('alpo_batch_size_per_gpu', 1),
        gradient_accumulation_steps=config.get('alpo_gradient_accumulation_steps', 1),
        learning_rate=config.get('alpo_lr', config.get('dpo_lr', 1e-6)),
        num_train_epochs=config.get('alpo_epochs', config.get('dpo_epochs', 1)),
        lr_scheduler_type=config.get('alpo_lr_scheduler_type', 'cosine'),
        warmup_ratio=config.get('alpo_warmup_ratio', 0.1),
        logging_steps=config.get('alpo_logging_steps', 10),
        save_steps=config.get('alpo_save_steps', 5000000),
        save_total_limit=config.get('alpo_save_total_limit', 1),
        bf16=config.get('alpo_bf16', True),
        remove_unused_columns=False,
        report_to=[],
        optim=config.get('alpo_optim', 'adamw_torch'),
        deepspeed=deepspeed_path,
        ddp_timeout=config.get('alpo_ddp_timeout', 180000000),
    )

    if ref_model is not None:
        ref_model.to(training_args.device)

    trainer = ALPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        beta_scale=config.get('alpo_beta_scale', 1.0),
        average_log_prob=config.get('alpo_average_log_prob', False),
        reference_free=reference_free,
        disable_shuffling=config.get('alpo_disable_shuffling', False),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
