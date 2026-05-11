import yaml
import json
import os.path as osp
from prompt_template import template_dict


def save_pn_pe_config(model_path, trpe_model, lang):
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'r', encoding='utf-8')
    dataset_info = json.load(dataset_info_file)
    
    dataset_name = f'proper_noun_queries_train_{lang}'
    if dataset_name not in dataset_info:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": None,
                "history": None
            }
        }
    
    per_device_eval_batch_size = 2 if trpe_model in ['Qwen2.5-32B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen2.5-32B', 'Qwen2.5-72B'] else 5
    config = {
        'model_name_or_path': osp.join(model_path, 'vanilla', trpe_model),
        'stage': 'sft',
        'do_predict': True,
        'finetuning_type': 'full',
        'eval_dataset': dataset_name,
        'template': template_dict[trpe_model],
        'cutoff_len': 8192,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'output_dir': f'TermRecognition/train/{trpe_model}/{lang}',
        'overwrite_output_dir': True,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'predict_with_generate': True
    }
    
    api_config = {
        'model_name_or_path': osp.join(model_path, 'vanilla', trpe_model),
        'template': template_dict[trpe_model],
        'infer_backend': 'vllm'
    }
    
    with open(osp.join('..','LLaMAFactory','TermRecognition','train',trpe_model,lang,'predict.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    
    with open(osp.join('..','LLaMAFactory','TermRecognition','train',trpe_model,lang,'api.yaml'), 'w') as file:
        yaml.dump(api_config, file, default_flow_style=False, allow_unicode=True)
        
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'w', encoding='utf-8')
    json.dump(dataset_info, dataset_info_file, ensure_ascii=False, indent=4)
    
def save_pn_infer_config(model_path, tr_model, trpe_model,lang):
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'r', encoding='utf-8')
    dataset_info = json.load(dataset_info_file)
    
    dataset_name = f'term_recognition_test_{tr_model}_{trpe_model}_{lang}'
    if dataset_name not in dataset_info:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": None,
                "history": None
            }
        }
    
    per_device_eval_batch_size = 2 if tr_model in ['Qwen2.5-32B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen2.5-32B', 'Qwen2.5-72B'] else 5
    config = {
        'model_name_or_path': osp.join(model_path, 'llamafactory', f'{tr_model}_{trpe_model}', lang, 'tr_default'),
        'stage': 'sft',
        'do_predict': True,
        'finetuning_type': 'full',
        'eval_dataset': dataset_name,
        'template': template_dict[tr_model],
        'cutoff_len': 8192,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'output_dir': f'TermRecognition/test/{tr_model}_{trpe_model}/{lang}',
        'overwrite_output_dir': True,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'predict_with_generate': True
    }
    
    api_config = {
        'model_name_or_path': osp.join(model_path, 'llamafactory', f'{tr_model}_{trpe_model}', lang, 'tr_default'),
        'template': template_dict[tr_model]
    }
    
    with open(osp.join('..','LLaMAFactory','TermRecognition','test', f'{tr_model}_{trpe_model}', lang, 'predict.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

    with open(osp.join('..','LLaMAFactory','TermRecognition','test', f'{tr_model}_{trpe_model}', lang, 'api.yaml'), 'w') as file:
        yaml.dump(api_config, file, default_flow_style=False, allow_unicode=True)  
        
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'w', encoding='utf-8')
    json.dump(dataset_info, dataset_info_file, ensure_ascii=False, indent=4)
    
def save_sft_infer_config(model_path, sft_model, tr_model, trpe_model, lang):
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'r', encoding='utf-8')
    dataset_info = json.load(dataset_info_file)
    
    dataset_name = f'translation_test_{tr_model}_{trpe_model}_{lang}'
    if dataset_name not in dataset_info:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": None,
                "history": None
            }
        }
    
    per_device_eval_batch_size = 2 if sft_model in ['Qwen2.5-32B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen2.5-32B', 'Qwen2.5-72B'] else 5
    config = {
        'model_name_or_path': osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, 'sft_default'),
        'stage': 'sft',
        'do_predict': True,
        'finetuning_type': 'full',
        'eval_dataset': dataset_name,
        'template': template_dict[sft_model],
        'cutoff_len': 8192,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'output_dir': f'../MainPipeline/info/{lang}/inference/test_{sft_model}_{tr_model}_{trpe_model}',
        'overwrite_output_dir': True,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'predict_with_generate': True
    }
    
    api_config = {
        'model_name_or_path': osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, 'sft_default'),
        'template': template_dict[sft_model]
    }
    
    with open(osp.join('..','LLaMAFactory', 'YamlConfig', f'predict_{sft_model}_{tr_model}_{trpe_model}_{lang}.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    
    with open(osp.join('..','LLaMAFactory', 'YamlConfig', f'api_{sft_model}_{tr_model}_{trpe_model}_{lang}.yaml'), 'w') as file:
        yaml.dump(api_config, file, default_flow_style=False, allow_unicode=True)
        
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'w', encoding='utf-8')
    json.dump(dataset_info, dataset_info_file, ensure_ascii=False, indent=4)
    
def save_sample_config(model_path, sft_model, trpe_model, lang, sft_proportion, temperature, top_p, top_k):
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'r', encoding='utf-8')
    dataset_info = json.load(dataset_info_file)
    
    dataset_name = f'sample_{trpe_model}_{lang}_{sft_proportion}'
    if dataset_name not in dataset_info:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": None,
                "history": None
            }
        }
    
    per_device_eval_batch_size = 2 if sft_model in ['Qwen2.5-32B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen2.5-32B', 'Qwen2.5-72B'] else 5
    config = {
        'model_name_or_path': osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, f'sft_{sft_proportion}'),
        'stage': 'sft',
        'do_predict': True,
        'finetuning_type': 'full',
        'eval_dataset': dataset_name,
        'template': template_dict[sft_model],
        'cutoff_len': 8192,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'output_dir': f'./VividnessAlignment/{lang}/sample_{sft_model}_{trpe_model}_{sft_proportion}',
        'overwrite_output_dir': True,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'predict_with_generate': True,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k
    }
    
    api_config = {
        'model_name_or_path': osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, f'sft_{sft_proportion}'),
        'template': template_dict[sft_model],
        'infer_backend': 'vllm',
        'temperature': temperature
    }
    
    with open(osp.join('..','LLaMAFactory', 'YamlConfig', f'sample_{sft_model}_{trpe_model}_{lang}_{sft_proportion}.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        
    with open(osp.join('..','LLaMAFactory', 'YamlConfig', f'sample_api_{sft_model}_{trpe_model}_{lang}_{sft_proportion}.yaml'), 'w') as file:
        yaml.dump(api_config, file, default_flow_style=False, allow_unicode=True)
        
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'w', encoding='utf-8')
    json.dump(dataset_info, dataset_info_file, ensure_ascii=False, indent=4)
    
def save_sft_train_config(model_path, sft_model, trpe_model, lang, batch_size_per_gpu, gas, lr, num_epochs, sft_proportion=1.0):
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'r', encoding='utf-8')
    dataset_info = json.load(dataset_info_file)
    
    dataset_name = f'translation_train_{trpe_model}_{lang}' if sft_proportion == 1.0 else f'translation_train_{trpe_model}_{lang}_{sft_proportion}'
    if dataset_name not in dataset_info:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": None,
                "history": None
            }
        }
    
    output_name = 'sft_default' if sft_proportion == 1.0 else f'sft_{sft_proportion}'
    config = {
        'model_name_or_path': osp.join(model_path, 'vanilla', sft_model),
        'trust_remote_code': True,
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'full',
        'deepspeed': 'config/deepspeed/ds_z3_config.json',
        'dataset': dataset_name,
        'template': template_dict[sft_model],
        'cutoff_len': 8192,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'output_dir': osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, output_name),
        'logging_steps': 10,
        'save_steps': 5000000,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'per_device_train_batch_size': batch_size_per_gpu,
        'gradient_accumulation_steps': gas,
        'learning_rate': lr,
        'num_train_epochs': num_epochs,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'bf16': True,
        'ddp_timeout': 180000000,
        'val_size': 0.03,
        'per_device_eval_batch_size': 4,
        'eval_strategy': 'steps',
        'eval_steps': 200
    }
    
    yaml_name = f'sft_{sft_model}_{trpe_model}_{lang}.yaml' if sft_proportion == 1.0 else f'sft_{sft_model}_{trpe_model}_{lang}_{sft_proportion}.yaml'
    with open(osp.join('..','LLaMAFactory', 'YamlConfig', yaml_name), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'w', encoding='utf-8')
    json.dump(dataset_info, dataset_info_file, ensure_ascii=False, indent=4)
    
    
def save_tr_train_config(model_path, tr_model,trpe_model, lang, batch_size_per_gpu, gas, lr, num_epochs):
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'r', encoding='utf-8')
    dataset_info = json.load(dataset_info_file)
    
    dataset_name = f'term_recognition_train_{trpe_model}_{lang}'
    if dataset_name not in dataset_info:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": None,
                "history": None
            }
        }
    
    # train_epochs少一个，容易过拟合
    config = {
        'model_name_or_path': osp.join(model_path, 'vanilla', tr_model),
        'trust_remote_code': True,
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'full',
        'deepspeed': 'config/deepspeed/ds_z3_config.json',
        'dataset': dataset_name,
        'template': template_dict[tr_model],
        'cutoff_len': 8192,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'output_dir': osp.join(model_path, 'llamafactory', f'{tr_model}_{trpe_model}', lang, 'tr_default'),
        'logging_steps': 10,
        'save_steps': 5000000,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'per_device_train_batch_size': batch_size_per_gpu,
        'gradient_accumulation_steps': gas,
        'learning_rate': lr,
        'num_train_epochs': num_epochs - 1,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'bf16': True,
        'ddp_timeout': 180000000,
        'val_size': 0.03,
        'per_device_eval_batch_size': 4,
        'eval_strategy': 'steps',
        'eval_steps': 200
    }
    
    with open(osp.join('..','LLaMAFactory', 'YamlConfig', f'tr_{tr_model}_{trpe_model}_{lang}.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'w', encoding='utf-8')
    json.dump(dataset_info, dataset_info_file, ensure_ascii=False, indent=4)
    
def save_dpo_config(model_path, sft_model, trpe_model, lang, batch_size_per_gpu, gas, lr, num_epochs, dpo_mode, sft_proportion=1.0, dpo_finetuning_type='full'):
    dataset_info_file = open(osp.join('..', 'LLaMAFactory','data', 'dataset_info.json'), 'r', encoding='utf-8')
    dataset_info = json.load(dataset_info_file)
    
    dataset_name = f'dpo_train_{sft_model}_{trpe_model}_{lang}_{dpo_mode}_{sft_proportion}'
    if dataset_name not in dataset_info:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
              "ranking": True,
              "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected"
              }
        }

    sft_name = 'sft_default' if sft_proportion == 1.0 else f'sft_{sft_proportion}'
    sft_model_path = osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, sft_name)
    output_name = f'dpo_{dpo_mode}_{sft_proportion}_{dpo_finetuning_type}'
    template = 'empty' if dpo_mode == 'segment' else template_dict[sft_model]
    config = {
        'model_name_or_path': sft_model_path,
        'trust_remote_code': True,
        'stage': 'dpo',
        'do_train': True,
        'finetuning_type': 'full',
        'deepspeed': 'config/deepspeed/ds_z3_config.json',
        'pref_loss': 'sigmoid',  # choices: [sigmoid (dpo), orpo, simpo]
        'dataset': dataset_name,
        'disable_shuffling': True,
        'template': template,
        'cutoff_len': 8192,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'output_dir': osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, output_name),
        'logging_steps': 10,
        'save_steps': 5000000,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'per_device_train_batch_size': batch_size_per_gpu,
        'gradient_accumulation_steps': gas,
        'learning_rate': lr,
        'num_train_epochs': num_epochs,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'bf16': True,
        'ddp_timeout': 180000000,
        'val_size': 0.03,
        'per_device_eval_batch_size': 4,
        'eval_strategy': 'steps',
        'eval_steps': 200
    }
    
    if dpo_finetuning_type == 'lora':
        config['finetuning_type'] = 'lora'
        config['lora_rank'] = 8
        config['lora_target'] = 'all'
        config['pref_beta'] = 0.1
    
    yaml_name = f'dpo_{sft_model}_{trpe_model}_{lang}_{dpo_mode}_{sft_proportion}_{dpo_finetuning_type}.yaml'
    with open(osp.join('..','LLaMAFactory', 'YamlConfig', yaml_name), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        
    dataset_info_file = open(osp.join('..','LLaMAFactory','data','dataset_info.json'), 'w', encoding='utf-8')
    json.dump(dataset_info, dataset_info_file, ensure_ascii=False, indent=4)