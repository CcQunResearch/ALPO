import os.path as osp
import yaml
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import lang_dict
from template_utils import apply_model_chat_template, get_generation_stop_text
from yaml_config import save_dpo_config


if __name__ == '__main__':
    random.seed(1024)
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    sft_model = config['sft_model']
    trpe_model = config['trpe_model']
    lang = config['lang']
    model_path = config['model_path']
    sft_proportion = config['sft_proportion']
    sample_num = config['sample_num']
    temperature = config['temperature']
    label_preference_model = config['label_preference_model']
    label_preference_vote_num = config['label_preference_vote_num']
    label_preference_retry_num = config['label_preference_retry_num']
    label_preference_num_workers = config['label_preference_num_workers']
    top_p = config['top_p']
    top_k = config['top_k']
    dpo_mode = config['dpo_mode']
    gpus = config['gpus']
    dpo_global_batch_size = config['dpo_global_batch_size']
    dpo_lr = config['dpo_lr']
    dpo_epochs = config['dpo_epochs']
    dpo_finetuning_type = config['dpo_finetuning_type']
    
    src_lang = lang_dict[lang.split('2')[0]]
    src_lang = '中文' if src_lang == '简体中文' else src_lang
    tgt_lang = lang_dict[lang.split('2')[1]]
    tgt_lang = '中文' if tgt_lang == '简体中文' else tgt_lang

    output_dir = osp.join(dirname, '..', 'LLaMAFactory', 'VividnessAlignment', lang, f'sample_{sft_model}_{trpe_model}_{sft_proportion}')
    
    if sft_proportion != 1.0:
        direct_sample_save_path_json = osp.join(output_dir, 'direct_sample_dataset.json')
        direct_sample_dataset = json.load(open(direct_sample_save_path_json, 'r', encoding='utf-8'))
        random.seed(1024)
        random.shuffle(direct_sample_dataset)
        
        all_preference_data = []
        if dpo_mode == 'outcome':
            print('[6] Construct outcome dpo dataset...')
            for direct_sample in tqdm(direct_sample_dataset):
                preference_data = {
                    "instruction": direct_sample['instruction'],
                    "input": None,
                    "chosen": None,
                    "rejected": None
                }
                source = []
                chosen = []
                rejected = []
                for segment in direct_sample['segment data']:
                    source += segment['segment source']
                    segment_targets = segment['segment targets']
                    if 'label model' in segment:
                        average_scores = segment['vividness scores']['average']
                        if len(average_scores) >= 8:
                            top3_indices = sorted(range(len(average_scores)), key=lambda i: average_scores[i], reverse=True)[:3]
                            score_max_index = min(top3_indices, key=lambda i: len(segment_targets[i]))
                            min_value = sorted(average_scores)[2]
                        else:
                            score_max_index = average_scores.index(max(average_scores))
                            min_value = min(average_scores)

                        score_min_index = average_scores.index(min_value)
                        chosen += segment_targets[score_max_index].split('\n')
                        rejected += segment_targets[score_min_index].split('\n')
                    else:
                        random_chosen_index = random.randint(0, len(segment_targets) - 1)
                        chosen += segment_targets[random_chosen_index].split('\n')
                        rejected += segment_targets[random_chosen_index].split('\n')
                
                assert len(source) == len(chosen) == len(rejected), 'source, chosen and rejected should have the same length'
                chosen_response = '\n'.join([f'{i+1}. {t}' for i, t in enumerate(chosen)])
                rejected_response = '\n'.join([f'{i+1}. {t}' for i, t in enumerate(rejected)])
                preference_data['chosen'] = chosen_response
                preference_data['rejected'] = rejected_response
                all_preference_data.append(preference_data)
            print('[6] Construct outcome dpo dataset done.')
        elif dpo_mode == 'segment':
            sft_model_path = osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, f'sft_{sft_proportion}')
            tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
            stop_token = get_generation_stop_text(tokenizer, sft_model)
            print('[6] Construct segment dpo dataset...')
            for direct_sample in tqdm(direct_sample_dataset):
                source = []
                chosen = []
                rejected = []
                prefix_instruction = apply_model_chat_template(tokenizer, direct_sample['instruction'], sft_model)
                for i, segment in enumerate(direct_sample['segment data']):
                    if 'label model' in segment:
                        average_scores = segment['vividness scores']['average']
                        segment_targets = segment['segment targets']  
                        if len(average_scores) >= 8:
                            top3_indices = sorted(range(len(average_scores)), key=lambda i: average_scores[i], reverse=True)[:3]
                            score_max_index = min(top3_indices, key=lambda i: len(segment_targets[i]))
                            min_value = sorted(average_scores)[2]
                        else:
                            score_max_index = average_scores.index(max(average_scores))
                            min_value = min(average_scores)

                        score_min_index = average_scores.index(min_value)
                        postfix = '\n' if i != len(direct_sample['segment data']) - 1 else stop_token
                        segment_source = segment['segment source']
                        segment_chosen = segment_targets[score_max_index].split('\n')
                        segment_rejected = segment_targets[score_min_index].split('\n')
                        
                        prefix_response = '\n'.join([f'{i+1}. {t}' for i, t in enumerate(chosen)]) + '\n'
                        segment_chosen_response = '\n'.join([f'{i+1}. {t}' for i, t in zip(list(range(len(chosen), len(chosen) + len(segment_chosen))), segment_chosen)]) + postfix
                        segment_rejected_response = '\n'.join([f'{i+1}. {t}' for i, t in zip(list(range(len(chosen), len(chosen) + len(segment_rejected))), segment_rejected)]) + postfix
                        preference_data = {
                            "instruction": prefix_instruction + prefix_response,
                            "input": None,
                            "chosen": segment_chosen_response,
                            "rejected": segment_rejected_response
                        }
                        all_preference_data.append(preference_data)
                    
                        source += segment_source
                        chosen += segment_chosen
                        rejected += segment_rejected         
            print('[6] Construct segment dpo dataset done.')
        
        save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'dpo_train_{sft_model}_{trpe_model}_{lang}_{dpo_mode}_{sft_proportion}.json')
        save_file_json = open(save_path_json, 'w', encoding='utf-8')
        json.dump(all_preference_data, save_file_json, ensure_ascii=False, indent=4)
        
        gpu_num = len(gpus.split(','))
        batch_size_per_gpu = 1 if sft_model in ['Qwen2.5-32B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen2.5-32B', 'Qwen2.5-72B'] else 3
        gas = dpo_global_batch_size // (batch_size_per_gpu * gpu_num)
        assert dpo_global_batch_size == batch_size_per_gpu * gpu_num * gas, \
            "global_batch_size must be divisible by train_micro_batch_size_per_gpu * gpu_count"
        save_dpo_config(model_path, sft_model, trpe_model, lang, batch_size_per_gpu, gas, dpo_lr, dpo_epochs, dpo_mode, sft_proportion, dpo_finetuning_type)
