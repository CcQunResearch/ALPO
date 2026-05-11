import os
import os.path as osp
import random
import yaml
import re
import json
from tqdm import tqdm
from prompt_template import translation_template, proper_noun_slot_dict
from utils import lang_dict, extract_training_queries_and_responses
from yaml_config import save_sft_train_config


if __name__ == '__main__':
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    sft_model = config['sft_model']
    trpe_model = config['trpe_model']
    lang = config['lang']
    model_path = config['model_path']
    src_lang_str = lang_dict[lang.split('2')[0]]
    lang_str = lang_dict[lang.split('2')[1]]
    sft_proportion = config['sft_proportion']
    gpus = config['gpus']
    sft_global_batch_size = config['sft_global_batch_size']
    sft_lr = config['sft_lr']
    sft_epochs = config['sft_epochs']
    info_dir = osp.join(dirname, 'info', lang)
    dialogue_file_names = sorted([file for file in os.listdir(osp.join(info_dir, 'episode_results')) if 'json' in file])

    print('[3-1] Construct sft queries and responses...')
    all_data = []
    all_audio_complete_tag = []
    all_fewshot = []
    for dialogue_file_name in tqdm(dialogue_file_names):
        play_name = dialogue_file_name.strip('.json')
        episode_result = json.load(open(osp.join(info_dir, 'episode_results', dialogue_file_name), 'r', encoding='utf-8'))
        proper_noun_dict = json.load(open(osp.join(info_dir, 'proper_noun', f'{play_name}_filter.json'), 'r', encoding='utf-8'))
        pn_identify_dict = json.load(open(osp.join(info_dir, 'proper_noun', f'{play_name}_identify.json'), 'r', encoding='utf-8'))
        play_data, pn_fewshot = extract_training_queries_and_responses(play_name, episode_result, proper_noun_dict, translation_template, src_lang_str,
                                                                       lang_str, pn_identify_dict=pn_identify_dict, pn_consis=True)
        all_data += play_data
        all_fewshot += pn_fewshot
        
    if len(all_fewshot) == 0:
        all_fewshot.append({"dialogue": re.sub(r'\([^)]*\)', '', proper_noun_slot_dict[lang][-2]), "term": proper_noun_slot_dict[lang][-1]})
    print('[3-1] Construct sft queries and responses done.')

    fewshot_path_json = osp.join(info_dir, f'proper_noun_fewshot_{trpe_model}_{lang}.json')
    fewshot_file_json = open(fewshot_path_json, 'w', encoding='utf-8')
    json.dump(all_fewshot, fewshot_file_json, ensure_ascii=False, indent=4)

    random.seed(1024)
    random.shuffle(all_data)
    seprate_index = int(len(all_data) * sft_proportion)
    sft_data = all_data[:seprate_index]
    dataset_name = f'translation_train_{trpe_model}_{lang}' if sft_proportion == 1.0 else f'translation_train_{trpe_model}_{lang}_{sft_proportion}'
    save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'{dataset_name}.json')
    save_file_json = open(save_path_json, 'w', encoding='utf-8')
    json.dump(sft_data, save_file_json, ensure_ascii=False, indent=4)

    if sft_proportion != 1.0:
        alignment_data = all_data[seprate_index:]
        alignment_dataset_name = f'alignment_raw_{trpe_model}_{lang}_{sft_proportion}'
        alignment_save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'VividnessAlignment', f'{alignment_dataset_name}.json')
        alignment_save_file_json = open(alignment_save_path_json, 'w', encoding='utf-8')
        json.dump(alignment_data, alignment_save_file_json, ensure_ascii=False, indent=4)
    
    gpu_num = len(gpus.split(','))
    if sft_model in ['Qwen2.5-32B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen2.5-32B', 'Qwen2.5-72B']:
        batch_size_per_gpu = 1
    elif sft_model in ['Meta-Llama-3.1-8B', 'Meta-Llama-3.1-8B-Instruct']:
        batch_size_per_gpu = 12
    else:
        batch_size_per_gpu = 3
    gas = sft_global_batch_size // (batch_size_per_gpu * gpu_num)
    assert sft_global_batch_size == batch_size_per_gpu * gpu_num * gas, \
        "global_batch_size must be divisible by train_micro_batch_size_per_gpu * gpu_count"
    save_sft_train_config(model_path, sft_model, trpe_model, lang, batch_size_per_gpu, gas, sft_lr, sft_epochs, sft_proportion)