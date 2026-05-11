import os
import os.path as osp
import yaml
import json
import random
from utils import lang_dict, get_sft_prompt
from prompt_template import translation_template
from yaml_config import save_sample_config

def extract_sample_dataset(alignment_raw_data, sample_num, src_lang_str, lang_str):
    sample_dataset = []
    for i, raw_data in enumerate(alignment_raw_data):
        dialogue_content = '\n'.join([f'{i+1}. {s}' for i, s in enumerate(raw_data['source'])])
        full_proper_noun_content = raw_data['proper noun']
        prompts = [get_sft_prompt(src_lang_str, lang_str, translation_template, full_proper_noun_content, dialogue_content) for _ in range(sample_num)]
        for prompt in prompts:
            sample_dataset.append({"instruction": prompt, "input": None, "output": None, "raw index": i})
    return sample_dataset

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
    sample_num = config['sample_num']
    temperature = config['temperature']
    top_p = config['top_p']
    top_k = config['top_k']
    
    if sft_proportion != 1.0:
        print('[3-2] Construct sample dataset...')
        alignment_save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'VividnessAlignment', f'alignment_raw_{trpe_model}_{lang}_{sft_proportion}.json')
        alignment_raw_data = json.load(open(alignment_save_path_json, 'r', encoding='utf-8'))
        sample_dataset = extract_sample_dataset(alignment_raw_data, sample_num, src_lang_str, lang_str)
        save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'sample_{trpe_model}_{lang}_{sft_proportion}.json')
        save_file_json = open(save_path_json, 'w', encoding='utf-8')
        json.dump(sample_dataset, save_file_json, ensure_ascii=False, indent=4)
        save_sample_config(model_path, sft_model, trpe_model, lang, sft_proportion, temperature, top_p, top_k)
        print('[3-2] Construct sample dataset done.')