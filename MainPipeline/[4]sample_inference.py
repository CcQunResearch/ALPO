import os.path as osp
import os
import yaml
import json
from vllm_infer import infer
from utils import lang_dict
from transformers import AutoTokenizer
from generation_check import *
from template_utils import apply_model_chat_template, get_generation_stop_token_id

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
    gpus = config['gpus']
    
    output_dir = osp.join(dirname, '..', 'LLaMAFactory', 'VividnessAlignment', lang, f'sample_{sft_model}_{trpe_model}_{sft_proportion}')
    os.makedirs(output_dir, exist_ok=True)
    
    if sft_proportion != 1.0:
        print('[4] Sample alignment data...')
        sample_save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'sample_{trpe_model}_{lang}_{sft_proportion}.json')
        sample_dataset = json.load(open(sample_save_path_json, 'r', encoding='utf-8'))
        model_name = osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, f'sft_{sft_proportion}')
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prompts = [apply_model_chat_template(tokenizer, sample_data['instruction'], sft_model) for sample_data in sample_dataset]
        stop_token_id = get_generation_stop_token_id(tokenizer, sft_model)
        results = infer(prompts, gpus, model_name, 650, stop_token_id, temperature=0.7, top_p=0.9, top_k=40)
        print('[4] Sample alignment data done.')
        
        generated_predictions = []
        for prompt, result in zip(prompts, results):
            generated_predictions.append({"prompt": prompt, "predict": result})
        with open(osp.join(output_dir, 'generated_predictions.jsonl'), 'w', encoding='utf-8') as f:
            for item in generated_predictions:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
