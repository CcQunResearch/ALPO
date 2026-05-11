import requests
import json
import yaml
import time
import random
import os
import os.path as osp
from tqdm import tqdm
from vllm_infer import infer
from transformers import AutoTokenizer
from template_utils import apply_model_chat_template, get_generation_stop_token_id

if __name__ == '__main__':
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    tr_model = config['tr_model']
    trpe_model = config['trpe_model']
    model_path = config['model_path']
    port = config['port']
    lang = config['lang']
    gpus = config['gpus']
    dataset_path = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'term_recognition_test_{tr_model}_{trpe_model}_{lang}.json')
    output_dir = osp.join(dirname, '..', 'LLaMAFactory', 'TermRecognition', 'test', f'{tr_model}_{trpe_model}', lang)
    model_name = osp.join(model_path, 'llamafactory', f'{tr_model}_{trpe_model}', lang, 'tr_default')
    os.makedirs(output_dir, exist_ok=True)

    print('Identify test dataset terms...')
    dataset = json.load(open(dataset_path, 'r', encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts = [apply_model_chat_template(tokenizer, item['instruction'], tr_model) for item in dataset]
    stop_token_id = get_generation_stop_token_id(tokenizer, tr_model)
    results = infer(prompts, gpus, model_name, 300, stop_token_id, temperature=0.2, top_p=0.3, top_k=25)
    
    generated_predictions = []
    for prompt, result in zip(prompts, results):
        generated_predictions.append({"prompt": prompt, "predict": result})
    with open(osp.join(output_dir, 'generated_predictions.jsonl'), 'w', encoding='utf-8') as f:
        for item in generated_predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print('Identify test dataset terms done.')
