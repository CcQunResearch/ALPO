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
from prompt_template import translation_template
from utils import lang_dict, get_sft_prompt
from generation_check import check_quality, extract_tr_from_prompt, extract_zh_from_prompt
from template_utils import apply_model_chat_template, get_generation_stop_token_id

if __name__ == '__main__':
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    tr_model = config['tr_model']
    trpe_model = config['trpe_model']
    sft_model = config['sft_model']
    model_path = config['model_path']
    port = config['port']
    lang = config['lang']
    gpus = config['gpus']
    retry_num = config['retry_num']
    sft_proportion = config['sft_proportion']
    dpo_mode = config['dpo_mode']
    dpo_finetuning_type = config['dpo_finetuning_type']
    alignment_method = config.get('alignment_method', 'dpo')
    alpo_finetuning_type = config.get('alpo_finetuning_type', dpo_finetuning_type)
    src_lang_str = lang_dict[lang.split('2')[0]]
    lang_str = lang_dict[lang.split('2')[1]]
    segment_step = 5
    dataset_path = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'translation_test_{tr_model}_{trpe_model}_{lang}.json')
    if sft_proportion == 1.0:
        output_name = f'sft_{sft_model}_{tr_model}_{trpe_model}'
        aligned_finetuning_type = 'full'
    elif alignment_method == 'alpo':
        output_name = f'alpo_{sft_model}_{tr_model}_{trpe_model}_{sft_proportion}_{alpo_finetuning_type}'
        aligned_finetuning_type = alpo_finetuning_type
    else:
        output_name = f'dpo_{sft_model}_{tr_model}_{trpe_model}_{dpo_mode}_{sft_proportion}_{dpo_finetuning_type}'
        aligned_finetuning_type = dpo_finetuning_type
    output_dir = osp.join(dirname, '..', 'Inference', lang, output_name)
    sft_name = 'sft_default' if sft_proportion == 1.0 else f'sft_{sft_proportion}'
    init_model = osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, sft_name)
    if sft_proportion == 1.0:
        infer_model = 'sft_default'
    elif alignment_method == 'alpo':
        infer_model = f'alpo_{sft_proportion}_{alpo_finetuning_type}'
    else:
        infer_model = f'dpo_{dpo_mode}_{sft_proportion}_{dpo_finetuning_type}'
    model_name = osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, infer_model)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = init_model if sft_proportion != 1.0 and aligned_finetuning_type == 'lora' else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    stop_token_id = get_generation_stop_token_id(tokenizer, sft_model)

    print('Tranlate test dataset...')
    dataset = json.load(open(dataset_path, 'r', encoding='utf-8'))
    prompts = [apply_model_chat_template(tokenizer, item['instruction'], sft_model) for item in dataset]
    if sft_proportion != 1.0 and aligned_finetuning_type == 'lora':
        results = infer(prompts, gpus, init_model, 1300, stop_token_id, temperature=0.5, top_p=0.7, top_k=30, lora_name=model_name)
    else:
        results = infer(prompts, gpus, model_name, 1300, stop_token_id, temperature=0.5, top_p=0.7, top_k=30)
    
    generated_predictions = []
    for prompt, result in zip(prompts, results):
        generated_predictions.append({"prompt": prompt, "predict": result})
    print('Tranlate test dataset done.')
    
    print('Process exception prompts...')
    exception_indexes = []
    for i, prediction in enumerate(generated_predictions):
        response = prediction["predict"]
        if not check_quality(prediction["prompt"], response):
            exception_indexes.append(i)
    
    retry_indexes = []
    retry_prompts = []
    for exception_index in exception_indexes:
        prompt = generated_predictions[exception_index]["prompt"]
        proper_noun_content = '\n'.join(extract_tr_from_prompt(prompt))
        dialogue_list = extract_zh_from_prompt(prompt)
        dialogue_content = '\n'.join([f'{i+1}. {s}' for i, s in enumerate(dialogue_list)])
        
        for _ in range(retry_num):
            retry_indexes.append(exception_index)
            retry_instruction = get_sft_prompt(src_lang_str, lang_str, translation_template, proper_noun_content, dialogue_content)
            retry_prompts.append(apply_model_chat_template(tokenizer, retry_instruction, sft_model))
    if sft_proportion != 1.0 and aligned_finetuning_type == 'lora':
        retry_results = infer(retry_prompts, gpus, init_model, 1300, stop_token_id, temperature=0.5, top_p=0.7, top_k=30, lora_name=model_name)
    else:
        retry_results = infer(retry_prompts, gpus, model_name, 1300, stop_token_id, temperature=0.5, top_p=0.7, top_k=30)
    
    pass_indexes = []
    for retry_index, retry_prompt, retry_result in zip(retry_indexes, retry_prompts, retry_results):
        if retry_index in pass_indexes:
            continue
        if check_quality(retry_prompt, retry_result):
            generated_predictions[retry_index]["predict"] = retry_result
            pass_indexes.append(retry_index)
    exception_indexes = [i for i in exception_indexes if i not in pass_indexes]
    print('Process exception prompts done.')
    
    print('Final process exception with segment...')
    segment_indexes = []
    segment_prompts = []
    for exception_index in exception_indexes:
        prompt = generated_predictions[exception_index]["prompt"]
        proper_noun_content = '\n'.join(extract_tr_from_prompt(prompt))
        dialogue_list = extract_zh_from_prompt(prompt)
        dialogue_content = '\n'.join([f'{i+1}. {s}' for i, s in enumerate(dialogue_list)])
        curr = 0
        lines = dialogue_content.split('\n')
        while curr < len(lines):
            segment_dialogue = '\n'.join(lines[curr:curr+segment_step])
            segment_indexes.append(exception_index)
            segment_instruction = get_sft_prompt(src_lang_str, lang_str, translation_template, proper_noun_content, segment_dialogue)
            segment_prompts.append(apply_model_chat_template(tokenizer, segment_instruction, sft_model))
            curr += segment_step
    if sft_proportion != 1.0 and aligned_finetuning_type == 'lora':
        segment_results = infer(segment_prompts, gpus, init_model, 300, stop_token_id, temperature=0.5, top_p=0.7, top_k=30, lora_name=model_name)       
    else:
        segment_results = infer(segment_prompts, gpus, model_name, 300, stop_token_id, temperature=0.5, top_p=0.7, top_k=30)       
    
    segment_dict = {}
    for segment_index, segment_prompt, segment_result in zip(segment_indexes, segment_prompts, segment_results):
        if segment_index not in segment_dict:
            segment_dict[segment_index] = {'success_tag': True, 'results': []}
        if not check_quality(segment_prompt, segment_result):
            segment_dict[segment_index]['success_tag'] = False
        segment_dict[segment_index]['results'].append(segment_result.strip())
    
    exception_indexes = []  
    for exception_index in segment_dict:
        if not segment_dict[exception_index]['success_tag']:
            exception_indexes.append(exception_index)
            continue
        response = '\n'.join(segment_dict[exception_index]['results'])
        generated_predictions[exception_index]["predict"] = response
    print('Final process exception with segment done.')
    
    with open(osp.join(output_dir, 'generated_predictions.jsonl'), 'w', encoding='utf-8') as f:
        for item in generated_predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n') 
    final_exception_indexes_json = osp.join(output_dir, 'final_exception_indexes.json')
    json.dump(exception_indexes, open(final_exception_indexes_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
