import os.path as osp
import os
import yaml
import json
import time
import random
import concurrent.futures
from tqdm import tqdm
from prompt_template import vividness_preference_template, vividness_fewshot
from utils import lang_dict, extract_local_sentence_indexes, csv_dict_reader, extract_first_json_from_text
from generation_check import check_quality, extract_tar
from chat import chat_ali

num_to_alpha = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E',
    6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O',
    16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
    21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y',
    26: 'Z'
}

alpha_to_num = {v: k for k, v in num_to_alpha.items()}

def extract_segment_and_context(segment_index, segment_original_line, all_source, context_len=5):
    begin_index = segment_original_line[segment_index[0]]
    end_index = segment_original_line[segment_index[-1]]
    prefix_context = [all_source[i] for i in range(max(0, begin_index - context_len), begin_index)] if begin_index > 0 else []
    segment_source = [all_source[segment_original_line[id]] for id in segment_index]
    suffix_context = [all_source[i] for i in range(end_index + 1, min(len(all_source), end_index + context_len + 1))] if end_index < len(all_source) - 1 else []
    return prefix_context, segment_source, suffix_context

def get_label_prompt(lang, src_lang, tgt_lang, prefix_context, segment_source, suffix_context):
    score_format = vividness_fewshot['score format']
    task_source = '\n'.join([f'[上下文] {text}' for text in prefix_context] + [f'[待评估] {text}' for text in segment_source] + [f'[上下文] {text}' for text in suffix_context])
    prompt_template = vividness_preference_template.format(src_lang, tgt_lang, src_lang, tgt_lang, src_lang, vividness_fewshot[lang], src_lang, task_source, tgt_lang, '<shuffled_task_translation>', score_format)
    return prompt_template


def check_score_format(d):
    keys = sorted(d.keys())
    expected_length = len(keys)
    expected_keys = [chr(ord('A') + i) for i in range(expected_length)]
    
    for key in keys:
        if len(key) != 1 or not key.isupper() or not key.isalpha():
            return False
    if keys != expected_keys:
        return False

    for value in d.values():
        if not isinstance(value, int) or value < 0 or value > 100:
            return False
    
    return True


def label_preference(prompt, model, num_translation, retry_num=3):
    retry_count = 0
    while retry_count < retry_num:
        try:
            response, _ = chat_ali(prompt, model, temperature=0.3)
        except Exception as e:
            print(f"{model}发生异常：{e}")
            retry_count += 1
            time.sleep(3)
        else:
            score_dict = extract_first_json_from_text(response)
            if score_dict is None:
                retry_count += 1
                time.sleep(3)
                continue
            if check_score_format(score_dict) and len(score_dict) == num_translation:
                score = list(score_dict.values())
                if len(set(score)) / len(score) < 0.25:
                    retry_count += 1
                    time.sleep(3)
                    continue
                
                return score
            else:
                retry_count += 1
                time.sleep(3)
                continue
    return None

def vote_preference(prompt_template, model, segment_targets, vote_num=3, retry_num=3):
    scores = []
    for _ in range(vote_num):
        targets_index = list(range(len(segment_targets)))
        random.shuffle(targets_index)
        translations = []
        for i in range(len(segment_targets)):
            segment_target = segment_targets[targets_index[i]]
            translation = f'译文{num_to_alpha[i+1]}：\n{segment_target}'
            translations.append(translation)
        task_translation = '\n\n'.join(translations)
        prompt = prompt_template.replace('<shuffled_task_translation>', task_translation)
        score = label_preference(prompt, model, len(segment_targets), retry_num=retry_num)
        
        if score is not None:
            score = [score[targets_index.index(i)] for i in list(range(len(segment_targets)))]
            
            scores.append(score)
    if len(scores) == 0:
        return None
    result_dict = {
        'scores': scores,
        'average': [round(sum(score) / len(scores), 2) for score in zip(*scores)]
    }
    return result_dict
    
    
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
    
    src_lang = lang_dict[lang.split('2')[0]]
    src_lang = '中文' if src_lang == '简体中文' else src_lang
    tgt_lang = lang_dict[lang.split('2')[1]]
    tgt_lang = '中文' if tgt_lang == '简体中文' else tgt_lang

    output_dir = osp.join(dirname, '..', 'LLaMAFactory', 'VividnessAlignment', lang, f'sample_{sft_model}_{trpe_model}_{sft_proportion}')
    
    if sft_proportion != 1.0:
        # load sample data - 10x
        sample_save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'sample_{trpe_model}_{lang}_{sft_proportion}.json')
        sample_dataset = json.load(open(sample_save_path_json, 'r', encoding='utf-8'))
        
        # load generated predictions - 10x
        generated_predictions = []
        with open(osp.join(output_dir, 'generated_predictions.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                generated_predictions.append(json.loads(line))
        
        # load raw data - 1x
        alignment_save_path_json = osp.join(dirname, '..', 'LLaMAFactory', 'VividnessAlignment', f'alignment_raw_{trpe_model}_{lang}_{sft_proportion}.json')
        alignment_raw_data = json.load(open(alignment_save_path_json, 'r', encoding='utf-8'))
        
        begin_index = 0
        cache_results_path = osp.join(output_dir, 'cache_preference_results.json')
        if osp.exists(cache_results_path):
            cache_preference_results = json.load(open(cache_results_path, 'r', encoding='utf-8'))
            begin_index = len(cache_preference_results)
        else:
            cache_preference_results = []
        
        print('[5] Construct raw index to samples...')
        raw_index_2_samples = {}
        for sample_data, prediction in zip(sample_dataset, generated_predictions):
            raw_index = sample_data['raw index']
            if raw_index not in raw_index_2_samples:
                raw_index_2_samples[raw_index] = []
            if check_quality(sample_data['instruction'], prediction['predict'], print_log=False):
                raw_index_2_samples[raw_index].append(prediction['predict'])
        print('[5] Construct raw index to samples done.')
        
        print('[5] Prepare preference label data...')
        direct_sample_dataset = []
        ready_to_label_dataset = []
        
        di = 0
        for i, raw_data in enumerate(tqdm(alignment_raw_data)):
            sample_predictions = raw_index_2_samples[i]
            direct_sample_data = {
                'instruction': raw_data['instruction'],
                'do train': True,
                'segment data': []
            }
            
            if len(sample_predictions) == 0:
                continue
            
            source = raw_data['source']
            target = raw_data['target']
            source_sentence_label = raw_data['source sentence label']
            segment_indexes = extract_local_sentence_indexes(source_sentence_label)
            
            source_file_path = osp.join(dirname, '..', 'Data', 'Source', lang, 'train', raw_data['play name'], f'{raw_data["play name"]} {raw_data["episode"]}_{lang_dict[lang.split("2")[0]]}.csv')
            source_data_dict = csv_dict_reader(source_file_path)
            
            for si, segment_index in enumerate(segment_indexes):
                prefix_context, segment_source, suffix_context = extract_segment_and_context(segment_index, raw_data['original line'], source_data_dict['Text'], context_len=7) # list of list
                segment_targets = ['\n'.join([target[id] for id in segment_index])] # list of str
                for sample_prediction in sample_predictions:
                    segment_target = '\n'.join([extract_tar(sample_prediction)[id] for id in segment_index])
                    if segment_target not in segment_targets:
                        segment_targets.append(segment_target)
                segment_data = {
                    'prefix context': prefix_context,
                    'segment source': segment_source,
                    'suffix context': suffix_context,
                    'segment targets': segment_targets
                }
                direct_sample_data['segment data'].append(segment_data)
                
                if len(segment_targets) > 1:
                    prompt_template = get_label_prompt(lang, src_lang, tgt_lang, prefix_context, segment_source, suffix_context)
                    ready_to_label_dataset.append({'data index': di, 'segment index': si, 'prompt template': prompt_template, 'segment targets': segment_targets})
                
            direct_sample_dataset.append(direct_sample_data)
            di += 1
        print('[5] Prepare preference label data done.')
        
        print('[5] Label vividness preference...')
        tasks = [[r2l['prompt template'], label_preference_model, r2l['segment targets'], label_preference_vote_num, label_preference_retry_num] for r2l in ready_to_label_dataset]
        
        tasks = tasks[begin_index:]
        split_step = 2000
        all_split_tasks = [tasks[i:i+split_step] for i in range(0, len(tasks), split_step)]
        results = cache_preference_results
        for split_tasks in all_split_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=label_preference_num_workers) as executor:
                futures = [executor.submit(vote_preference, *para) for para in split_tasks]
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
                    pass
            split_results = [future.result() for future in futures]
            results += split_results
            json.dump(results, open(cache_results_path, 'w', encoding='utf-8'), ensure_ascii=False)
        
        for r2l, result in zip(ready_to_label_dataset, results):
            if result is None:
                continue
            di = r2l['data index']
            si = r2l['segment index']
            direct_sample_dataset[di]['segment data'][si]['label model'] = label_preference_model
            direct_sample_dataset[di]['segment data'][si]['vote num'] = len(result['scores'])
            direct_sample_dataset[di]['segment data'][si]['vividness scores'] = result
           
        print('[5] Label vividness preference done.')
        
        direct_sample_save_path_json = osp.join(output_dir, 'direct_sample_dataset.json')
        json.dump(direct_sample_dataset, open(direct_sample_save_path_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        