import os
import os.path as osp
import shutil
import json
import yaml
from tqdm import tqdm
from utils import lang_dict, extract_meta_info, raw_ass_to_csv, process_label_csv_2_train_csv, statistic_interval, extract_source_translation_segments_with_sentence_label
        
def ass2csv(label_path, source_path, target_path, meta_info, src_lang_str, tgt_lang_str):
    target_play_name = meta_info['target name']
    for episode in meta_info['episodes']:
        src_csv_file_name = f'{target_play_name}{episode}_{src_lang_str}.csv'
        process_label_csv_2_train_csv(label_path, target_path, src_csv_file_name)
        tgt_csv_file_name = f'{target_play_name}{episode}_{tgt_lang_str}.csv'
        tgt_source_file_path = osp.join(source_path, tgt_csv_file_name.replace('csv', 'ass'))
        tgt_csv_file_path = osp.join(target_path, tgt_csv_file_name)
        raw_ass_to_csv(tgt_source_file_path, tgt_csv_file_path)
        process_label_csv_2_train_csv(target_path, target_path, tgt_csv_file_name)

if __name__ == '__main__':
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    threshold_limit = config['threshold_limit']
    context_len = config['context_len']
    step = config['step']
    lang = config['lang']
    src_lang_str = lang_dict[lang.split('2')[0]]
    tgt_lang_str = lang_dict[lang.split('2')[1]]
        
    model_path = config['model_path']
    os.makedirs(osp.join(model_path, 'vanilla'), exist_ok=True)
    os.makedirs(osp.join(model_path, 'llamafactory'), exist_ok=True)
    os.makedirs(osp.join(model_path, 'alignment'), exist_ok=True)

    source_dir = osp.join(dirname, '..', 'Data', 'Source', lang, 'source(train)')
    label_dir = osp.join(dirname, '..', 'Data', 'Label', lang.split('2')[0])
    data_dir = osp.join(dirname, '..', 'Data', 'Source', lang, 'train') 
    info_dir = osp.join(dirname, 'info', lang)
    play_names = list(filter(lambda file: file != '.DS_Store', sorted(os.listdir(source_dir))))

    print('[1] Extract meta info...')
    if osp.exists(info_dir):
        shutil.rmtree(info_dir)
    os.makedirs(info_dir)
    
    meta = {}
    for play_name in play_names:
        source_path = osp.join(source_dir, play_name)
        meta_info = extract_meta_info(source_path, src_lang_str, tgt_lang_str)
        meta[play_name] = meta_info
    json.dump(meta, open(osp.join(info_dir, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('[1] Extract meta info done.')

    print('[1] Ass file to csv...')
    if osp.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    for play_name in tqdm(play_names):
        source_path = osp.join(source_dir, play_name)
        label_path = osp.join(label_dir, play_name)
        target_path = osp.join(data_dir, play_name)
        os.makedirs(target_path)
        ass2csv(label_path, source_path, target_path, meta[play_name], src_lang_str, tgt_lang_str)
    print('[1] Ass file to csv done.')

    print('[1] Map source to target lang...')
    map_results = {}
    for play_name in tqdm(play_names):
        target_path = osp.join(data_dir, play_name)
        map_result = statistic_interval(target_path, meta[play_name], src_lang_str, tgt_lang_str, threshold_limit=threshold_limit)
        map_results[play_name] = map_result
    json.dump(map_results, open(osp.join(info_dir, 'map_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('[1] Map source to target lang done.')

    print('[1] Extract dialogue fragment...')
    episode_results_dir = osp.join(info_dir, 'episode_results')
    if osp.exists(episode_results_dir):
        shutil.rmtree(episode_results_dir)
    os.makedirs(episode_results_dir)
    for play_name in tqdm(play_names):
        target_path = osp.join(data_dir, play_name)
        episode_result = extract_source_translation_segments_with_sentence_label(target_path, meta[play_name], map_results[play_name], src_lang_str, tgt_lang_str, context_len=context_len, step=step)
        json.dump(episode_result, open(osp.join(episode_results_dir, f'{play_name}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('[1] Extract dialogue fragment done.')
