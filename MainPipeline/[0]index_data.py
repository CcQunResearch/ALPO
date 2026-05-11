import os
import os.path as osp
import yaml
import pandas as pd
from tqdm import tqdm
from utils import lang_dict, raw_ass_to_csv, sentence_level_segmentation, refine_sentence_segmentation

if __name__ == '__main__':
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    sentence_split_model = config['sentence_split_model']
    sentence_split_refine_models = config['sentence_split_refine_models']
    sentence_split_retry_num = config['sentence_split_retry_num']
    sentence_split_num_workers = config['sentence_split_num_workers']
    lang = config['lang']
    src_lang = lang.split('2')[0]
    src_lang_str = lang_dict[src_lang]
    
    source_dir = osp.join(dirname, '..', 'Data', 'Source', lang, 'source(train)')
    label_dir = osp.join(dirname, '..', 'Data', 'Label', src_lang)
    play_names = list(filter(lambda file: file != '.DS_Store', sorted(os.listdir(source_dir))))
    os.makedirs(label_dir, exist_ok=True)
    
    print('[0] Sentence-level segment...')
    for play_name in tqdm(play_names):
        source_path = osp.join(source_dir, play_name)
        label_play_path = osp.join(label_dir, play_name)
        os.makedirs(label_play_path, exist_ok=True)
        episode_file_names = list(filter(lambda file: src_lang_str in file, sorted(os.listdir(source_path))))
        for episode_file_name in episode_file_names:
            source_file_path = osp.join(source_path, episode_file_name)
            segment_file_path = osp.join(label_play_path, episode_file_name.replace('ass', 'csv'))
            
            if osp.exists(segment_file_path) and 'sentence segment' in pd.read_csv(segment_file_path, delimiter='\t').columns:
                continue
            if not osp.exists(segment_file_path): 
                raw_ass_to_csv(source_file_path, segment_file_path)

            success_segment = sentence_level_segmentation(segment_file_path, lang, sentence_split_model, sentence_split_retry_num, sentence_split_num_workers)
            if not success_segment:
                print(f'{play_name} {episode_file_name} sentence segment failed.')

            for sentence_split_refine_model in sentence_split_refine_models.split(','):
                refine_sentence_segmentation(segment_file_path, lang, sentence_split_refine_model.strip(), sentence_split_retry_num, sentence_split_num_workers)
    print('[0] Sentence-level segment done.')