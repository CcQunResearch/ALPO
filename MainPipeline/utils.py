import re
import os
import csv
import time
import random
import pandas as pd
import os.path as osp
import concurrent.futures
from jsonfinder import jsonfinder
from itertools import combinations
from datetime import timedelta
from Levenshtein import distance as levenshtein_distance
from prompt_template import *
from chat import *

lang_dict = {
    "zh": "简体中文",
    "en": "英语",
    "th": "泰国语",
    "vi": "越南语",
    "id": "印尼语",
    "ms": "马来语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "ar": "阿拉伯语",
    "ja": "日语",
    "ko": "韩语",
    "de": "德语",
    "fr": "法语",
}

# 跟这些词相同就丢弃
filter_nouns = ["爸", "妈", "爷爷", "奶奶", "外公", "外婆", "哥哥", "姐姐", "哥", "姐", "弟弟", "妹妹", "儿子", "女儿",
                "丈夫", "妻子", "朋友", "同学", "老师", "人", "床", "父母", "爸爸", "妈妈", "爹", "娘", "上",
                "下", "左", "右", "爷", "高", "走", "滚", "等", "收", "打", "我女儿", "我儿子",
                "大脑", "真相", "生日", "谢谢", "美国", "中国", "日本", "韩国", "印度", "俄罗斯", "英国", "法国",
                "德国", "意大利", "西班牙", "葡萄牙", "希腊", "土耳其", "以色列", "埃及", "南非", "澳大利亚", "新西兰",
                "加拿大", "墨西哥", "巴西", "阿根廷", "哥伦比亚", "智利", "秘鲁", "委内瑞拉", "女大学生", "花", "药",
                "好", "坏", "高", "矮", "胖", "瘦", "大", "小", "多", "少", "长", "短", "宽", "窄", "高", "矮", "胖",
                "明天", "后天", "今天", "昨天", "前天", "店", "地址", "对不起", "谁", "一个"]

# 包含这些词就丢弃
reversed_filter_nouns = ["我", "你", "他", "她", "它", "咱", "您", "这", "那", "男朋友", "女朋友"]


def clean_punctuation(text):
    punctuation_list = [
        '。', '，', '、', '；', '：', '？', '！', '…', '—', '·', ''', ''', '"', '"',
        '（', '）', '［', '］', '【', '】', '〈', '〉', '「', '」', '『', '』',
        '〔', '〕', '“', '”'
    ]

    # 创建一个正则表达式模式，匹配所有给定的标点符号
    pattern = '|'.join(map(re.escape, punctuation_list))

    # 替换字符串中间的标点为空格
    text = re.sub(f'({pattern})', ' ', text)

    # 去除字符串末尾的标点
    text = re.sub(f'({pattern})$', '', text)

    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))


def is_chinese_only(text):
    # 定义常用中文标点符号列表
    punctuation = [
        '。', '，', '、', '；', '：', '？', '！', '…', '—', '·', ''', ''', '"', '"',
        '（', '）', '［', '］', '【', '】', '《', '》', '〈', '〉', '「', '」', '『', '』',
        '〔', '〕', '“', '”'
    ]

    # 移除所有空白字符和标点符号
    for p in punctuation:
        text = text.replace(p, '')
    text = ''.join(text.split())

    # 检查剩余的字符是否都是中文
    chinese_char_range = '\u4e00-\u9fff'
    pattern = f'^[{chinese_char_range}]+$'

    return bool(re.match(pattern, text) and text)  # 确保文本非空


def time2timestamp(time_str):
    hours, minutes, seconds = map(float, time_str.split(':'))
    time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    timestamp = time_delta.total_seconds()
    return timestamp


def csv_dict_reader(file_path, delimiter='\t'):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        data = {header: [] for header in reader.fieldnames}
        for row in reader:
            for header in reader.fieldnames:
                data[header].append(row[header])
    return data


def csv_dict_writer(file_path, data, delimiter='\t'):
    df = pd.DataFrame(data)
    df.to_csv(file_path, sep=delimiter, index=False)


def extract_meta_info(source_path, src_lang_str, tgt_lang_str):
    file_names = sorted(os.listdir(source_path))
    episodes = []
    target_play_name = ''
    for file_name in file_names:
        split_res = file_name.split('_')
        file_name_postfix = split_res[-2][-2:] + '_' + split_res[-1]
        src_pattern = f'\d+.*_{src_lang_str}.ass'
        pattern = f'\d+.*_{tgt_lang_str}.ass'
        if not re.match(src_pattern, file_name_postfix) and not re.match(pattern, file_name_postfix):
            continue
        else:
            if target_play_name == '':
                target_play_name = file_name[:-len(file_name_postfix)]
            episodes.append(file_name_postfix.split('_')[0])

    episodes = list(set(episodes))
    return {'target name': target_play_name, 'episodes': episodes}


def clear_raw_line(text):
    skip = False

    # 处理优酷字幕的信息标注，通常是联系邮箱等内容
    if r'{\an8}' in text:
        skip = True

    text = text.replace("\u200f", "")

    # [th]删除泰语字幕中的注释行，以及台词中开头出现的注释
    if text[0] == '=' and text[-1] == '=':
        skip = True
    text = re.sub(r'=.*?=\\N', '', text)

    if r'\N' in text:
        text = text.replace(r'\N', ' ')

    if not text or len(text) == 0:
        skip = True

    return text, skip


def raw_ass_to_csv(ass_path, csv_path):
    with open(ass_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line.startswith('Dialogue: '):
            content = line[len('Dialogue: '):]
            parts = content.split(',', 9)
            if len(parts) < 3:
                continue

            start = parts[1]
            end = parts[2]
            text = parts[9]
            text = text.replace('‎', '')
            if r'{\fad(120,120)}' in text:
                text = text.replace(r'{\fad(120,120)}', '')
            if r'{\c&Hffe5e5&}' in text:
                text = text.replace(r'{\c&Hffe5e5&}', '')
            if r'{\c&HE5E5E5&}' in text:
                text = text.replace(r'{\c&HE5E5E5&}', '')
            if r'{\c&HFFFFFF&}' in text:
                text = text.replace(r'{\c&HFFFFFF&}', '')
            if r'{\c}' in text: 
                text = text.replace(r'{\c}', '')
            if r'{\b1}' in text:
                text = text.replace(r'{\b1}', '')
            if r'{\b0}' in text:
                text = text.replace(r'{\b0}', '')
            

            # 处理英文字幕中的- -
            if text.strip() == '- -':
                continue
            if text.strip().startswith('- -'):
                text = text.strip()[3:]

            if text.strip() == '-':
                continue
            
            if len(text.strip()) == 0:
                continue

            data.append((start, end, text))

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Start', 'End', 'Text'])
        for row in data:
            writer.writerow(row)


def process_label_csv_2_train_csv(label_path, target_path, csv_file_name):
    label_csv_path = osp.join(label_path, csv_file_name)
    target_file_path = osp.join(target_path, csv_file_name)
    label_csv_dict = csv_dict_reader(label_csv_path)
    skip_indexes = []
    data_size = len(label_csv_dict['Text'])
    for i in range(len(label_csv_dict['Text'])):
        text = label_csv_dict['Text'][i]
        text, skip = clear_raw_line(text)
        if skip:
            skip_indexes.append(i)
        else:
            label_csv_dict['Text'][i] = text

    if 'sentence segment' in label_csv_dict:
        for skip_index in skip_indexes:
            if skip_index + 1 <= data_size - 1:
                label_csv_dict['sentence segment'][skip_index + 1] = '0'

    for header in label_csv_dict:
        label_csv_dict[header] = [label_csv_dict[header][i] for i in range(data_size) if i not in skip_indexes]
    csv_dict_writer(target_file_path, label_csv_dict)


def statistic_interval(target_path, meta_info, src_lang_str, tgt_lang_str, threshold_limit=0.7):
    map_result = {}
    target_play_name = meta_info['target name']
    for episode in meta_info['episodes']:
        # print(target_path, episode)
        source_file_path = osp.join(target_path, f'{target_play_name}{episode}_{src_lang_str}.csv')
        target_file_path = osp.join(target_path, f'{target_play_name}{episode}_{tgt_lang_str}.csv')
        source_csv_reader = pd.read_csv(open(source_file_path, 'r', encoding='utf-8'), sep='\t')
        target_csv_reader = pd.read_csv(open(target_file_path, 'r', encoding='utf-8'), sep='\t')

        source_duration = source_csv_reader['Start'].tolist()
        target_duration = target_csv_reader['Start'].tolist()
        source_duration = [time2timestamp(duration) for duration in source_duration]
        target_duration = [time2timestamp(duration) for duration in target_duration]

        statistic = {}
        step = abs(len(target_duration) - len(source_duration)) + 10
        min_intervals = []
        for i in range(len(source_duration)):
            compares = target_duration[max(0, i - step):min(i + step, len(target_duration))]
            intervals = [round(abs(source_duration[i] - compare), 2) for compare in compares]
            min_interval = min(intervals)
            min_intervals.append(min_interval)
            statistic[i] = (target_duration.index(compares[intervals.index(min_interval)]), min_interval)
        cleaned_min_intervals = [min_interval for min_interval in min_intervals if min_interval <= threshold_limit]
        threshold = min(threshold_limit, max(cleaned_min_intervals))
        s2c_map = {}
        for index, res in statistic.items():
            if res[1] <= threshold:
                s2c_map[index] = res[0]

        map_result[episode] = s2c_map

        miss_lines = []
        for i in range(len(source_duration)):
            if i not in s2c_map.keys():
                miss_lines.append(i)

        map_result[episode] = {'s2c map': s2c_map, 'miss lines': miss_lines}
    return map_result

def generate_groups_indices(total_length, step, context_length):
    groups = []
    s = 0

    while s < total_length:
        start = max(0, s - context_length)
        end = min(s + step - 1 + context_length, total_length - 1)

        group_indices = list(range(start, end + 1))
        groups.append(group_indices)

        s += step

    return groups


def extract_local_sentence_indexes(segment_labels):
    zero_indexes = [index for index, value in enumerate(segment_labels) if value == '0' or value == 0]
    result = []
    for i in range(len(zero_indexes) - 1):
        start, end = zero_indexes[i], zero_indexes[i + 1] - 1
        result.append(list(range(start, end + 1)))
    result.append(list(range(zero_indexes[-1], len(segment_labels))))
    return result


def generate_groups_indices_with_sengment_indexes(sengment_indexes, step, context_length):
    core_groups = []
    core_group_indexes = []
    curr_core_group = []
    curr_core_group_index = []
    for i, index_list in enumerate(sengment_indexes):
        curr_core_group.extend(index_list)
        curr_core_group_index.append(i)
        if len(curr_core_group) >= step or i == len(sengment_indexes) - 1:
            core_groups.append(curr_core_group)
            core_group_indexes.append(curr_core_group_index)
            curr_core_group = []
            curr_core_group_index = []

    groups = []
    for i, core_group in enumerate(core_groups):
        prefix_context = []
        postfix_context = []
        if i != 0:
            for j in range(core_group_indexes[i][0] - 1, -1, -1):
                prefix_context = sengment_indexes[j] + prefix_context
                if len(prefix_context) >= context_length:
                    break
        if i != len(core_groups) - 1:
            for j in range(core_group_indexes[i][-1] + 1, len(sengment_indexes)):
                postfix_context += sengment_indexes[j]
                if len(postfix_context) >= context_length:
                    break
        core_group_indexes[i] = [len(prefix_context), len(prefix_context) + len(core_group)]
        groups.append(prefix_context + core_group + postfix_context)

    return groups, core_group_indexes




def extract_source_translation_segments_with_sentence_label(target_path, meta_info, map_result, src_lang_str,
                                                            tgt_lang_str, context_len=5, step=25):
    episode_result = {}
    target_play_name = meta_info['target name']
    for episode in sorted(meta_info['episodes']):
        source_file_path = osp.join(target_path, f'{target_play_name}{episode}_{src_lang_str}.csv')
        target_file_path = osp.join(target_path, f'{target_play_name}{episode}_{tgt_lang_str}.csv')
        source_csv_dict = csv_dict_reader(source_file_path)
        target_csv_dict = csv_dict_reader(target_file_path)
        data_size = len(source_csv_dict['Text'])
        original_lines = list(range(data_size))
        s2c_map = map_result[episode]['s2c map']
        boundaries = [-1] + map_result[episode]['miss lines'] + [data_size]
        results = []
        sentence_segment_label = source_csv_dict['sentence segment']
        for i in range(len(boundaries) - 1):
            head_index = boundaries[i] + 1
            tail_index = boundaries[i + 1]
            
            if tail_index < len(sentence_segment_label) and sentence_segment_label[tail_index] == '1':
                check_label = sentence_segment_label[:tail_index]
                tail_index = len(check_label) - 1 - check_label[::-1].index('0')

            if tail_index - head_index < 10:
                continue
            src_text_fragment = source_csv_dict['Text'][head_index:tail_index]
            tgt_text_fragment = [target_csv_dict['Text'][s2c_map[id]] for id in range(head_index, tail_index)]
            src_start_timestamp_fragment = source_csv_dict['Start'][head_index:tail_index]
            src_end_timestamp_fragment = source_csv_dict['End'][head_index:tail_index]
            original_line_fragment = original_lines[head_index:tail_index]
            src_sentence_label_fragment = sentence_segment_label[head_index:tail_index]
            src_sentence_label_fragment[0] = '0'
            sentence_local_indexes = extract_local_sentence_indexes(src_sentence_label_fragment)
            groups, core_group_indexes = generate_groups_indices_with_sengment_indexes(sentence_local_indexes, step,
                                                                                       context_len)
            for j, group in enumerate(groups):
                src_text = [src_text_fragment[k] for k in group]
                tgt_text = [tgt_text_fragment[k] for k in group]
                src_start_timestamp = [src_start_timestamp_fragment[k] for k in group]
                src_end_timestamp = [src_end_timestamp_fragment[k] for k in group]
                source_duration = [round(time2timestamp(a) - time2timestamp(b), 3) for a, b in
                                   zip(src_end_timestamp, src_start_timestamp)]
                original_line = [original_line_fragment[k] for k in group]
                src_sentence_label = [int(src_sentence_label_fragment[k]) for k in group]
                res = {
                    'source': src_text,
                    'target': tgt_text,
                    'original line': original_line,
                    'source sentence label': src_sentence_label,
                    'source duration': source_duration,
                    'source start timestamp': src_start_timestamp,
                    'source end timestamp': src_end_timestamp,
                    'core index': core_group_indexes[j]
                }
                results.append(res)
        episode_result[episode] = results
    return episode_result


def extract_dialogue_translation(target_path, meta_info, map_result, src_lang_str, tgt_lang_str, context_len=5, step=25, merge_audio_path=False):
    episode_result = {}
    target_play_name = meta_info['target name']
    for episode in meta_info['episodes']:
        source_file_path = osp.join(target_path, f'{target_play_name}{episode}_{src_lang_str}.csv')
        target_file_path = osp.join(target_path, f'{target_play_name}{episode}_{tgt_lang_str}.csv')
        source_csv_reader = pd.read_csv(open(source_file_path, 'r', encoding='utf-8'), sep='\t')
        target_csv_reader = pd.read_csv(open(target_file_path, 'r', encoding='utf-8'), sep='\t')
        source_dialogue = source_csv_reader['Text'].tolist()
        target_dialogue = target_csv_reader['Text'].tolist()
        source_start_timestamp = source_csv_reader['Start'].tolist()
        source_end_timestamp = source_csv_reader['End'].tolist()
        source_original_line = list(range(len(source_dialogue)))
        source_audio_path = source_csv_reader['Audio Path'].tolist() if merge_audio_path else ['Not Found'] * len(source_dialogue)

        results = []
        s2c_map = map_result[episode]['s2c map']
        miss_lines = map_result[episode]['miss lines'] + [len(source_dialogue)]
        end = -1
        for miss_line in miss_lines:
            begin = end + 1
            end = miss_line
            dialogue_fragment = [str(source_dialogue[index]) for index in range(begin, end)]
            translation_fragment = [str(target_dialogue[s2c_map[index]]) for index in range(begin, end)]
            source_original_line_fragment = [source_original_line[index] for index in range(begin, end)]
            source_start_timestamp_fragment = [source_start_timestamp[index] for index in range(begin, end)]
            source_end_timestamp_fragment = [source_end_timestamp[index] for index in range(begin, end)]
            source_audio_path_fragment = [source_audio_path[index] for index in range(begin, end)] if merge_audio_path else ['Not Found'] * len(dialogue_fragment)

            head_index = 0
            tail_index = len(dialogue_fragment) + 100000
            if begin == 0 and end - begin - context_len > step:
                head_index = -context_len
                tail_index = step + context_len
            elif begin != 0 and end - begin - 2 * context_len > step:
                head_index = 0
                tail_index = step + 2 * context_len
            else:
                if len(dialogue_fragment) >= 15:
                    res = {
                        'source': dialogue_fragment,
                        'target': translation_fragment,
                        'original line': source_original_line_fragment
                    }
                    cs_timestamp = source_start_timestamp_fragment
                    ce_timestamp = source_end_timestamp_fragment
                    source_duration = [round(time2timestamp(a) - time2timestamp(b), 3) for a, b in
                                        zip(ce_timestamp, cs_timestamp)]
                    res['source duration'] = source_duration
                    res['source start timestamp'] = cs_timestamp
                    res['source end timestamp'] = ce_timestamp

                    if merge_audio_path:
                        res['chinese audio path'] = source_audio_path_fragment
                        res['audio complete'] = 'Not Found' not in res['chinese audio path']

                    results.append(res)
            while tail_index <= len(dialogue_fragment):
                res = {
                    'source': dialogue_fragment[max(0, head_index):tail_index],
                    'target': translation_fragment[max(0, head_index):tail_index],
                    'original line': source_original_line_fragment[max(0, head_index):tail_index]
                }
                cs_timestamp = source_start_timestamp_fragment[max(0, head_index):tail_index]
                ce_timestamp = source_end_timestamp_fragment[max(0, head_index):tail_index]
                source_duration = [round(time2timestamp(a) - time2timestamp(b), 3) for a, b in
                                    zip(ce_timestamp, cs_timestamp)]
                res['source duration'] = source_duration
                res['source start timestamp'] = cs_timestamp
                res['source end timestamp'] = ce_timestamp

                if merge_audio_path:
                    res['chinese audio path'] = source_audio_path_fragment[max(0, head_index):tail_index]
                    res['audio complete'] = 'Not Found' not in res['chinese audio path']

                results.append(res)
                head_index += step
                tail_index += step
            if tail_index < 100000 and len(dialogue_fragment) - head_index - 2 * context_len > 10:
                res = {
                    'source': dialogue_fragment[head_index:],
                    'target': translation_fragment[head_index:],
                    'original line': source_original_line_fragment[head_index:]
                }
                cs_timestamp = source_start_timestamp_fragment[head_index:]
                ce_timestamp = source_end_timestamp_fragment[head_index:]
                source_duration = [round(time2timestamp(a) - time2timestamp(b), 3) for a, b in
                                    zip(ce_timestamp, cs_timestamp)]
                res['source duration'] = source_duration
                res['source start timestamp'] = cs_timestamp
                res['source end timestamp'] = ce_timestamp

                if merge_audio_path:
                    res['chinese audio path'] = source_audio_path_fragment[head_index:]
                    res['audio complete'] = 'Not Found' not in res['chinese audio path']

                results.append(res)
        episode_result[episode] = results
    return episode_result

def extract_dialogue_translation_nogt(target_path, meta_info, src_lang_str, context_len=5, step=25):
    episode_result = {}
    target_play_name = meta_info['target name']
    for episode in meta_info['episodes']:
        source_file_path = osp.join(target_path, f'{target_play_name}{episode}_{src_lang_str}.csv')
        csv_data = csv_dict_reader(source_file_path)
        source_dialogue = csv_data['Text']
        source_start_timestamp = csv_data['Start']
        source_end_timestamp = csv_data['End']
        source_original_line = list(range(len(source_dialogue)))
        
        results = []
        groups = generate_groups_indices(len(source_dialogue), step, context_len)
        for group in groups:
            dialogue_fragment = [source_dialogue[index] for index in group]
            source_original_line_fragment = [source_original_line[index] for index in group]
            source_start_timestamp_fragment = [source_start_timestamp[index] for index in group]
            source_end_timestamp_fragment = [source_end_timestamp[index] for index in group]
            source_duration = [round(time2timestamp(a) - time2timestamp(b), 3) for a, b in zip(source_end_timestamp_fragment, source_start_timestamp_fragment)]
            res = {
                'source': dialogue_fragment,
                'original line': source_original_line_fragment,
                'source duration': source_duration,
                'source start timestamp': source_start_timestamp_fragment,
                'source end timestamp': source_end_timestamp_fragment
            }
            results.append(res)
        episode_result[episode] = results
    return episode_result


def extract_proper_noun(generation_results):
    result = {}
    for tmp_result in generation_results:
        predict = tmp_result['predict']
        for res in predict.split('\n'):
            pattern = re.compile(r'(.+?)（(.+?)）\s?-\s?(.+)')
            matches = pattern.findall(res)
            if len(matches) > 0:
                match = matches[0]
            else:
                continue

            word = match[0].strip()
            type = match[1].strip()
            translation = match[2].strip()

            if word in result.keys():
                types = type.split('/')
                for t in types:
                    if t not in result[word]['type'].keys():
                        result[word]['type'][t] = 1
                    else:
                        result[word]['type'][t] += 1
                if translation not in result[word]['translation'].keys():
                    result[word]['translation'][translation] = 1
                else:
                    result[word]['translation'][translation] += 1
                result[word]['count'] += 1
            else:
                result[word] = {
                    'type': {},
                    'translation': {translation: 1},
                    'count': 1
                }
                type = type.replace('+', '/')
                types = type.split('/')
                for t in types:
                    result[word]['type'][t] = 1

    return result


def find_contained_words(words):
    contained_words = set()

    # 遍历列表中的每一个词
    for word in words:
        for other_word in words:
            # 确保不比较同一个词，同时判断包含关系
            if word != other_word and word in other_word:
                contained_words.add((other_word, word))
                break

    return list(contained_words)


def filter_translation(translation, lang='nolang'):
    all_trans = list(translation.items())
    if lang in ['ko2zh', 'en2zh', 'ja2zh']:
        new_trans = []
        for tran in all_trans:
            if is_chinese_only(tran[0]):
                new_trans.append(tran)
        if len(new_trans) != 0:
            all_trans = new_trans
    sorted_all_trans = sorted(all_trans, key=lambda x: x[1], reverse=True)
    return sorted_all_trans[0][0]


def filter_proper_noun_result(proper_noun_result, lang='nolang', threshold=3):
    filter_result = {}
    proper_terms = []
    for term in proper_noun_result.keys():
        # 只有一个字
        if len(term) == 1:
            continue

        # 直接过滤
        if term in filter_nouns:
            # print(term)
            continue
        reversed_pass = False
        for rf in reversed_filter_nouns:
            if rf in term:
                reversed_pass = True
                break
        if reversed_pass:
            continue

        # 出现多次，且类型为地名、机构、人名等类型，则一致认为属于专有名词
        types = proper_noun_result[term]['type'].keys()
        if '地名' in types or '机构' in types or '人名' in types:
            trans = filter_translation(proper_noun_result[term]['translation'], lang)
            typee = filter_translation(proper_noun_result[term]['type'])
            proper_terms.append([term, typee, trans])
            continue

        # 剩下的通过阈值来判断是否作为专有名词
        if proper_noun_result[term]['count'] > threshold:
            trans = filter_translation(proper_noun_result[term]['translation'], lang)
            typee = filter_translation(proper_noun_result[term]['type'])
            proper_terms.append([term, typee, trans])
            continue

    for proper_term in proper_terms:
        filter_result[proper_term[0]] = {"translation": proper_term[2], "type": proper_term[1]}
    return filter_result


def easy_filter_proper_noun_result(proper_noun_result):
    filter_result = {}
    proper_terms = []
    for term in proper_noun_result.keys():
        # 只有一个字
        if len(term) == 1:
            continue

        # 直接过滤
        if term in filter_nouns:
            continue
        reversed_pass = False
        for rf in reversed_filter_nouns:
            if rf in term:
                reversed_pass = True
                break
        if reversed_pass:
            continue

        trans = filter_translation(proper_noun_result[term]['translation'])
        typee = filter_translation(proper_noun_result[term]['type'])
        proper_terms.append([term, typee, trans])

    for proper_term in proper_terms:
        filter_result[proper_term[0]] = {"translation": proper_term[2], "type": proper_term[1]}
    return filter_result

def extract_training_queries_and_responses(play_name, episode_result, proper_noun_dict, template, src_lang_str,
                                           lang_str, pn_identify_dict=None, pn_consis=False, evaluation_mode=True):
    play_data = []
    pn_fewshot = []

    trie = Trie()
    for word in set(proper_noun_dict.keys()):
        trie.insert(word)

    for episode, dialogue in episode_result.items():
        for ce_pair_dict in dialogue:
            ptypes = []
            if evaluation_mode:
                ce_pair_list = list(zip(ce_pair_dict["source"], ce_pair_dict["target"]))
                for i in range(len(ce_pair_list)):
                    if not ce_pair_list[i][0]:
                        ce_pair_list[i][0] = ""
                    if not ce_pair_list[i][1]:
                        ce_pair_list[i][1] = ""
            else:
                ce_pair_list = list(zip(ce_pair_dict["source"], [""] * len(ce_pair_dict["source"])))
                for i in range(len(ce_pair_list)):
                    if not ce_pair_list[i][0]:
                        ce_pair_list[i][0] = ""

            dialogue_content = '\n'.join([f'{i+1}. {ce_pair[0]}' for i, ce_pair in enumerate(ce_pair_list)])
            proper_noun_list = proper_noun_retrieve(dialogue_content, trie)
            proper_noun_list = sorted(proper_noun_list, key=lambda x: dialogue_content.index(x))
            if not pn_consis:
                proper_noun_pairs = [(p, proper_noun_dict[p]['type'], proper_noun_dict[p]['translation']) for p in
                                     proper_noun_list]
            else:
                target_dialogue_content = '\n'.join([ce_pair[1] for ce_pair in ce_pair_list])
                proper_noun_pairs = []
                for p in proper_noun_list:
                    trans_of_p = list(pn_identify_dict[p]['translation'].keys())
                    trans_of_p = nested_sort(trans_of_p)
                    chosen_trans = proper_noun_dict[p]['translation']
                    for trans in trans_of_p:
                        if trans in target_dialogue_content:
                            chosen_trans = trans
                            break
                    proper_noun_pairs.append((p, proper_noun_dict[p]['type'], chosen_trans))

            ptypes = list(set([pair[1] for pair in proper_noun_pairs]))
            full_proper_noun_content = '\n'.join(
                [f'{pair[0]}（{pair[1]}） - {pair[2]}' for pair in proper_noun_pairs]) if proper_noun_pairs else '无专有名词'

            prompt = get_sft_prompt(src_lang_str, lang_str, template, full_proper_noun_content, dialogue_content)
            response = '\n'.join([f'{i+1}. {ce_pair[1]}' for i, ce_pair in enumerate(ce_pair_list)]) if evaluation_mode else None

            res = {'instruction': prompt, 'input': None, 'output': response,
                   'source': ce_pair_dict['source'], 'target': ce_pair_dict['target'] if evaluation_mode else [],
                   'proper noun': full_proper_noun_content,'source duration': ce_pair_dict['source duration'],
                   'source start timestamp': ce_pair_dict['source start timestamp'],
                   'source end timestamp': ce_pair_dict['source end timestamp'], 'original line': ce_pair_dict['original line'],
                   'source sentence label': ce_pair_dict['source sentence label'] if 'source sentence label' in ce_pair_dict else [], 
                   'core index': ce_pair_dict['core index'] if 'core index' in ce_pair_dict else [],
                   'play name': play_name, 'episode': episode}

            play_data.append(res)

            if len(proper_noun_list) > 8 and len(ptypes) > 3:
                pn_fewshot.append({"dialogue": dialogue_content, "term": full_proper_noun_content})
    return play_data, pn_fewshot

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word


def remove_subwords(word_list):
    result = []
    for i, word in enumerate(word_list):
        is_substring = False
        for j, other_word in enumerate(word_list):
            if i != j and word in other_word:
                is_substring = True
                break
        if not is_substring:
            result.append(word)
    return result


def proper_noun_retrieve(text, trie):
    result = []

    # 在文本中搜索每个词
    for i in range(len(text)):
        node = trie.root
        for j in range(i, len(text)):
            if text[j] not in node.children:
                break
            node = node.children[text[j]]
            if node.is_word:
                result.append(text[i:j + 1])

    result = list(set(result))
    result = remove_subwords(result)
    return result


def nested_sort(words):
    # 对列表进行排序，按长度降序
    words.sort(key=len, reverse=True)

    result = []

    while words:
        word = words.pop(0)
        # 将当前词加入结果列表
        result.append(word)
        # 去掉被当前词包含的词
        words = [w for w in words if word not in w]

    return result


def get_sft_prompt(src_lang_str, lang_str, template, proper_noun_content, dialogue_content):
    replace_dict = get_replace_dict()
    template = template.replace("<<1>>", random.choice(replace_dict["<<1>>"]))
    template = template.replace("<<2>>", random.choice(replace_dict["<<2>>"]))
    template = template.replace("<<3>>", random.choice(replace_dict["<<3>>"]))
    template = template.replace("<<4>>", random.choice(replace_dict["<<4>>"]))
    template = template.replace("<<5>>", random.choice(replace_dict["<<5>>"]))
    template = template.replace("<<6>>", random.choice(replace_dict["<<6>>"]))
    replace_src_lang_str = src_lang_str
    if src_lang_str == "简体中文":
        replace_src_lang_str = "中文"
    replace_lang_str = lang_str
    if lang_str == "简体中文":
        replace_lang_str = "中文"
    template = template.replace("<src_lang_str>", replace_src_lang_str)
    template = template.replace("<lang_str>", replace_lang_str)
    prompt = template.format(proper_noun_content, dialogue_content)
    return prompt


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root != y_root:
            self.parent[y_root] = x_root


def tokenize(s):
    """
    多语言分词器（支持中日韩+拉丁字母）
    处理规则：
    1. 中日韩字符（汉字/假名/Hangul）按单字切分
    2. 拉丁语系按空格切分单词
    3. 保留原始大小写（需根据场景决定是否转小写）
    """
    tokens = []

    # Unicode范围定义
    cjk_blocks = [
        ('Hiragana', r'\u3040-\u309F'),  # 日文平假名
        ('Katakana', r'\u30A0-\u30FF'),  # 日文片假名
        ('Hangul_Syllables', r'\uAC00-\uD7AF'),  # 韩文音节
        ('CJK_Unified', r'\u4E00-\u9FFF'),  # 中日韩统一汉字
    ]

    # 构建正则表达式模式
    cjk_pattern = '|'.join(f'([{block}]+)' for _, block in cjk_blocks)
    other_pattern = '([^{}]+)'.format(''.join(block for _, block in cjk_blocks))
    pattern = re.compile(f'{cjk_pattern}|{other_pattern}')

    # 执行分词
    for match in pattern.finditer(s):
        # 按捕获组顺序获取匹配结果
        groups = match.groups()

        # 处理CJK字符（每组单独处理）
        cjk_matched = False
        for i in range(len(cjk_blocks)):
            if groups[i]:
                tokens.extend(list(groups[i]))  # 单字切分
                cjk_matched = True
                break

        # 处理非CJK字符
        if not cjk_matched and groups[-1]:
            # 拉丁语系按空格切分
            words = groups[-1].strip().split()
            tokens.extend(words)

    return set(tokens)


def cluster_strings(strings, threshold=0.5, similarity_type='jaccard'):
    n = len(strings)
    dsu = DSU(n)
    tokens_list = [tokenize(s) for s in strings]

    for i, j in combinations(range(n), 2):
        set1 = tokens_list[i]
        set2 = tokens_list[j]
        intersection = len(set1 & set2)

        if similarity_type == 'jaccard':
            union = len(set1 | set2)
            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union
        elif similarity_type == 'count':
            similarity = intersection
        else:
            raise ValueError("similarity_type must be 'jaccard' or 'count'")

        if similarity >= threshold:
            dsu.union(i, j)

    clusters = {}
    for idx in range(n):
        root = dsu.find(idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(strings[idx])

    return list(clusters.values())


def extract_json_from_text(text):
    objs = []
    for _, _, obj in jsonfinder(text, json_only=True):
        objs.append(obj)
    return objs

def extract_first_json_from_text(text):
    try:
        for _, _, obj in jsonfinder(text, json_only=True):
            return obj
    except Exception:
        return None

def align_filter_result(filter_result, lang, model):
    terms = list(filter_result.keys())
    threshold = 0.3
    while threshold <= 0.8:
        clusters = cluster_strings(terms, threshold=threshold, similarity_type='jaccard')
        if len(clusters) == 0:
            break
        max_cluster_size = max([len(cluster) for cluster in clusters])
        if max_cluster_size > 15:
            threshold += 0.05
        else:
            break

    src_lang = lang_dict[lang.split('2')[0]]
    src_lang = '中文' if src_lang == '简体中文' else src_lang
    tgt_lang = lang_dict[lang.split('2')[1]]
    tgt_lang = '中文' if tgt_lang == '简体中文' else tgt_lang
    for cluster in clusters:
        if len(cluster) > 1:
            print(f"Aligning cluster: {cluster}")
            term_trans = ''
            for i in range(len(cluster)):
                tmp = f"{i + 1}.\n原文：{cluster[i]}\n译文：{filter_result[cluster[i]]['translation']}\n类型：{filter_result[cluster[i]]['type']}\n"
                term_trans += tmp
            term_trans = term_trans.strip()
            prompt = term_adjust_template.format(term_adjust_few_shot, f'{src_lang}翻译{tgt_lang}', src_lang, tgt_lang,
                                                 src_lang, tgt_lang, term_trans)
            iter = 0
            while True:
                try:
                    response, _ = chat_ali(prompt, model)
                    align_result = extract_json_from_text(response)[0]
                except Exception as e:
                    print(f"{model}发生异常：{e}")
                    align_result = None
                finally:
                    if align_result:
                        print(f'过滤后结果：\n{term_trans}')
                        print(f'对齐后结果：\n{align_result}')
                        print('==================================================================')
                        for term in align_result.keys():
                            if term in terms and align_result[term]["require correction"]:
                                filter_result[term]['translation'] = align_result[term]['translation']
                        break
                    iter += 1
                    if iter > 5:
                        break
    return filter_result


def check_equal(origin, target):
    if origin in target:
        return True
    target = target.replace('[连贯]', '').replace('[不连贯]', '')
    if levenshtein_distance(origin, target) / len(origin) <= 0.35:
        return True
    return False


def sentence_segment(dialogue, prompt, model, retry_num=3):
    retry_count = 0
    while retry_count < retry_num:
        try:
            response, _ = chat_ali(prompt, model, temperature=random.uniform(0, 0.5))
        except Exception as e:
            print(f"{model}发生异常：{e}")
            retry_count += 1
            time.sleep(1)
        else:
            dialogue_lines = dialogue.split('\n')
            result_lines = response.split('\n')
            candidates = [line.strip() for line in result_lines if
                          line.strip().startswith('[连贯]') or line.strip().startswith('[不连贯]')]

            if len(dialogue_lines) != len(candidates):
                retry_count += 1
                continue
            else:
                consistency_marks = []
                coherent_label = []
                for i in range(len(dialogue_lines)):
                    if check_equal(dialogue_lines[i].strip(), candidates[i].strip()):
                        consistency_marks.append(1)
                    else:
                        print(dialogue_lines[i], candidates[i])
                        consistency_marks.append(0)
                    if candidates[i].startswith('[连贯]'):
                        coherent_label.append(1)
                    else:
                        coherent_label.append(0)
                if sum(consistency_marks) / len(consistency_marks) >= 0.8 and len(coherent_label) == len(dialogue_lines):
                    return coherent_label
                else:
                    print('sum(consistency_marks) / len(consistency_marks)',
                          sum(consistency_marks) / len(consistency_marks))
                    retry_count += 1
                    continue
    return None


def sentence_level_segmentation(segment_file_path, lang, model, retry_num, num_workers):
    src_lang = lang_dict[lang.split('2')[0]]
    src_lang = '中文' if src_lang == '简体中文' else src_lang
    if src_lang == '中文':
        template = sentence_segment_template_zh
    elif src_lang == '英语':
        template = sentence_segment_template_en
    elif src_lang == '日语':
        template = sentence_segment_template_ja
    elif src_lang == '韩语':
        template = sentence_segment_template_ko
    else:
        return False

    with open(segment_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        dialogue_list = [row['Text'] for row in reader]

    steps = [25, 30, 35]
    content_length = 5
    all_episode_results = []
    for step in steps:
        groups = generate_groups_indices(len(dialogue_list), step, content_length)
        tasks = []
        for group in groups:
            dialogue = '\n'.join([dialogue_list[i] for i in group]).strip()
            prompt = template.format(src_lang, src_lang, dialogue)
            tasks.append([dialogue, prompt, model, retry_num])

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(sentence_segment, *para) for para in tasks]
            concurrent.futures.wait(futures)
        results = [future.result() for future in futures]
        episode_result = []
        error_count = 0
        for i, result in enumerate(results):
            begin = 0
            end = len(groups[i])
            if i != 0:
                begin = content_length
            if i != len(groups) - 1:
                end = begin + step
            if not result:
                episode_result.extend([-1] * (end - begin))
                error_count += end - begin
                continue
            else:
                episode_result.extend(result[begin:end])
        episode_result[0] = 0
        if len(episode_result) == len(dialogue_list):
            all_episode_results.append(episode_result)

    if len(all_episode_results) == 0:
        print(f"No valid results found, model: {model}, step: {step}, file name: {segment_file_path}")
        return False

    transposed = zip(*all_episode_results)
    vote_result = []
    for i, elements in enumerate(transposed):
        count0 = elements.count(0)
        count1 = elements.count(1)
        vote_result.append(0 if count0 >= count1 else 1)
        count_minus1 = elements.count(-1)
        if count_minus1 == len(steps):
            print(f"    failed to label line {i + 1}, marked as 0.")

    with open(segment_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        rows = list(reader)

    headers.append("sentence segment")
    for i, row in enumerate(rows):
        row.append(vote_result[i])

    with open(segment_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        writer.writerows(rows)

    return True

def find_consecutive_indexes(list, tar=0, threshold=8):
    result = []
    start = None
    n = len(list)
    for i in range(n):
        if list[i] == tar:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                length = end - start + 1
                if length >= 8:
                    result.append((start, end))
                start = None
    if start is not None:
        end = n - 1
        length = end - start + 1
        if length >= threshold:
            result.append((start, end))
    return result

def refine_sentence_segmentation(segment_file_path, lang, model, retry_num, num_workers):
    src_lang = lang_dict[lang.split('2')[0]]
    src_lang = '中文' if src_lang == '简体中文' else src_lang
    if src_lang == '中文':
        template = sentence_segment_template_zh
    elif src_lang == '英语':
        template = sentence_segment_template_en
    elif src_lang == '日语':
        template = sentence_segment_template_ja
    elif src_lang == '韩语':
        template = sentence_segment_template_ko
    else:
        return False

    with open(segment_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        dialogue_list = [row['Text'] for row in rows]
        label_list = [int(row['sentence segment']) for row in rows]
    
    zero_indexes = find_consecutive_indexes(label_list, tar=0, threshold=8)
    zero_indexes = [[index[0], index[1] - 1] for index in zero_indexes]
    one_indexes = find_consecutive_indexes(label_list, tar=1, threshold=6)
    one_indexes = [[index[0] - 1, index[1]] for index in one_indexes]
    
    if len(zero_indexes) > 0 or len(one_indexes) > 0:
        indexes = zero_indexes + one_indexes
        tasks = []
        for index in indexes:
            begin, end = index
            dialogue = '\n'.join(dialogue_list[begin: end]).strip()
            prompt = template.format(src_lang, src_lang, dialogue)
            tasks.append([dialogue, prompt, model, retry_num])

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(sentence_segment, *para) for para in tasks]
            concurrent.futures.wait(futures)
        results = [future.result() for future in futures]
        
        for index, res in zip(indexes, results):
            begin, end = index    
            if res is not None:
                label_list[begin:end] = res

        label_list[0] = 0
        label_csv_dict = csv_dict_reader(segment_file_path)
        label_csv_dict['sentence segment'] = label_list
        csv_dict_writer(segment_file_path, label_csv_dict)    
    else:
        return False