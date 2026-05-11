import json
import os
import os.path as osp
import random

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from alpo_utils import (
    choose_prefix_lines,
    compute_dynamic_betas,
    compute_segment_weights,
    format_numbered_lines,
    get_average_scores,
    is_trainable_segment,
    linear_schedule,
    select_chosen_rejected_indices,
    split_target_lines,
)
from template_utils import apply_model_chat_template, get_generation_stop_text


def build_segment_plan(direct_sample, config, rng):
    min_candidates = config.get('alpo_min_candidates', 3)
    min_score_gap = config.get('alpo_min_score_gap', 5.0)
    chosen_top_k = config.get('alpo_chosen_top_k', 3)
    rejected_rank = config.get('alpo_rejected_rank', 3)

    segment_plan = []
    candidate_counts = []
    gates = []
    gaps = []

    for segment in direct_sample['segment data']:
        targets = segment.get('segment targets', [])
        candidate_lines = [split_target_lines(target) for target in targets]
        scores = get_average_scores(segment)
        trainable = is_trainable_segment(segment, min_candidates, min_score_gap)

        if scores:
            chosen_index, rejected_index = select_chosen_rejected_indices(
                scores,
                targets,
                rng,
                chosen_top_k=chosen_top_k,
                rejected_rank=rejected_rank,
            )
            chosen_score = scores[chosen_index]
            rejected_score = scores[rejected_index]
        else:
            chosen_index = 0
            rejected_index = 0
            chosen_score = 0.0
            rejected_score = 0.0

        gap = max(0.0, chosen_score - rejected_score)
        candidate_count = len(targets)
        gate = bool(trainable and gap > 0)

        segment_plan.append({
            'candidate_lines': candidate_lines,
            'chosen_lines': candidate_lines[chosen_index] if candidate_lines else [],
            'rejected_lines': candidate_lines[rejected_index] if candidate_lines else [],
            'chosen_index': chosen_index,
            'rejected_index': rejected_index,
            'chosen_score': chosen_score,
            'rejected_score': rejected_score,
            'score_gap': gap,
            'candidate_count': candidate_count,
            'gate': gate,
        })
        candidate_counts.append(candidate_count)
        gates.append(gate)
        gaps.append(gap)

    weights = compute_segment_weights(candidate_counts, gates)
    betas = compute_dynamic_betas(gaps, gates)
    for plan_item, weight, beta in zip(segment_plan, weights, betas):
        plan_item['weight'] = weight
        plan_item['beta'] = beta
    return segment_plan


def build_records(direct_sample_dataset, config, sft_model, tokenizer):
    seed = config.get('alpo_seed', 1024)
    lambda_values = linear_schedule(
        config.get('alpo_lambda_start', 0.2),
        config.get('alpo_lambda_end', 0.6),
        config.get('alpo_prefix_stages', 5),
    )
    stop_token = get_generation_stop_text(tokenizer, sft_model)

    records = []
    skipped_samples = 0
    skipped_segments = 0

    for sample_index, direct_sample in enumerate(tqdm(direct_sample_dataset, desc='Building ALPO data')):
        sample_rng = random.Random(seed + sample_index)
        segment_plan = build_segment_plan(direct_sample, config, sample_rng)
        if not any(item['gate'] for item in segment_plan):
            skipped_samples += 1
            continue

        base_prompt = apply_model_chat_template(tokenizer, direct_sample['instruction'], sft_model)
        for stage_index, lambda_value in enumerate(lambda_values):
            stage_rng = random.Random(seed + sample_index * 100003 + stage_index)
            prefix_lines = []

            for segment_index, plan_item in enumerate(segment_plan):
                if plan_item['gate']:
                    segment_count = len(plan_item['chosen_lines'])
                    if segment_count == 0:
                        skipped_segments += 1
                    else:
                        prefix_response = ''
                        if prefix_lines:
                            prefix_response = format_numbered_lines(prefix_lines, 1, suffix='\n')
                        start_index = len(prefix_lines) + 1
                        is_last_segment = segment_index == len(segment_plan) - 1
                        suffix = stop_token if is_last_segment and stop_token else '\n'

                        records.append({
                            'prompt': base_prompt + prefix_response,
                            'chosen': format_numbered_lines(plan_item['chosen_lines'], start_index, suffix=suffix),
                            'rejected': format_numbered_lines(plan_item['rejected_lines'], start_index, suffix=suffix),
                            'weight': plan_item['weight'],
                            'beta': plan_item['beta'],
                            'lambda': lambda_value,
                            'stage_index': stage_index,
                            'sample_index': sample_index,
                            'segment_index': segment_index,
                            'chosen_index': plan_item['chosen_index'],
                            'rejected_index': plan_item['rejected_index'],
                            'chosen_score': plan_item['chosen_score'],
                            'rejected_score': plan_item['rejected_score'],
                            'score_gap': plan_item['score_gap'],
                            'candidate_count': plan_item['candidate_count'],
                        })

                next_prefix = choose_prefix_lines(
                    plan_item['candidate_lines'],
                    plan_item['chosen_lines'],
                    lambda_value,
                    stage_rng,
                )
                prefix_lines.extend(next_prefix)

    summary = {
        'records': len(records),
        'samples': len(direct_sample_dataset),
        'skipped_samples': skipped_samples,
        'skipped_segments': skipped_segments,
        'lambda_values': lambda_values,
    }
    return records, summary


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

    if sft_proportion == 1.0:
        print('[6-alpo] sft_proportion is 1.0, skip ALPO dataset construction.')
        raise SystemExit(0)

    output_dir = osp.join(
        dirname,
        '..',
        'LLaMAFactory',
        'VividnessAlignment',
        lang,
        f'sample_{sft_model}_{trpe_model}_{sft_proportion}',
    )
    direct_sample_path = osp.join(output_dir, 'direct_sample_dataset.json')
    direct_sample_dataset = json.load(open(direct_sample_path, 'r', encoding='utf-8'))

    sft_model_path = config.get(
        'alpo_model_name_or_path',
        osp.join(model_path, 'llamafactory', f'{sft_model}_{trpe_model}', lang, f'sft_{sft_proportion}'),
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)

    records, summary = build_records(direct_sample_dataset, config, sft_model, tokenizer)

    data_dir = osp.join(dirname, '..', 'LLaMAFactory', 'data')
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = f'alpo_train_{sft_model}_{trpe_model}_{lang}_{sft_proportion}'
    output_path = osp.join(data_dir, f'{dataset_name}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary_path = osp.join(output_dir, 'alpo_dataset_summary.json')
    json.dump(summary, open(summary_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(f'[6-alpo] Saved {len(records)} ALPO segment records to {output_path}')
