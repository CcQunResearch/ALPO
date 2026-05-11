import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def split_target_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text).split('\n') if line.strip()]


def get_average_scores(segment: Dict[str, Any]) -> Optional[List[float]]:
    score_block = segment.get('vividness scores')
    if not isinstance(score_block, dict):
        return None
    scores = score_block.get('average')
    if not isinstance(scores, list):
        return None
    targets = segment.get('segment targets', [])
    if len(scores) != len(targets):
        return None
    try:
        return [float(score) for score in scores]
    except (TypeError, ValueError):
        return None


def is_trainable_segment(
    segment: Dict[str, Any],
    min_candidates: int = 3,
    min_score_gap: float = 5.0,
) -> bool:
    targets = segment.get('segment targets', [])
    scores = get_average_scores(segment)
    if scores is None:
        return False
    if len(targets) <= min_candidates:
        return False
    return max(scores) - min(scores) > min_score_gap


def select_chosen_rejected_indices(
    scores: Sequence[float],
    targets: Sequence[str],
    rng: random.Random,
    chosen_top_k: int = 3,
    rejected_rank: int = 3,
) -> Tuple[int, int]:
    if not scores:
        raise ValueError('scores must not be empty')
    if len(scores) != len(targets):
        raise ValueError('scores and targets must have the same length')

    indices = list(range(len(scores)))
    ranked_desc = sorted(indices, key=lambda idx: (scores[idx], -len(targets[idx])), reverse=True)
    top_k = max(1, min(chosen_top_k, len(ranked_desc)))
    chosen_index = rng.choice(ranked_desc[:top_k])

    ranked_asc = sorted(indices, key=lambda idx: (scores[idx], len(targets[idx])))
    rank_pos = min(max(rejected_rank, 1) - 1, len(ranked_asc) - 1)
    rejected_index = ranked_asc[rank_pos]
    if rejected_index == chosen_index and len(ranked_asc) > 1:
        rejected_index = next(idx for idx in ranked_asc if idx != chosen_index)
    return chosen_index, rejected_index


def compute_segment_weights(candidate_counts: Sequence[int], gates: Sequence[bool]) -> List[float]:
    if len(candidate_counts) != len(gates):
        raise ValueError('candidate_counts and gates must have the same length')
    denominator = sum(max(0, int(count)) for count in candidate_counts)
    if denominator <= 0:
        return [0.0 for _ in candidate_counts]
    return [
        (float(count) / denominator) if gate else 0.0
        for count, gate in zip(candidate_counts, gates)
    ]


def compute_dynamic_betas(gaps: Sequence[float], gates: Sequence[bool]) -> List[float]:
    if len(gaps) != len(gates):
        raise ValueError('gaps and gates must have the same length')
    active_gaps = [float(gap) for gap, gate in zip(gaps, gates) if gate and gap > 0]
    if not active_gaps:
        return [0.0 for _ in gaps]
    max_gap = max(active_gaps)
    return [
        (float(gap) / max_gap) if gate and gap > 0 else 0.0
        for gap, gate in zip(gaps, gates)
    ]


def linear_schedule(start: float, end: float, steps: int) -> List[float]:
    if steps <= 1:
        return [float(end)]
    stride = (float(end) - float(start)) / (steps - 1)
    return [float(start) + stride * i for i in range(steps)]


def format_numbered_lines(lines: Iterable[str], start_index: int = 1, suffix: str = '') -> str:
    normalized_lines = [str(line).strip() for line in lines if str(line).strip()]
    text = '\n'.join(
        f'{start_index + offset}. {line}'
        for offset, line in enumerate(normalized_lines)
    )
    return f'{text}{suffix}'


def choose_prefix_lines(
    candidate_lines: Sequence[List[str]],
    chosen_lines: List[str],
    lambda_value: float,
    rng: random.Random,
) -> List[str]:
    if not candidate_lines:
        return chosen_lines
    if rng.random() < lambda_value:
        return chosen_lines
    return list(rng.choice(list(candidate_lines)))
