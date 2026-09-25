"""Per-entity F_0.5, macro-averaged over S1 entities (plan §9).

Follows the challenge definition:
  - true singleton predicted empty: score 1.0
  - true singleton predicted non-empty: score 0.0
  - non-singleton predicted empty: score 0.0
  - otherwise: F_0.5 on the two ID sets

The macro average is over ALL S1 entities in the eval set (singletons count).
"""
from __future__ import annotations
from collections.abc import Iterable, Mapping

BETA2 = 0.25       # β = 0.5 → β² = 0.25


def entity_f_beta(pred: set[str], true: set[str], beta2: float = BETA2) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def macro_f_beta(preds: Mapping[str, Iterable[str]],
                 truths: Mapping[str, Iterable[str]]) -> dict:
    """Return dict with overall + slice metrics.

    - keys of `preds` and `truths` are the S1 entity IDs to score.
    - if `preds` is missing an S1 id present in truths, that S1 scores 0
      unless it is a true singleton.
    """
    all_ids = set(truths.keys())
    per_entity: dict[str, float] = {}
    singleton_scores: list[float] = []
    nonsingleton_scores: list[float] = []
    for sid in all_ids:
        true_set = set(truths.get(sid, []))
        pred_set = set(preds.get(sid, []))
        s = entity_f_beta(pred_set, true_set)
        per_entity[sid] = s
        if not true_set:
            singleton_scores.append(s)
        else:
            nonsingleton_scores.append(s)

    def _avg(x): return sum(x) / len(x) if x else 0.0

    return {
        "macro_f_0_5": _avg(list(per_entity.values())),
        "singleton_f_0_5": _avg(singleton_scores),
        "nonsingleton_f_0_5": _avg(nonsingleton_scores),
        "n_entities": len(per_entity),
        "n_singletons_true": len(singleton_scores),
        "n_predicted_empty": sum(1 for sid in all_ids if not preds.get(sid)),
        "predicted_singleton_rate": (
            sum(1 for sid in all_ids if not preds.get(sid)) / max(len(all_ids), 1)
        ),
        "mean_predicted_matches": (
            sum(len(preds.get(sid, [])) for sid in all_ids) / max(len(all_ids), 1)
        ),
        "per_entity": per_entity,
    }
