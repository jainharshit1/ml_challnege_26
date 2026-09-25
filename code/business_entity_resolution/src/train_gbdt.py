"""Stage-B LightGBM training (plan §7.2, §7.3).

Trains on group G pairs (from artifacts/final_scores/*features.parquet with
y column), with early stopping on a 10% held-out slice of G *entities*
(entity-based split, not pair-based, so pairs from the same S1 stay together).

Also provides a LOCO training entry point for the France-proxy runs.

Output:
    artifacts/models/stage_b_lgbm.txt
    artifacts/models/stage_b_features.txt
    artifacts/models/stage_b_feature_importance.tsv
    (LOCO variants: stage_b_lgbm__loco_{country}.txt)

Scoring:
    Writes artifacts/final_scores/{split}__{country}__scores.parquet
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import splits as splits_mod
from .io_utils import write_parquet


def _load_group_features(group: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    split_df = splits_mod.load()
    ids = set(split_df.loc[split_df["group"] == group, "entity_id"])
    parts_x, parts_k, parts_y = [], [], []
    for p in C.FINAL_SCORES_DIR.glob("train__*__features.parquet"):
        df = pd.read_parquet(p)
        df = df[df["s1_id"].isin(ids)]
        if df.empty:
            continue
        parts_k.append(df[["s1_id", "cand_id", "cand_source"]])
        parts_y.append(df["y"].values)
        parts_x.append(df.drop(columns=["s1_id", "cand_id", "cand_source", "y"]))
    if not parts_x:
        raise RuntimeError(f"No G features found for group {group}")
    X = pd.concat(parts_x, ignore_index=True).astype("float32")
    K = pd.concat(parts_k, ignore_index=True)
    y = np.concatenate(parts_y)
    return X, K, y


def _entity_split(K: pd.DataFrame, val_frac: float = 0.1
                  ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(C.SEED)
    entities = K["s1_id"].unique()
    rng.shuffle(entities)
    n_val = int(len(entities) * val_frac)
    val_set = set(entities[:n_val])
    mask = K["s1_id"].isin(val_set).values
    return ~mask, mask


def train(loco_country: str | None = None) -> Path:
    import lightgbm as lgb
    C.ensure_dirs()

    print(f"[train_gbdt] loading G features (loco={loco_country})…")
    X, K, y = _load_group_features("G")
    if loco_country:
        # LOCO: train on the ONE country that is NOT the target country
        split_df = splits_mod.load().set_index("entity_id")
        countries = K["s1_id"].map(split_df["country"])
        keep = countries != loco_country
        X, K, y = X[keep].reset_index(drop=True), K[keep].reset_index(drop=True), y[keep]
    print(f"[train_gbdt] training pairs: {len(X):,}  positives: {int(y.sum()):,}")

    tr, va = _entity_split(K)
    dtr = lgb.Dataset(X[tr].values, label=y[tr])
    dva = lgb.Dataset(X[va].values, label=y[va], reference=dtr)
    device = C.lgb_device()
    print(f"[train_gbdt] LightGBM device: {device}")
    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        device_type=device,
        num_leaves=C.LGB_NUM_LEAVES,
        learning_rate=C.LGB_LR,
        feature_fraction=C.LGB_FEATURE_FRACTION,
        bagging_fraction=C.LGB_BAGGING_FRACTION,
        bagging_freq=1,
        verbose=-1,
        seed=C.SEED,
    )
    booster = lgb.train(
        params, dtr,
        num_boost_round=C.LGB_N_ESTIMATORS,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(C.LGB_EARLY_STOP), lgb.log_evaluation(200)],
    )
    tag = f"__loco_{loco_country}" if loco_country else ""
    model_path = C.MODELS_DIR / f"stage_b_lgbm{tag}.txt"
    booster.save_model(str(model_path))
    (C.MODELS_DIR / f"stage_b_features{tag}.txt").write_text(
        "\n".join(X.columns), encoding="utf-8"
    )
    imp = pd.DataFrame({
        "feature": X.columns,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(C.MODELS_DIR / f"stage_b_feature_importance{tag}.tsv",
               sep="\t", index=False)
    print(f"[train_gbdt] wrote {model_path}")
    return model_path


def _load_booster(loco_country: str | None = None):
    import lightgbm as lgb
    tag = f"__loco_{loco_country}" if loco_country else ""
    return lgb.Booster(model_file=str(C.MODELS_DIR / f"stage_b_lgbm{tag}.txt"))


def score_all(loco_country: str | None = None) -> None:
    tag = f"__loco_{loco_country}" if loco_country else ""
    booster = _load_booster(loco_country)
    feat_names = (C.MODELS_DIR / f"stage_b_features{tag}.txt").read_text(
        encoding="utf-8"
    ).split()
    for p in C.FINAL_SCORES_DIR.glob("*__features.parquet"):
        split, country, _ = p.stem.split("__")
        df = pd.read_parquet(p)
        X = df.drop(columns=["s1_id", "cand_id", "cand_source"]
                    + (["y"] if "y" in df.columns else [])
                    ).reindex(columns=feat_names).fillna(-1.0).astype("float32")
        p_hat = booster.predict(X.values)
        out = df[["s1_id", "cand_id", "cand_source"]].copy()
        out["p_match"] = p_hat.astype(np.float32)
        out_path = C.FINAL_SCORES_DIR / f"{split}__{country}__scores{tag}.parquet"
        write_parquet(out, out_path)
        print(f"[train_gbdt] scored {out_path}  ({len(out):,} pairs)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "loco":
        # LOCO variants
        for c in ("US", "India"):
            train(loco_country=c)
            score_all(loco_country=c)
    else:
        train()
        score_all()
