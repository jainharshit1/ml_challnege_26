"""End-to-end orchestration.

Runs the full pipeline in dependency order. Each stage is idempotent: re-runs
skip already-materialised parquet files, so you can crash and restart. Use
--only STAGE to run one stage.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from . import config as C


STAGES = [
    "stage0",
    "splits",
    "abbrev_seed",     # first pass: seed abbreviations only
    "normalize_first",
    "abbrev_mine",     # mine on R∪G, refresh abbreviations.json
    "normalize_final", # re-normalize with mined abbreviations
    "idf",
    "embed",
    "block",
    "prefilter_train",
    "prefilter_score", # scores all (train/test/all countries) & writes candidate_pairs.tsv
    "reranker_train",
    "reranker_score",
    "features",
    "gbdt_train",
    "gbdt_score",
    "tune",
    "loco",            # optional: train + score both LOCO variants
    "france",          # France procedure + diagnostics
    "decide",          # write output/matching_results.tsv
]


def _run(stage: str) -> None:
    t0 = time.time()
    print(f"\n===== [{stage}] =====")

    if stage == "stage0":
        from . import stage0_checks
        stage0_checks.run()

    elif stage == "splits":
        from . import splits
        splits.build()

    elif stage == "abbrev_seed":
        # Write seed-only abbreviations.json so first normalize pass works.
        from . import abbreviations as A
        C.ensure_dirs()
        (C.ABBREV_PATH).write_text(
            json.dumps({"name": dict(A.SEED_NAME),
                        "address": dict(A.SEED_ADDR)}, indent=2),
            encoding="utf-8",
        )
        print(f"[abbrev_seed] wrote seed-only {C.ABBREV_PATH}")

    elif stage == "normalize_first":
        from . import normalize
        normalize.normalize_all()

    elif stage == "abbrev_mine":
        from . import abbreviations
        abbreviations.build()

    elif stage == "normalize_final":
        from . import normalize
        # remove first-pass parquets; they were written with seed-only mapping
        for p in C.NORMALIZED_DIR.glob("*.parquet"):
            p.unlink()
        normalize.normalize_all()

    elif stage == "idf":
        from . import normalize
        normalize.compute_and_save_idf()

    elif stage == "embed":
        from . import embed
        embed.embed_all()

    elif stage == "block":
        from . import blocking
        blocking.block_all_partitions()
        blocking.blocking_recall_report()

    elif stage == "prefilter_train":
        from . import prefilter
        prefilter.train()

    elif stage == "prefilter_score":
        from . import prefilter
        from joblib import Parallel, delayed
        countries = sorted({p.stem.split("__")[-1] for p in
                            C.BLOCKING_DIR.glob("*__*.parquet")})
        jobs = [(split, country) for split in ("train", "test") for country in countries
                if (C.BLOCKING_DIR / f"{split}__{country}.parquet").exists()]
        import os as _os
        n_jobs = int(_os.environ.get("PREFILTER_JOBS", "3"))
        n_jobs = min(len(jobs), max(1, n_jobs))
        print(f"[prefilter_score] {len(jobs)} (split,country) jobs on {n_jobs} parallel workers")
        Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(prefilter.score_partition)(s, c) for s, c in jobs
        )
        prefilter.prefilter_recall_report()
        prefilter.write_candidate_pairs_tsv()

    elif stage == "reranker_train":
        from . import reranker_train
        reranker_train.train()

    elif stage == "reranker_score":
        from . import reranker_infer
        reranker_infer.score_all()

    elif stage == "features":
        from . import features as F
        from joblib import Parallel, delayed
        countries = sorted({p.stem.split("__")[-1] for p in
                            C.PREFILTER_DIR.glob("*__*.parquet")})
        jobs = [(split, country) for split in ("train", "test") for country in countries
                if (C.PREFILTER_DIR / f"{split}__{country}.parquet").exists()]
        # Each worker loads ~15 GB of DFs; cap by RAM/60GB assumption + user knob
        import os as _os
        n_jobs = int(_os.environ.get("FEATURES_JOBS", "3"))
        n_jobs = min(len(jobs), max(1, n_jobs))
        print(f"[features] {len(jobs)} (split,country) jobs on {n_jobs} parallel workers")
        Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(F.build_and_save)(s, c) for s, c in jobs
        )

    elif stage == "gbdt_train":
        from . import train_gbdt
        train_gbdt.train()

    elif stage == "gbdt_score":
        from . import train_gbdt
        train_gbdt.score_all()

    elif stage == "tune":
        from . import tune_thresholds
        r = tune_thresholds.tune()
        Path(C.REPORTS_DIR / "thresholds_in_country.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8"
        )

    elif stage == "loco":
        from . import train_gbdt, tune_thresholds
        for c in ("US", "India"):
            train_gbdt.train(loco_country=c)
            train_gbdt.score_all(loco_country=c)
            tune_thresholds.tune(loco_tag=f"__loco_{c}")

    elif stage == "france":
        from . import tune_thresholds
        fr = tune_thresholds.france_procedure()
        tune_thresholds.france_diagnostic(
            fr["france_pair"], fr["france_singleton"], fr["france_margin"]
        )
        (C.REPORTS_DIR / "france_thresholds.json").write_text(
            json.dumps(fr, indent=2), encoding="utf-8"
        )

    elif stage == "decide":
        from . import decide as D, tune_thresholds
        inc = json.loads(
            (C.REPORTS_DIR / "thresholds_in_country.json").read_text("utf-8")
        )
        fr_path = C.REPORTS_DIR / "france_thresholds.json"
        fr_overrides = None
        if fr_path.exists():
            fr = json.loads(fr_path.read_text("utf-8"))
            fr_overrides = (fr["france_pair"], fr["france_singleton"],
                             fr["france_margin"])
        D.decide_all_test(
            pair_thresh=inc["pair_thresh"],
            singleton_thresh=inc["singleton_thresh"],
            margin_thresh=inc.get("margin_thresh"),
            france_overrides=fr_overrides,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")

    _dt = time.time() - t0
    print(f"===== [{stage}] done in {_dt:.1f}s =====")

    # Append timing to stage_timings.json
    import json as _json
    tf = C.REPORTS_DIR / "stage_timings.json"
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timings = _json.loads(tf.read_text()) if tf.exists() else {}
    timings[stage] = {"seconds": _dt, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tf.write_text(_json.dumps(timings, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these stages (in the order given).")
    parser.add_argument("--from-stage", default=None,
                        help="Skip stages before this one.")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Stages to skip.")
    args = parser.parse_args()

    if args.only:
        stages = args.only
    else:
        stages = STAGES
        if args.from_stage:
            i = stages.index(args.from_stage)
            stages = stages[i:]
        stages = [s for s in stages if s not in args.skip]

    for s in stages:
        _run(s)


if __name__ == "__main__":
    main()
