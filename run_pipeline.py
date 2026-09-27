"""
Amazon ML Challenge 2026 — Memory-Safe Pipeline Runner
Target: keep total system RAM <= 70% (~10.9 GB on 15.6 GB machine)

Memory strategy:
  - Build S2 indexes from targeted column reads (no full S2 df in RAM during retrieval)
  - Stream retrieval in small S1 chunks, writing parquet incrementally
  - Delete S2 indexes before building S3 indexes
  - Feature matrix written to memmap; never hold full matrix in RAM
  - LightGBM n_jobs=2 to cap CPU
  - Checkpoint every expensive stage; resume safely
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_config, paths, macro_f05, Timer
from src.data_loader import load_source_cached, load_ground_truth_cached
from src.candidate_gen import (
    SourceIndex,
    retrieve_candidates_streaming,
    merge_candidate_parquets,
    measure_candidate_recall,
)
from src.features import compute_features, make_lookup, FEATURE_NAMES, N_FEATURES
from src.training_pairs import stratified_s1_split, build_training_pairs, build_validation_pairs
from src.models import DeterministicBaseline, LRModel, LGBMModel
from src.threshold import (
    optimize_threshold,
    apply_threshold,
    write_output_files,
    candidates_df_to_dict,
    error_analysis,
)
from src.hard_negative_mining import mine_hard_negatives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def mem_report(label: str = "") -> float:
    m = psutil.virtual_memory()
    pct = m.percent
    s = f"RAM {m.used/1024**3:.2f}GB/{m.total/1024**3:.1f}GB ({pct:.1f}%)"
    if label:
        s = f"[{label}] {s}"
    print(f"  {s}")
    if pct > 72:
        print(f"  WARNING: RAM at {pct:.1f}% -- approaching 70% ceiling")
    return pct


# ---------------------------------------------------------------------------
# Chunked feature computation with memory-mapped output
# ---------------------------------------------------------------------------

def compute_features_memmap(
    candidates_df: pd.DataFrame,
    s1_lookup: dict,
    cand_lookup: dict,
    out_path: Path,
    chunk_size: int = 50000,
) -> np.ndarray:
    """Compute features in chunks, writing to memory-mapped .npy file."""
    n = len(candidates_df)
    fp = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float32, shape=(n, N_FEATURES)
    )

    empty = ("", "", "", "")
    s1_ids    = candidates_df["s1_id"].values
    cand_ids  = candidates_df["candidate_id"].values
    chans_arr = candidates_df["channels"].values
    nchan_arr = candidates_df["n_channels"].values

    t0 = time.perf_counter()
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for idx in range(chunk_start, chunk_end):
            s1_data   = s1_lookup.get(s1_ids[idx], empty)
            cand_data = cand_lookup.get(cand_ids[idx], empty)
            fp[idx] = compute_features(
                s1_name_norm=s1_data[0],
                s1_addr_norm=s1_data[1],
                s1_addr_dig =s1_data[2],
                s1_country  =s1_data[3],
                cand_name_norm=cand_data[0],
                cand_addr_norm=cand_data[1],
                cand_addr_dig =cand_data[2],
                cand_country  =cand_data[3],
                channels  =chans_arr[idx],
                n_channels=int(nchan_arr[idx]),
            )
        if chunk_end % 200000 < chunk_size or chunk_end == n:
            elapsed = time.perf_counter() - t0
            rate = chunk_end / max(elapsed, 0.001)
            eta  = (n - chunk_end) / max(rate, 0.001)
            print(f"  Features: {chunk_end:,}/{n:,}  "
                  f"{rate:.0f} pairs/s  ETA {eta/60:.1f}m")
            m = psutil.virtual_memory()
            if m.percent > 72:
                fp.flush()

    fp.flush()
    del fp
    gc.collect()
    result = np.lib.format.open_memmap(
        str(out_path), mode="r", dtype=np.float32, shape=(n, N_FEATURES)
    )
    return result


# ---------------------------------------------------------------------------
# Build lookup from parquet (only rows needed)
# ---------------------------------------------------------------------------

def build_lookup_from_parquet(
    parquet_path: Path,
    needed_ids: set[str] | None = None,
) -> dict[str, tuple]:
    """Load parquet, filter to needed IDs, return lookup dict."""
    df = pq.read_table(
        parquet_path,
        columns=["entity_id", "name_norm", "address_norm", "address_dig", "country"],
    ).to_pandas()
    if needed_ids is not None:
        df = df[df["entity_id"].isin(needed_ids)]
    result = dict(zip(
        df["entity_id"],
        zip(df["name_norm"], df["address_norm"], df["address_dig"], df["country"]),
    ))
    del df
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()
    cfg = get_config()
    p   = paths()

    for d in [p["experiments_dir"], p["models_dir"], p["reports_dir"]]:
        d.mkdir(parents=True, exist_ok=True)

    mem_report("pipeline start")

    # =========================================================================
    # STAGE 1: Load S1 + ground truth (both fit easily in RAM)
    # =========================================================================
    banner("STAGE 1: Data loading")

    with Timer("load S1 train"):
        s1_train = load_source_cached("train", "source1")
    mem_report("after S1 load")

    gt = load_ground_truth_cached("train")
    print(f"  S1: {len(s1_train):,}  GT: {len(gt):,} entries, "
          f"{sum(len(v) for v in gt.values()):,} links")

    zero_m   = sum(1 for v in gt.values() if len(v) == 0)
    single_m = sum(1 for v in gt.values() if len(v) == 1)
    multi_m  = sum(1 for v in gt.values() if len(v) > 1)
    print(f"  Zero-match: {zero_m:,} ({100*zero_m/len(gt):.2f}%)")
    print(f"  Singleton:  {single_m:,} ({100*single_m/len(gt):.2f}%)")
    print(f"  Multi:      {multi_m:,} ({100*multi_m/len(gt):.2f}%)")

    # =========================================================================
    # STAGE 2: Stratified S1 train/val split
    # =========================================================================
    banner("STAGE 2: Train/val split")

    train_cfg = cfg["training"]
    train_s1_ids, val_s1_ids = stratified_s1_split(
        gt,
        val_fraction=train_cfg["val_fraction"],
        seed=train_cfg["random_seed"],
    )
    train_s1_set = set(train_s1_ids)
    val_s1_set   = set(val_s1_ids)
    print(f"  Train: {len(train_s1_ids):,}  Val: {len(val_s1_ids):,}")

    # =========================================================================
    # STAGE 3: Candidate generation (streaming, sequential S2 then S3)
    # =========================================================================
    banner("STAGE 3: Candidate generation")

    rcfg = cfg["retrieval"]
    cache_dir = p["cache_dir"]

    cand_s2_path  = cache_dir / "train_candidates_s2.parquet"
    cand_s3_path  = cache_dir / "train_candidates_s3.parquet"
    cand_all_path = cache_dir / "train_candidates.parquet"

    s2_parquet = p["train_dir"] / "train_source2.tsv"   # raw TSV (used for cache check)
    s2_norm_parquet = cache_dir / "train_source2_norm.parquet"
    s3_norm_parquet = cache_dir / "train_source3_norm.parquet"

    if cand_all_path.exists():
        print(f"  Loading cached candidates: {cand_all_path}")
        with Timer("load candidates"):
            candidates_df = pd.read_parquet(cand_all_path)
        print(f"  Loaded {len(candidates_df):,} candidate pairs")
        mem_report("after candidate load")
    else:
        # --- S2 phase ---
        if not cand_s2_path.exists():
            print("  Building S2 indexes from parquet (no full df in RAM)...")
            idx_s2 = SourceIndex("S2")
            idx_s2.build_from_parquet(
                s2_norm_parquet,
                rare_token_min_len=rcfg["rare_token_min_len"],
                rare_token_max_df=rcfg["rare_token_max_df"],
                ngram_n=rcfg["ngram_size"],
                ngram_max_features=rcfg.get("ngram_max_features", 30000),
            )
            mem_report("after S2 index build")

            print("  Streaming S2 retrieval...")
            with Timer("S2 retrieval"):
                n_s2 = retrieve_candidates_streaming(
                    s1_train, idx_s2, cand_s2_path,
                    chunk_size=rcfg.get("retrieval_chunk_size", 5000),
                    ngram_batch_size=rcfg.get("ngram_batch_size", 2000),
                    ngram_top_k=rcfg["ngram_top_k"],
                    max_bucket=rcfg["max_bucket_size"],
                )
            del idx_s2
            gc.collect()
            mem_report("after S2 retrieval + idx freed")
            print(f"  S2 pairs: {n_s2:,}")
        else:
            print(f"  S2 cache exists: {cand_s2_path}")

        # --- S3 phase ---
        if not cand_s3_path.exists():
            print("  Building S3 indexes from parquet (no full df in RAM)...")
            idx_s3 = SourceIndex("S3")
            idx_s3.build_from_parquet(
                s3_norm_parquet,
                rare_token_min_len=rcfg["rare_token_min_len"],
                rare_token_max_df=rcfg["rare_token_max_df"],
                ngram_n=rcfg["ngram_size"],
                ngram_max_features=rcfg.get("ngram_max_features", 30000),
            )
            mem_report("after S3 index build")

            print("  Streaming S3 retrieval...")
            with Timer("S3 retrieval"):
                n_s3 = retrieve_candidates_streaming(
                    s1_train, idx_s3, cand_s3_path,
                    chunk_size=rcfg.get("retrieval_chunk_size", 5000),
                    ngram_batch_size=rcfg.get("ngram_batch_size", 2000),
                    ngram_top_k=rcfg["ngram_top_k"],
                    max_bucket=rcfg["max_bucket_size"],
                )
            del idx_s3
            gc.collect()
            mem_report("after S3 retrieval + idx freed")
            print(f"  S3 pairs: {n_s3:,}")
        else:
            print(f"  S3 cache exists: {cand_s3_path}")

        # --- Merge S2 + S3 ---
        print("  Merging S2+S3 candidates...")
        with Timer("merge candidates"):
            n_merged = merge_candidate_parquets(cand_s2_path, cand_s3_path, cand_all_path)
        print(f"  Merged: {n_merged:,} pairs")
        cand_s2_path.unlink(missing_ok=True)
        cand_s3_path.unlink(missing_ok=True)

        with Timer("load merged candidates"):
            candidates_df = pd.read_parquet(cand_all_path)
        print(f"  Total candidate pairs: {len(candidates_df):,}")
        mem_report("after merge + load")

    # =========================================================================
    # STAGE 4: Candidate recall measurement
    # =========================================================================
    banner("STAGE 4: Candidate recall")

    val_cands_sub = candidates_df[candidates_df["s1_id"].isin(val_s1_set)]
    recall_metrics = measure_candidate_recall(val_cands_sub, gt, list(val_s1_set))
    del val_cands_sub
    print(f"  Recall (val):      {recall_metrics['candidate_recall']:.4f}")
    print(f"  Mean cands/S1:     {recall_metrics['mean_candidates_per_s1']:.1f}")
    print(f"  P50/P95/P99:       {recall_metrics['p50_candidates']} / "
          f"{recall_metrics['p95_candidates']} / {recall_metrics['p99_candidates']}")
    print(f"  Recalled:          {recall_metrics['total_recalled']:,} / "
          f"{recall_metrics['total_gt_positives']:,}")
    save_json(recall_metrics, p["reports_dir"] / "candidate_recall_val.json")

    if recall_metrics["candidate_recall"] < 0.80:
        print("  WARNING: Candidate recall < 80% -- retrieval is the binding constraint")

    # =========================================================================
    # STAGE 5: Build entity lookup tables (targeted reads, only needed IDs)
    # =========================================================================
    banner("STAGE 5: Entity lookup tables")

    candidate_ids_all = set(candidates_df["candidate_id"].unique())
    print(f"  Unique candidate IDs: {len(candidate_ids_all):,}")

    s1_lookup = make_lookup(s1_train)
    print(f"  S1 lookup: {len(s1_lookup):,}")
    mem_report("after S1 lookup")

    print("  Building S2 lookup (needed IDs only)...")
    s23_lookup = build_lookup_from_parquet(s2_norm_parquet, candidate_ids_all)
    mem_report("after S2 lookup")

    print("  Building S3 lookup (needed IDs only)...")
    s3_lkp = build_lookup_from_parquet(s3_norm_parquet, candidate_ids_all)
    s23_lookup.update(s3_lkp)
    del s3_lkp, candidate_ids_all
    gc.collect()
    print(f"  S2+S3 lookup: {len(s23_lookup):,}")
    mem_report("after S2+S3 lookup")

    # =========================================================================
    # STAGE 6: Feature computation (chunked memmap)
    # =========================================================================
    banner("STAGE 6: Feature computation")

    feat_path = cache_dir / "train_features.npy"
    feat_meta = cache_dir / "train_features.meta.json"

    need_feats = True
    if feat_path.exists() and feat_meta.exists():
        meta = json.loads(feat_meta.read_text(encoding="utf-8"))
        if meta.get("n_pairs") == len(candidates_df) and meta.get("n_features") == N_FEATURES:
            print(f"  Feature cache valid: {len(candidates_df):,} x {N_FEATURES}")
            X_all = np.lib.format.open_memmap(
                str(feat_path), mode="r", dtype=np.float32,
                shape=(len(candidates_df), N_FEATURES),
            )
            need_feats = False
        else:
            print("  Feature cache stale -- recomputing")
            feat_path.unlink(missing_ok=True)

    if need_feats:
        with Timer("compute features"):
            X_all = compute_features_memmap(
                candidates_df, s1_lookup, s23_lookup, feat_path,
                chunk_size=50000,
            )
        feat_meta.write_text(json.dumps({
            "n_pairs": len(candidates_df),
            "n_features": N_FEATURES,
            "feature_names": FEATURE_NAMES,
        }), encoding="utf-8")
        print(f"  Features: {X_all.shape}")

    mem_report("after features")

    # =========================================================================
    # STAGE 7: Training + validation pairs
    # =========================================================================
    banner("STAGE 7: Training/validation pairs")

    # Build pair key -> row index mapping
    print("  Building pair index...")
    s1_ids_arr   = candidates_df["s1_id"].values
    cand_ids_arr = candidates_df["candidate_id"].values
    pair_key_to_idx: dict[tuple, int] = {}
    for i in range(len(candidates_df)):
        pair_key_to_idx[(s1_ids_arr[i], cand_ids_arr[i])] = i
    print(f"  Pair index: {len(pair_key_to_idx):,}")
    mem_report("after pair index")

    def extract_X(pairs_df: pd.DataFrame) -> np.ndarray:
        indices = []
        for row in pairs_df.itertuples(index=False):
            indices.append(pair_key_to_idx.get((row.s1_id, row.candidate_id), -1))
        arr   = np.array(indices, dtype=np.int64)
        valid = arr >= 0
        arr[~valid] = 0
        X = X_all[arr].copy()   # copy out of memmap
        X[~valid] = 0.0
        return X

    with Timer("build training pairs"):
        train_pairs_df, y_train = build_training_pairs(
            candidates_df, gt, train_s1_set,
            s1_lookup, s23_lookup,
            neg_ratio_random=train_cfg["neg_ratio_random"],
            neg_ratio_hard=train_cfg["neg_ratio_hard"],
            max_hard_per_s1=train_cfg["max_hard_negatives_per_s1"],
            seed=train_cfg["random_seed"],
        )
    X_train = extract_X(train_pairs_df)
    print(f"  X_train: {X_train.shape}  pos: {y_train.sum():,}")
    mem_report("after X_train")

    with Timer("build validation pairs"):
        val_pairs_df = build_validation_pairs(candidates_df, gt, val_s1_set)
    y_val = val_pairs_df["label"].values.astype(np.int32)

    val_cands_df = (
        candidates_df[candidates_df["s1_id"].isin(val_s1_set)]
        .reset_index(drop=True)
    )
    X_val_all = extract_X(val_cands_df)
    print(f"  X_val_all: {X_val_all.shape}  val pairs: {len(val_cands_df):,}")
    mem_report("after X_val_all")

    # =========================================================================
    # STAGE 8: Model ladder
    # =========================================================================
    banner("STAGE 8: Model training")

    mcfg  = cfg["model"]
    tcfg  = cfg["threshold"]
    results: dict = {}

    def run_threshold(name, scores):
        return optimize_threshold(
            val_cands_df, scores, gt, val_s1_set,
            search_min=tcfg["search_min"],
            search_max=tcfg["search_max"],
            search_steps=tcfg["search_steps"],
        )

    # L0 deterministic
    print("\n  [L0] Deterministic baseline")
    l0 = DeterministicBaseline()
    l0.fit(X_train, y_train)
    l0_scores = l0.predict_proba(X_val_all)[:, 1]
    l0_res    = run_threshold("L0", l0_scores)
    results["L0"] = {"threshold": l0_res["best_threshold"],
                     "macro_f05": l0_res["best_macro_f05"],
                     "metrics":   l0_res["final_metrics"]}
    print(f"  L0: threshold={l0_res['best_threshold']:.4f}  F0.5={l0_res['best_macro_f05']:.6f}")
    l0.save(p["models_dir"] / "l0_deterministic.pkl")
    mem_report("after L0")

    # L1 Logistic Regression
    print("\n  [L1] Logistic Regression")
    l1 = LRModel()
    l1.fit(X_train, y_train)
    l1_scores = l1.predict_proba(X_val_all)[:, 1]
    l1_res    = run_threshold("L1", l1_scores)
    results["L1"] = {"threshold": l1_res["best_threshold"],
                     "macro_f05": l1_res["best_macro_f05"],
                     "metrics":   l1_res["final_metrics"]}
    print(f"  L1: threshold={l1_res['best_threshold']:.4f}  F0.5={l1_res['best_macro_f05']:.6f}")
    l1.save(p["models_dir"] / "l1_logistic.pkl")
    mem_report("after L1")

    # L2 LightGBM (n_jobs=2 for memory/CPU safety)
    print("\n  [L2] LightGBM (n_jobs=2)")
    import lightgbm as lgb
    pos  = int(y_train.sum())
    neg  = int((y_train == 0).sum())
    with Timer("LightGBM fit"):
        lgbm_clf = lgb.LGBMClassifier(
            n_estimators    =mcfg["lgbm_n_estimators"],
            learning_rate   =mcfg["lgbm_learning_rate"],
            num_leaves      =mcfg["lgbm_num_leaves"],
            min_child_samples=mcfg["lgbm_min_child_samples"],
            subsample       =mcfg["lgbm_subsample"],
            colsample_bytree=mcfg["lgbm_colsample_bytree"],
            reg_alpha       =mcfg["lgbm_reg_alpha"],
            reg_lambda      =mcfg["lgbm_reg_lambda"],
            scale_pos_weight=neg / max(pos, 1),
            random_state    =mcfg["random_seed"],
            n_jobs=2,
            verbose=-1,
        )
        lgbm_clf.fit(X_train, y_train, feature_name=FEATURE_NAMES)
    mem_report("after LightGBM fit")

    l2_scores = lgbm_clf.predict_proba(X_val_all)[:, 1]
    l2_res    = run_threshold("L2", l2_scores)
    results["L2"] = {"threshold": l2_res["best_threshold"],
                     "macro_f05": l2_res["best_macro_f05"],
                     "metrics":   l2_res["final_metrics"]}
    print(f"  L2: threshold={l2_res['best_threshold']:.4f}  F0.5={l2_res['best_macro_f05']:.6f}")

    fi_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": lgbm_clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(p["reports_dir"] / "lgbm_feature_importance.csv", index=False)
    print("  Top 10 features:")
    print(fi_df.head(10).to_string(index=False))

    # Wrap for save/load compatibility
    l2 = LGBMModel.__new__(LGBMModel)
    l2._model = lgbm_clf
    l2.save(p["models_dir"] / "l2_lgbm.pkl")
    mem_report("after L2")

    save_json(results, p["reports_dir"] / "model_comparison.json")

    # =========================================================================
    # STAGE 9: Select best model
    # =========================================================================
    banner("STAGE 9: Select best model")

    scores_map = {
        "L0": results["L0"]["macro_f05"],
        "L1": results["L1"]["macro_f05"],
        "L2": results["L2"]["macro_f05"],
    }
    best_name = max(scores_map, key=scores_map.get)
    print(f"  Scores: {scores_map}")
    print(f"  Best:   {best_name}")

    model_obj_map   = {"L0": l0,       "L1": l1,       "L2": l2}
    thresh_res_map  = {"L0": l0_res,   "L1": l1_res,   "L2": l2_res}
    scores_arr_map  = {"L0": l0_scores,"L1": l1_scores,"L2": l2_scores}

    best_model      = model_obj_map[best_name]
    best_thresh_res = thresh_res_map[best_name]
    best_scores     = scores_arr_map[best_name]
    cur_threshold   = best_thresh_res["best_threshold"]
    cur_f05         = best_thresh_res["best_macro_f05"]
    print(f"  {best_name}  threshold={cur_threshold:.4f}  F0.5={cur_f05:.6f}")

    # =========================================================================
    # STAGE 10: Error analysis
    # =========================================================================
    banner("STAGE 10: Error analysis")

    with Timer("error analysis"):
        err = error_analysis(
            val_cands_df, best_scores, gt, val_s1_set,
            cur_threshold, s1_lookup, s23_lookup, max_examples=30,
        )
    print(f"  FP: {err['fp_count']:,}  types: {err['fp_types']}")
    print(f"  FN: {err['fn_count']:,}  types: {err['fn_types']}")
    save_json(err, p["reports_dir"] / "error_analysis_round1.json")

    # =========================================================================
    # STAGE 11: Hard-negative mining + retrain
    # =========================================================================
    banner("STAGE 11: Hard-negative mining")

    if best_name in ("L1", "L2"):
        train_cands_df = (
            candidates_df[candidates_df["s1_id"].isin(train_s1_set)]
            .reset_index(drop=True)
        )
        print(f"  Scoring {len(train_cands_df):,} training candidates...")
        X_tr_all    = extract_X(train_cands_df)
        train_scores = best_model.predict_proba(X_tr_all)[:, 1]
        del X_tr_all
        gc.collect()
        mem_report("after train scoring for HN")

        hard_neg_df = mine_hard_negatives(
            train_cands_df, train_scores, gt, train_s1_set,
            min_score=0.3, max_per_s1=5, top_n=50000,
        )
        del train_scores
        gc.collect()

        if len(hard_neg_df) > 0:
            X_hn   = extract_X(hard_neg_df)
            y_hn   = np.zeros(len(hard_neg_df), dtype=np.int32)
            X_aug  = np.vstack([X_train, X_hn])
            y_aug  = np.concatenate([y_train, y_hn])
            del X_hn
            gc.collect()
            print(f"  Augmented: {X_aug.shape}  pos: {y_aug.sum():,}")
            mem_report("after augmentation")

            pos_a = int(y_aug.sum())
            neg_a = int((y_aug == 0).sum())
            with Timer("LightGBM retrain HN"):
                lgbm_hn = lgb.LGBMClassifier(
                    n_estimators    =mcfg["lgbm_n_estimators"],
                    learning_rate   =mcfg["lgbm_learning_rate"],
                    num_leaves      =mcfg["lgbm_num_leaves"],
                    min_child_samples=mcfg["lgbm_min_child_samples"],
                    subsample       =mcfg["lgbm_subsample"],
                    colsample_bytree=mcfg["lgbm_colsample_bytree"],
                    reg_alpha       =mcfg["lgbm_reg_alpha"],
                    reg_lambda      =mcfg["lgbm_reg_lambda"],
                    scale_pos_weight=neg_a / max(pos_a, 1),
                    random_state    =mcfg["random_seed"],
                    n_jobs=2,
                    verbose=-1,
                )
                lgbm_hn.fit(X_aug, y_aug, feature_name=FEATURE_NAMES)
            del X_aug, y_aug
            gc.collect()
            mem_report("after HN retrain")

            hn_scores = lgbm_hn.predict_proba(X_val_all)[:, 1]
            hn_res    = run_threshold("L2+HN", hn_scores)
            hn_f05    = hn_res["best_macro_f05"]
            print(f"  L2+HN: threshold={hn_res['best_threshold']:.4f}  F0.5={hn_f05:.6f}")

            if hn_f05 > cur_f05:
                print(f"  HN improved F0.5: {cur_f05:.6f} -> {hn_f05:.6f}")
                best_name   = "L2+HN"
                best_scores = hn_scores
                cur_threshold = hn_res["best_threshold"]
                cur_f05       = hn_f05

                l2_hn = LGBMModel.__new__(LGBMModel)
                l2_hn._model = lgbm_hn
                l2_hn.save(p["models_dir"] / "l2_lgbm_hn.pkl")
                best_model = l2_hn

                results["L2+HN"] = {"threshold": hn_res["best_threshold"],
                                    "macro_f05": hn_f05,
                                    "metrics":   hn_res["final_metrics"]}
                save_json(results, p["reports_dir"] / "model_comparison.json")

                with Timer("error analysis round 2"):
                    err2 = error_analysis(
                        val_cands_df, best_scores, gt, val_s1_set,
                        cur_threshold, s1_lookup, s23_lookup, max_examples=30,
                    )
                print(f"  Round2 FP: {err2['fp_count']:,}  FN: {err2['fn_count']:,}")
                save_json(err2, p["reports_dir"] / "error_analysis_round2.json")
            else:
                print(f"  HN did NOT improve ({hn_f05:.6f} <= {cur_f05:.6f})")
        else:
            print("  No hard negatives above threshold")
    else:
        print("  Skipping HN (best is L0)")

    # =========================================================================
    # STAGE 12: Final validation summary
    # =========================================================================
    banner("STAGE 12: Validation summary")

    final_preds  = apply_threshold(val_cands_df, best_scores, cur_threshold, list(val_s1_set))
    gt_val       = {s1: set(v) for s1, v in gt.items() if s1 in val_s1_set}
    val_metrics  = macro_f05(final_preds, gt_val)
    print(f"  Model:       {best_name}")
    print(f"  Threshold:   {cur_threshold:.4f}")
    print(f"  Macro F0.5:  {val_metrics['macro_f05']:.6f}")
    print(f"  Precision:   {val_metrics['micro_precision']:.6f}")
    print(f"  Recall:      {val_metrics['micro_recall']:.6f}")
    print(f"  Zero-match:  {val_metrics['zero_match_accuracy']}")
    print(f"  N S1 (val):  {val_metrics['n_s1']}")

    save_json({"model": best_name, "threshold": cur_threshold,
               "val_metrics": val_metrics, "all_results": results},
              p["reports_dir"] / "final_val_summary.json")

    best_model.save(p["models_dir"] / "final_model.pkl")
    with open(p["models_dir"] / "final_config.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": best_name, "threshold": cur_threshold,
                   "val_macro_f05": cur_f05, "feature_names": FEATURE_NAMES},
                  f, indent=2)

    # Free training data before test inference
    del X_train, X_val_all, val_cands_df, val_pairs_df, train_pairs_df
    del pair_key_to_idx, s1_lookup, s23_lookup, candidates_df
    gc.collect()
    mem_report("after training data freed")

    # =========================================================================
    # STAGE 13: Test data inference
    # =========================================================================
    banner("STAGE 13: Test data inference")

    s1_test = load_source_cached("test", "source1")
    print(f"  Test S1: {len(s1_test):,}")
    mem_report("after test S1 load")

    s2_test_norm = cache_dir / "test_source2_norm.parquet"
    s3_test_norm = cache_dir / "test_source3_norm.parquet"
    test_s2_path = cache_dir / "test_candidates_s2.parquet"
    test_s3_path = cache_dir / "test_candidates_s3.parquet"
    test_all_path = cache_dir / "test_candidates.parquet"

    if test_all_path.exists():
        print("  Loading cached test candidates...")
        test_cands_df = pd.read_parquet(test_all_path)
        print(f"  Loaded {len(test_cands_df):,} test candidate pairs")
        mem_report("after test candidates load")
    else:
        # Ensure test normalization caches exist
        _ = load_source_cached("test", "source2")
        del _; gc.collect()
        _ = load_source_cached("test", "source3")
        del _; gc.collect()

        if not test_s2_path.exists():
            print("  Building test S2 indexes...")
            tidx_s2 = SourceIndex("S2")
            tidx_s2.build_from_parquet(
                s2_test_norm,
                rare_token_min_len=rcfg["rare_token_min_len"],
                rare_token_max_df=rcfg["rare_token_max_df"],
                ngram_n=rcfg["ngram_size"],
                ngram_max_features=rcfg.get("ngram_max_features", 30000),
            )
            mem_report("after test S2 index")
            with Timer("test S2 retrieval"):
                retrieve_candidates_streaming(
                    s1_test, tidx_s2, test_s2_path,
                    chunk_size=rcfg.get("retrieval_chunk_size", 5000),
                    ngram_batch_size=rcfg.get("ngram_batch_size", 2000),
                    ngram_top_k=rcfg["ngram_top_k"],
                    max_bucket=rcfg["max_bucket_size"],
                )
            del tidx_s2; gc.collect()
            mem_report("after test S2 retrieval")

        if not test_s3_path.exists():
            print("  Building test S3 indexes...")
            tidx_s3 = SourceIndex("S3")
            tidx_s3.build_from_parquet(
                s3_test_norm,
                rare_token_min_len=rcfg["rare_token_min_len"],
                rare_token_max_df=rcfg["rare_token_max_df"],
                ngram_n=rcfg["ngram_size"],
                ngram_max_features=rcfg.get("ngram_max_features", 30000),
            )
            mem_report("after test S3 index")
            with Timer("test S3 retrieval"):
                retrieve_candidates_streaming(
                    s1_test, tidx_s3, test_s3_path,
                    chunk_size=rcfg.get("retrieval_chunk_size", 5000),
                    ngram_batch_size=rcfg.get("ngram_batch_size", 2000),
                    ngram_top_k=rcfg["ngram_top_k"],
                    max_bucket=rcfg["max_bucket_size"],
                )
            del tidx_s3; gc.collect()
            mem_report("after test S3 retrieval")

        print("  Merging test S2+S3...")
        n_test = merge_candidate_parquets(test_s2_path, test_s3_path, test_all_path)
        test_s2_path.unlink(missing_ok=True)
        test_s3_path.unlink(missing_ok=True)
        test_cands_df = pd.read_parquet(test_all_path)
        print(f"  Test candidates: {len(test_cands_df):,}")
        mem_report("after test merge")

    # Build test lookups
    test_cand_ids = set(test_cands_df["candidate_id"].unique())
    s1_test_lookup = make_lookup(s1_test)
    s23_test_lookup = build_lookup_from_parquet(s2_test_norm, test_cand_ids)
    s3_tl = build_lookup_from_parquet(s3_test_norm, test_cand_ids)
    s23_test_lookup.update(s3_tl)
    del s3_tl, test_cand_ids
    gc.collect()
    print(f"  Test lookups: S1={len(s1_test_lookup):,}  S23={len(s23_test_lookup):,}")
    mem_report("after test lookups")

    # Compute test features
    test_feat_path = cache_dir / "test_features.npy"
    test_feat_meta = cache_dir / "test_features.meta.json"

    need_test_feats = True
    if test_feat_path.exists() and test_feat_meta.exists():
        tm = json.loads(test_feat_meta.read_text(encoding="utf-8"))
        if tm.get("n_pairs") == len(test_cands_df) and tm.get("n_features") == N_FEATURES:
            print("  Test feature cache valid")
            X_test = np.lib.format.open_memmap(
                str(test_feat_path), mode="r", dtype=np.float32,
                shape=(len(test_cands_df), N_FEATURES),
            )
            need_test_feats = False

    if need_test_feats:
        with Timer("compute test features"):
            X_test = compute_features_memmap(
                test_cands_df, s1_test_lookup, s23_test_lookup,
                test_feat_path, chunk_size=50000,
            )
        test_feat_meta.write_text(json.dumps({
            "n_pairs": len(test_cands_df),
            "n_features": N_FEATURES,
        }), encoding="utf-8")
    mem_report("after test features")

    # Score in batches
    print("  Scoring test candidates...")
    test_scores = np.zeros(len(test_cands_df), dtype=np.float32)
    batch = 100000
    for b0 in range(0, len(test_cands_df), batch):
        b1 = min(b0 + batch, len(test_cands_df))
        chunk = X_test[b0:b1].copy()
        test_scores[b0:b1] = best_model.predict_proba(chunk)[:, 1]
        del chunk
        if b0 % 500000 < batch:
            print(f"  Scored {b1:,}/{len(test_cands_df):,}")
    mem_report("after test scoring")

    all_test_s1 = list(s1_test["entity_id"])
    test_preds  = apply_threshold(test_cands_df, test_scores, cur_threshold, all_test_s1)
    n_matched   = sum(1 for v in test_preds.values() if v)
    n_zero      = sum(1 for v in test_preds.values() if not v)
    print(f"  Test S1s: {len(test_preds):,}  matched: {n_matched:,}  zero: {n_zero:,}")

    test_cand_dict = candidates_df_to_dict(test_cands_df, all_test_s1)

    # =========================================================================
    # STAGE 14: Write outputs
    # =========================================================================
    banner("STAGE 14: Write output files")

    write_output_files(test_preds, test_cand_dict, p["output_dir"])

    # =========================================================================
    # STAGE 15: Official validator
    # =========================================================================
    banner("STAGE 15: Official validator")

    import subprocess
    matching_path  = p["output_dir"] / "matching_results.tsv"
    candidate_path = p["output_dir"] / "candidate_pairs.tsv"
    result = subprocess.run(
        [sys.executable, str(p["validator"]),
         "--matching",   str(matching_path),
         "--candidate",  str(candidate_path),
         "--test-dir",   str(p["test_dir"])],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])
    validator_pass = result.returncode == 0
    print("  VALIDATOR:", "PASS" if validator_pass else "FAIL")

    # =========================================================================
    # Done
    # =========================================================================
    total_elapsed = time.perf_counter() - t_total
    banner(f"PIPELINE COMPLETE -- {total_elapsed/60:.1f} minutes")
    print(f"  Model:      {best_name}")
    print(f"  Val F0.5:   {cur_f05:.6f}")
    print(f"  Threshold:  {cur_threshold:.4f}")
    print(f"  Output:     {p['output_dir']}")

    summary = {
        "best_model": best_name,
        "threshold":  cur_threshold,
        "val_macro_f05": cur_f05,
        "val_metrics":   val_metrics,
        "candidate_recall_val": recall_metrics,
        "total_runtime_minutes": round(total_elapsed / 60, 2),
        "validator_pass": validator_pass,
    }
    save_json(summary, p["reports_dir"] / "pipeline_summary.json")
    return summary


if __name__ == "__main__":
    main()
