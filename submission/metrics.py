#!/usr/bin/env python3
"""
metrics.py — local helpers for AUC/AP and Jaccard across single, many, aggregate files,
and a single paired evaluation (test_dataset_k ↔ train_dataset_k).

Key features:
- Auto-detect CSV vs TSV by extension.
- Auto-detect ID/probability/dataset columns with clear errors listing available columns.
- Aggregate mode can infer dataset per repertoire_id by scanning tests_root/ test_dataset_*/metadata.csv when your
  aggregate file lacks a useful dataset column (e.g., it's a constant 'test_datasets').
- Still supports explicit --id_col / --prob_col / --label_col if you want to force names.

Examples for your layout:

  # AUC/AP and Jaccard for ONE pair (e.g., test_dataset_1 ↔ train_dataset_1)
  python metrics.py pair-eval \
    --test_ds test_dataset_1 \
    --preds_file outputs/test_dataset_1/predictions.csv \
    --tests_root data/test_datasets/test_datasets \
    --sample_sub data/sample_submissions.csv \
    --train_ranked outputs/train_dataset_1/ranked_sequences.csv \
    --truth_root /path/to/reference_ranked \
    --k 50000

  # AUC per dataset from aggregate predictions
  python metrics.py auc-agg \
    --preds_agg /data/home/qp241528/AIRR/outputs/train_datasets_test_predictions.tsv \
    --tests_root /data/home/qp241528/AIRR/data/test_datasets/test_datasets

  # If your aggregate uses different column names, pass them explicitly:
  python metrics.py auc-agg \
    --preds_agg /.../train_datasets_test_predictions.tsv \
    --tests_root /.../test_datasets/test_datasets \
    --id_col ID \
    --prob_col label_positive_probability

  # Jaccard per dataset from aggregate sequences (needs truth ranked lists laid out as /truth_root/train_dataset_*/ranked_sequences.csv):
  python metrics.py jaccard-agg \
    --seqs_agg /data/home/qp241528/AIRR/outputs/train_datasets_important_sequences.tsv \
    --truth_root /path/to/reference_ranked \
    --k 50000
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------- helpers ----------------------

def _read_table(path: str) -> pd.DataFrame:
    """Read TSV if file endswith .tsv (case-insensitive), else CSV."""
    sep = "\t" if path.lower().endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"})


def _rank_keys(df: pd.DataFrame, cols=("junction_aa","v_call","j_call")) -> List[str]:
    return (df[cols[0]].astype(str) + "|" +
            df[cols[1]].astype(str) + "|" +
            df[cols[2]].astype(str)).tolist()


def _detect_col(df: pd.DataFrame, preferred: str, candidates: List[str], purpose: str) -> str:
    """Return a column name present in df: prefer `preferred` if provided; else first match in candidates.
       Raise helpful error listing available columns.
    """
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No {purpose} column found. Tried: {[preferred] + candidates if preferred else candidates}. "
                   f"Available columns: {list(df.columns)}")


def _detect_dataset_column(df: pd.DataFrame) -> str:
    for c in ["dataset","test_dataset","dataset_name","set","split"]:
        if c in df.columns:
            return c
    raise KeyError("No dataset column found (tried: dataset, test_dataset, dataset_name, set, split). "
                   f"Available columns: {list(df.columns)}")


def _build_rep_to_dataset_map(tests_root: str, id_col: str = "repertoire_id") -> Dict[str, str]:
    """Map repertoire_id -> test_dataset_* folder by reading each metadata.csv under tests_root."""
    mapping = {}
    for name in sorted(os.listdir(tests_root)):
        dpath = os.path.join(tests_root, name)
        if not (os.path.isdir(dpath) and name.startswith("test_dataset_")):
            continue
        meta = os.path.join(dpath, "metadata.csv")
        if not os.path.isfile(meta):
            continue
        m = _read_table(meta)
        # tolerate variation in id col in metadata
        id_truth = id_col if id_col in m.columns else None
        if id_truth is None:
            for alt in ["repertoire_id","ID","id","rep_id","repertoire","sample_id"]:
                if alt in m.columns:
                    id_truth = alt
                    break
        if id_truth is None:
            # skip if no usable id column
            continue
        for rid in m[id_truth].astype(str):
            mapping[rid] = name
    return mapping


# ---------------------- single-file metrics ----------------------

def compute_auc(pred_csv: str,
                truth_csv: str,
                id_col: str = "repertoire_id",
                prob_col: str = "label_positive_probability",
                label_col: str = "label_positive") -> dict:
    p = _read_table(pred_csv)
    t = _read_table(truth_csv)

    id_col   = _detect_col(p, id_col,  ["repertoire_id","ID","id","rep_id","repertoire","sample_id"], "ID (pred)")
    prob_col = _detect_col(p, prob_col,["label_positive_probability","prob","probability","pred","prediction","score"], "probability (pred)")

    # Detect id/label in truth
    id_truth = id_col if id_col in t.columns else _detect_col(t, None, [id_col,"repertoire_id","ID","id","rep_id","repertoire","sample_id"], "ID (truth)")
    label_col = _detect_col(t, label_col, ["label_positive","label","y","target"], "label (truth)")

    df = p[[id_col, prob_col]].merge(t[[id_truth, label_col]], left_on=id_col, right_on=id_truth, how="inner")
    if df.empty:
        raise ValueError("No overlapping IDs between predictions and truth.")

    y_true  = df[label_col].astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"}).astype(int)
    y_score = df[prob_col].astype(float)

    if y_true.nunique() < 2:
        raise ValueError("Truth labels contain a single class; ROC AUC is undefined.")

    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    return {"auc_roc": float(auc), "average_precision": float(ap), "n": int(len(df))}


def compute_jaccard(submitted_ranked_csv: str,
                    truth_ranked_csv: str,
                    k: int = 50000,
                    cols = ("junction_aa","v_call","j_call")) -> dict:
    sub = _read_table(submitted_ranked_csv).copy()
    tru = _read_table(truth_ranked_csv).copy()

    for c in cols:
        if c not in sub.columns:
            raise ValueError(f"Column '{c}' not in submitted CSV. Available: {list(sub.columns)}")
        if c not in tru.columns:
            raise ValueError(f"Column '{c}' not in truth CSV. Available: {list(tru.columns)}")

    sub = sub.head(k)
    tru = tru.head(k)

    s_keys = set(_rank_keys(sub, cols))
    t_keys = set(_rank_keys(tru, cols))
    inter  = len(s_keys & t_keys)
    union  = len(s_keys | t_keys)
    j      = inter / union if union else 0.0
    return {"jaccard": float(j), "intersection": int(inter), "union": int(union),
            "k": int(min(k, len(s_keys), len(t_keys)))}


# ---------------------- many-dataset (tree) ----------------------

def compute_auc_many(preds_root: str,
                     tests_root: str,
                     pred_filename: str = "predictions.csv",
                     id_col: str = "repertoire_id",
                     prob_col: str = "label_positive_probability",
                     label_col: str = "label_positive") -> pd.DataFrame:
    rows = []
    for dirpath, _, filenames in os.walk(preds_root):
        if pred_filename in filenames:
            pred_path = os.path.join(dirpath, pred_filename)
            ds_name   = os.path.basename(os.path.dirname(pred_path))
            truth_path= os.path.join(tests_root, ds_name, "metadata.csv")
            if not os.path.isfile(truth_path):
                rows.append({"dataset": ds_name, "status": "missing_truth",
                             "auc_roc": np.nan, "average_precision": np.nan, "n": 0})
                continue
            try:
                res = compute_auc(pred_path, truth_path, id_col=id_col, prob_col=prob_col, label_col=label_col)
                rows.append({"dataset": ds_name, "status": "ok", **res})
            except Exception as e:
                rows.append({"dataset": ds_name, "status": f"error: {e}",
                             "auc_roc": np.nan, "average_precision": np.nan, "n": 0})
    out = pd.DataFrame(rows).sort_values("dataset")
    out.to_csv(os.path.join(preds_root, "auc_summary.csv"), index=False)
    return out


def compute_jaccard_many(subs_root: str,
                         truth_root: str,
                         sub_filename: str = "ranked_sequences.csv",
                         k: int = 50000,
                         cols = ("junction_aa","v_call","j_call")) -> pd.DataFrame:
    rows = []
    for dirpath, _, filenames in os.walk(subs_root):
        if sub_filename in filenames:
            sub_path = os.path.join(dirpath, sub_filename)
            ds_name  = os.path.basename(os.path.dirname(sub_path))
            truth_path = os.path.join(truth_root, ds_name, sub_filename)
            if not os.path.isfile(truth_path):
                rows.append({"dataset": ds_name, "status": "missing_truth",
                             "jaccard": np.nan, "intersection": 0, "union": 0, "k": k})
                continue
            try:
                res = compute_jaccard(sub_path, truth_path, k=k, cols=cols)
                rows.append({"dataset": ds_name, "status": "ok", **res})
            except Exception as e:
                rows.append({"dataset": ds_name, "status": f"error: {e}",
                             "jaccard": np.nan, "intersection": 0, "union": 0, "k": k})
    out = pd.DataFrame(rows).sort_values("dataset")
    out.to_csv(os.path.join(subs_root, "jaccard_summary.csv"), index=False)
    return out


# ---------------------- aggregate-friendly ----------------------

def compute_auc_from_aggregate(preds_agg_path: str,
                               tests_root: str,
                               id_col: str = "repertoire_id",
                               prob_col: str = "label_positive_probability",
                               label_col: str = "label_positive") -> pd.DataFrame:
    """Compute per-dataset AUC from a single aggregate predictions file.
       If a dataset column is missing OR useless (e.g., single constant like 'test_datasets'),
       we infer dataset names by mapping repertoire IDs via tests_root metadata.
    """
    P = _read_table(preds_agg_path)

    # Detect/confirm ID & prob columns
    id_col   = _detect_col(P, id_col,  ["repertoire_id","ID","id","rep_id","repertoire","sample_id"], "ID (preds_agg)")
    prob_col = _detect_col(P, prob_col,["label_positive_probability","prob","probability","pred","prediction","score"], "probability (preds_agg)")

    # Try to use dataset column if present and meaningful; otherwise infer
    use_inferred = False
    try:
        ds_col = _detect_dataset_column(P)
        uniq = P[ds_col].astype(str).str.strip().unique().tolist()
        if (len(uniq) == 1) or (not any(str(v).startswith("test_dataset_") for v in uniq)):
            use_inferred = True
    except KeyError:
        use_inferred = True

    if use_inferred:
        rep2ds = _build_rep_to_dataset_map(tests_root, id_col="repertoire_id")  # test metadata typically uses 'repertoire_id'
        ds_col = "_dataset"
        P[ds_col] = P[id_col].astype(str).map(rep2ds)
        if P[ds_col].isna().all():
            raise ValueError(
                "Could not infer dataset for any repertoire ID from tests_root metadata.\n"
                f"Pred columns: {list(P.columns)}\n"
                f"tests_root: {tests_root}\n"
                "Fix by adding a per-row dataset column to the aggregate, or pass --id_col that matches test metadata."
            )
    else:
        P[ds_col] = P[ds_col].astype(str)

    rows = []
    for ds_name, g in P.groupby(ds_col):
        truth_path = os.path.join(tests_root, ds_name, "metadata.csv")
        if not os.path.isfile(truth_path):
            rows.append({"dataset": ds_name, "status": "missing_truth",
                         "auc_roc": np.nan, "average_precision": np.nan, "n": 0})
            continue
        t = _read_table(truth_path)

        # choose matching ID column in truth
        id_truth = id_col if id_col in t.columns else None
        if id_truth is None:
            for alt in [id_col, "repertoire_id","ID","id","rep_id","repertoire","sample_id"]:
                if alt in t.columns:
                    id_truth = alt
                    break
        if id_truth is None:
            rows.append({"dataset": ds_name, "status": f"no_id_in_truth(cols={list(t.columns)})",
                         "auc_roc": np.nan, "average_precision": np.nan, "n": 0})
            continue

        # detect label column in truth
        try:
            label_truth = _detect_col(t, label_col, ["label_positive","label","y","target"], "label (truth)")
        except KeyError as e:
            rows.append({"dataset": ds_name, "status": f"{e}",
                         "auc_roc": np.nan, "average_precision": np.nan, "n": 0})
            continue

        df = g[[id_col, prob_col]].merge(t[[id_truth, label_truth]], left_on=id_col, right_on=id_truth, how="inner")
        if df.empty:
            rows.append({"dataset": ds_name, "status": "no_overlap",
                         "auc_roc": np.nan, "average_precision": np.nan, "n": 0})
            continue

        y_true  = df[label_truth].astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"}).astype(int)
        y_score = df[prob_col].astype(float)
        if y_true.nunique() < 2:
            rows.append({"dataset": ds_name, "status": "one_class",
                         "auc_roc": np.nan, "average_precision": np.nan, "n": len(df)})
            continue
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)
        rows.append({"dataset": ds_name, "status": "ok",
                     "auc_roc": float(auc), "average_precision": float(ap), "n": int(len(df))})

    out = pd.DataFrame(rows).sort_values("dataset")
    out_csv = os.path.join(os.path.dirname(preds_agg_path), "auc_summary_from_aggregate.csv")
    out.to_csv(out_csv, index=False)
    print(f"[AUC-agg] Wrote {out_csv}")
    return out


def compute_jaccard_from_aggregate(seqs_agg_path: str,
                                   truth_root: str,
                                   k: int = 50000,
                                   cols = ("junction_aa","v_call","j_call")) -> pd.DataFrame:
    S = _read_table(seqs_agg_path)
    ds_col = _detect_dataset_column(S)
    rows = []
    for ds_name, g in S.groupby(ds_col):
        truth_path = os.path.join(truth_root, ds_name, "ranked_sequences.csv")
        if not os.path.isfile(truth_path):
            rows.append({"dataset": ds_name, "status": "missing_truth",
                         "jaccard": np.nan, "intersection": 0, "union": 0, "k": k})
            continue
        sort_by = None
        for cand in ["score","importance_score","odds","p"]:
            if cand in g.columns:
                sort_by = cand
                break
        if sort_by is not None:
            g = g.sort_values(sort_by, ascending=False)
        sub_top = g[[cols[0], cols[1], cols[2]]].head(k).copy()
        tru = _read_table(truth_path)
        for c in cols:
            if c not in tru.columns:
                rows.append({"dataset": ds_name, "status": f"missing_column_in_truth:{c}",
                             "jaccard": np.nan, "intersection": 0, "union": 0, "k": k})
                break
        else:
            s_keys = set(_rank_keys(sub_top, cols))
            t_keys = set(_rank_keys(tru, cols))
            inter  = len(s_keys & t_keys)
            union  = len(s_keys | t_keys)
            j      = inter / union if union else 0.0
            rows.append({"dataset": ds_name, "status": "ok",
                         "jaccard": float(j), "intersection": int(inter), "union": int(union),
                         "k": int(min(k, len(s_keys), len(t_keys)))})

    out = pd.DataFrame(rows).sort_values("dataset")
    out_csv = os.path.join(os.path.dirname(seqs_agg_path), "jaccard_summary_from_aggregate.csv")
    out.to_csv(out_csv, index=False)
    print(f"[Jaccard-agg] Wrote {out_csv}")
    return out


# ---------------------- single paired evaluation ----------------------

def _basic_pred_sanity(df: pd.DataFrame, id_col: str, prob_col: str) -> Dict[str, float]:
    s = df[prob_col].astype(float)
    return {
        "n_rows": int(len(df)),
        "prob_min": float(s.min()) if len(s) else float("nan"),
        "prob_max": float(s.max()) if len(s) else float("nan"),
        "prob_mean": float(s.mean()) if len(s) else float("nan"),
        "prob_std": float(s.std(ddof=0)) if len(s) else float("nan"),
        "n_missing_id": int(df[id_col].isna().sum()),
        "n_missing_prob": int(df[prob_col].isna().sum()),
    }


def pair_eval(test_ds: str,
              preds_file: str,
              tests_root: str,
              sample_sub: str = None,
              train_ranked: str = None,
              truth_root: str = None,
              k: int = 50000) -> Dict[str, object]:
    """
    Evaluate a single pair: test_dataset_* predictions and (optionally) Jaccard for the paired train_dataset_*.

    - Validates predictions file shape and ID coverage vs sample_submissions (if provided).
    - If tests_root/<test_ds>/metadata.csv has labels, computes AUC/AP.
    - If truth_root has train_dataset_* / ranked_sequences.csv and train_ranked is given, computes Jaccard.

    Returns a dict with all results.
    """
    report = {"test_dataset": test_ds}

    # ---- load predictions (CSV or TSV) and detect columns
    P = _read_table(preds_file)
    id_col   = _detect_col(P, None, ["repertoire_id", "ID", "id", "rep_id", "repertoire", "sample_id"], "ID (pred)")
    prob_col = _detect_col(P, None, ["label_positive_probability", "prob", "probability", "pred", "prediction", "score"], "probability (pred)")
    report["pred_cols"] = [id_col, prob_col]
    report["pred_summary"] = _basic_pred_sanity(P, id_col, prob_col)

    # ---- coverage vs sample_submissions (optional but useful)
    cov = {"checked": False}
    if sample_sub and os.path.isfile(sample_sub):
        SS = _read_table(sample_sub)
        ds_col = _detect_dataset_column(SS)
        id_col_ss = _detect_col(SS, None, ["ID", "repertoire_id", "id", "rep_id", "repertoire", "sample_id"], "ID (sample_submissions)")
        SS = SS[SS[ds_col].astype(str) == test_ds]
        want = set(SS[id_col_ss].astype(str))
        have = set(P[id_col].astype(str))
        cov = {
            "checked": True,
            "n_expected": int(len(want)),
            "n_pred_ids": int(len(have)),
            "n_missing": int(len(want - have)),
            "n_extra": int(len(have - want)),
        }
    report["coverage_vs_sample_submissions"] = cov

    # ---- AUC/AP if truth exists for the test dataset
    truth_meta = os.path.join(tests_root, test_ds, "metadata.csv")
    if os.path.isfile(truth_meta):
        try:
            auc_res = compute_auc(preds_file, truth_meta, id_col=id_col, prob_col=prob_col, label_col="label_positive")
            report["auc"] = {"status": "ok", **auc_res, "truth_path": truth_meta}
        except Exception as e:
            report["auc"] = {"status": f"error: {e}", "truth_path": truth_meta}
    else:
        report["auc"] = {"status": "unavailable", "reason": f"no labels at {truth_meta}"}

    # ---- Jaccard for paired train dataset, if provided
    jac = {"status": "skipped"}
    if train_ranked and truth_root:
        # map test_dataset_X -> train_dataset_X
        train_ds = "train_dataset_" + test_ds.split("_")[-1]
        truth_ranked = os.path.join(truth_root, train_ds, "ranked_sequences.csv")
        if os.path.isfile(train_ranked) and os.path.isfile(truth_ranked):
            try:
                jac_res = compute_jaccard(train_ranked, truth_ranked, k=k, cols=("junction_aa","v_call","j_call"))
                jac = {"status": "ok", "train_dataset": train_ds, **jac_res, "truth_path": truth_ranked}
            except Exception as e:
                jac = {"status": f"error: {e}", "train_dataset": train_ds, "truth_path": truth_ranked}
        else:
            jac = {"status": "unavailable",
                   "reason": f"missing file(s). train_ranked={os.path.isfile(train_ranked)} truth_ranked={os.path.isfile(truth_ranked)}",
                   "train_dataset": train_ds,
                   "truth_path": truth_ranked}
    report["jaccard"] = jac

    return report


# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description="Local metrics (single, many, aggregate, and single paired eval).")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Single-file AUC
    p_auc = subparsers.add_parser("auc", help="Compute ROC AUC (and AP) from prediction and truth CSVs.")
    p_auc.add_argument("--pred", required=True)
    p_auc.add_argument("--truth", required=True)
    p_auc.add_argument("--id_col", default="repertoire_id")
    p_auc.add_argument("--prob_col", default="label_positive_probability")
    p_auc.add_argument("--label_col", default="label_positive")

    # Single-file Jaccard
    p_j = subparsers.add_parser("jaccard", help="Compute Jaccard between two ranked sequence lists.")
    p_j.add_argument("--sub", required=True)
    p_j.add_argument("--truth", required=True)
    p_j.add_argument("--k", type=int, default=50000)
    p_j.add_argument("--cols", nargs=3, default=["junction_aa","v_call","j_call"])

    # Many-dataset (tree)
    p_am = subparsers.add_parser("auc-many", help="Compute AUC for every test_dataset_* under preds_root.")
    p_am.add_argument("--preds_root", required=True)
    p_am.add_argument("--tests_root", required=True)
    p_am.add_argument("--id_col", default="repertoire_id")
    p_am.add_argument("--prob_col", default="label_positive_probability")
    p_am.add_argument("--label_col", default="label_positive")

    p_jm = subparsers.add_parser("jaccard-many", help="Compute Jaccard for every train_dataset_* under subs_root.")
    p_jm.add_argument("--subs_root", required=True)
    p_jm.add_argument("--truth_root", required=True)
    p_jm.add_argument("--k", type=int, default=50000)
    p_jm.add_argument("--cols", nargs=3, default=["junction_aa","v_call","j_call"])

    # Aggregate
    p_aa = subparsers.add_parser("auc-agg", help="Compute AUC per dataset from a single aggregate predictions TSV.")
    p_aa.add_argument("--preds_agg", required=True)
    p_aa.add_argument("--tests_root", required=True)
    p_aa.add_argument("--id_col", default="repertoire_id")
    p_aa.add_argument("--prob_col", default="label_positive_probability")
    p_aa.add_argument("--label_col", default="label_positive")

    p_ja = subparsers.add_parser("jaccard-agg", help="Compute Jaccard per dataset from a single aggregate sequences TSV.")
    p_ja.add_argument("--seqs_agg", required=True)
    p_ja.add_argument("--truth_root", required=True)
    p_ja.add_argument("--k", type=int, default=50000)
    p_ja.add_argument("--cols", nargs=3, default=["junction_aa","v_call","j_call"])

    # Pair evaluation (single test↔train pair)
    p_pair = subparsers.add_parser("pair-eval", help="Evaluate one test dataset (AUC if labels exist) and paired train Jaccard.")
    p_pair.add_argument("--test_ds", required=True, help="e.g., test_dataset_1")
    p_pair.add_argument("--preds_file", required=True, help="Path to outputs/.../predictions.csv (or TSV aggregate).")
    p_pair.add_argument("--tests_root", required=True, help="Root containing test_dataset_*/ (to find truth metadata.csv if available).")
    p_pair.add_argument("--sample_sub", default=None, help="Path to data/sample_submissions.csv (optional, for ID coverage).")
    p_pair.add_argument("--train_ranked", default=None, help="Path to outputs/train_dataset_*/ranked_sequences.csv")
    p_pair.add_argument("--truth_root", default=None, help="Root with truth ranked lists: train_dataset_*/ranked_sequences.csv")
    p_pair.add_argument("--k", type=int, default=50000, help="Top-k for Jaccard.")

    args = parser.parse_args()

    if args.cmd == "auc":
        print(compute_auc(args.pred, args.truth, args.id_col, args.prob_col, args.label_col))
    elif args.cmd == "jaccard":
        print(compute_jaccard(args.sub, args.truth, k=args.k, cols=tuple(args.cols)))
    elif args.cmd == "auc-many":
        print(compute_auc_many(args.preds_root, args.tests_root,
                               id_col=args.id_col, prob_col=args.prob_col, label_col=args.label_col).to_string(index=False))
    elif args.cmd == "jaccard-many":
        print(compute_jaccard_many(args.subs_root, args.truth_root,
                                   k=args.k, cols=tuple(args.cols)).to_string(index=False))
    elif args.cmd == "auc-agg":
        print(compute_auc_from_aggregate(args.preds_agg, args.tests_root,
                                         id_col=args.id_col, prob_col=args.prob_col, label_col=args.label_col).to_string(index=False))
    elif args.cmd == "jaccard-agg":
        print(compute_jaccard_from_aggregate(args.seqs_agg, args.truth_root,
                                             k=args.k, cols=tuple(args.cols)).to_string(index=False))
    elif args.cmd == "pair-eval":
        rep = pair_eval(
            test_ds=args.test_ds,
            preds_file=args.preds_file,
            tests_root=args.tests_root,
            sample_sub=args.sample_sub,
            train_ranked=args.train_ranked,
            truth_root=args.truth_root,
            k=args.k,
        )
        import json
        print(json.dumps(rep, indent=2))
        out_json = os.path.join(os.path.dirname(args.preds_file), f"{args.test_ds}_evaluation.json")
        with open(out_json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"[pair-eval] Wrote {out_json}")
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

