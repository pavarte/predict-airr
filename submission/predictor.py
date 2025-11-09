# submission/predictor.py
import os
import glob
import json
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

# Headless plotting backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    brier_score_loss,
)
from scipy.stats import fisher_exact


# --------------------------- utility + logging helpers ---------------------------

def _log(msg: str) -> None:
    print(f"[predictor] {msg}", flush=True)


def _to_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer)): return x != 0
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _safe_makedirs(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        _log(f"WARNING: could not create directory {path}: {e}")


def _savefig(path: str, tight: bool = True) -> None:
    try:
        if tight:
            plt.tight_layout()
        plt.savefig(path, dpi=150)
        _log(f"Saved figure: {path}")
    except Exception as e:
        _log(f"WARNING: failed to save figure {path}: {e}")
    finally:
        plt.close()


# --------------------------- I/O: repertoires & meta ----------------------------

def _read_repertoire_tsv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed reading TSV: {path} ({e})")

    # duplicate_count normalization
    if "templates" in df.columns and "duplicate_count" not in df.columns:
        df["duplicate_count"] = pd.to_numeric(df["templates"], errors="coerce").fillna(1).astype(int)
    elif "duplicate_count" in df.columns:
        df["duplicate_count"] = pd.to_numeric(df["duplicate_count"], errors="coerce").fillna(1).astype(int)
    else:
        df["duplicate_count"] = 1

    # Ensure required text columns exist
    for col in ("junction_aa", "v_call", "j_call"):
        if col not in df.columns:
            _log(f"WARNING: column '{col}' missing in {path}; filling with empty strings.")
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Drop all-empty rows (sanity)
    before = len(df)
    df = df[(df["junction_aa"].astype(str).str.len() > 0) | (df["v_call"] != "") | (df["j_call"] != "")]
    dropped = before - len(df)
    if dropped > 0:
        _log(f"INFO: dropped {dropped} empty rows in {os.path.basename(path)}")

    return df[["junction_aa", "v_call", "j_call", "duplicate_count"]]


def _build_test_meta(test_dir: str) -> pd.DataFrame:
    """
    Build ['repertoire_id','filename'] for test sets that don't ship metadata.csv.

    Logic:
      1) If test_dir/metadata.csv exists and has needed cols, use it.
      2) Else read top-level sample_submissions.csv, filter to this dataset, use an ID column (e.g., 'ID').
      3) Else infer from *.tsv* files present in test_dir.
    """
    def _infer_from_files() -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(test_dir, "*.tsv*")))
        if not files:
            raise FileNotFoundError(f"No metadata.csv and no *.tsv* files found in {test_dir}.")
        def rid_from(path: str) -> str:
            name = os.path.basename(path)
            for suf in (".tsv.gz", ".tsv.bz2", ".tsv.xz", ".tsv"):
                if name.endswith(suf):
                    return name[: -len(suf)]
            return os.path.splitext(name)[0]
        return pd.DataFrame({
            "repertoire_id": [rid_from(p) for p in files],
            "filename": [os.path.basename(p) for p in files],
        })

    # 1) Use test_dir/metadata.csv if present
    meta_path = os.path.join(test_dir, "metadata.csv")
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        df.columns = [c.strip() for c in df.columns]
        if {"repertoire_id","filename"}.issubset(df.columns):
            return df[["repertoire_id","filename"]]
        if "file" in df.columns and "repertoire_id" in df.columns:
            return df.rename(columns={"file":"filename"})[["repertoire_id","filename"]]
        _log(f"WARNING: {meta_path} lacks required columns; columns={list(df.columns)}. Falling back.")

    # 2) Use top-level sample_submissions.csv, filter to this dataset, detect ID col
    cur = os.path.abspath(test_dir); sample_path = None
    while True:
        cand = os.path.join(cur, "sample_submissions.csv")
        if os.path.exists(cand):
            sample_path = cand; break
        parent = os.path.dirname(cur)
        if parent == cur: break
        cur = parent

    if sample_path:
        df_sub = pd.read_csv(sample_path)
        df_sub.columns = [c.strip() for c in df_sub.columns]
        _log(f"Found sample_submissions.csv with columns: {list(df_sub.columns)}")

        dataset_name = os.path.basename(os.path.abspath(test_dir))
        if "dataset" in df_sub.columns:
            before = len(df_sub)
            df_sub = df_sub[df_sub["dataset"] == dataset_name].copy()
            _log(f"Filtered to dataset={dataset_name}: {before} → {len(df_sub)} rows")

        # Find repertoire ID column (your file uses 'ID')
        cand_cols = ["repertoire_id","repertoire","sample_id","sample","subject_id","subject","id","ID"]
        rep_col = next((c for c in cand_cols if c in df_sub.columns), None)
        if rep_col is None:
            _log("WARNING: No repertoire ID column in sample_submissions; falling back to file inference.")
            return _infer_from_files()

        # Keep unique IDs and rename to repertoire_id
        df_ids = df_sub[[rep_col]].dropna().drop_duplicates().rename(columns={rep_col: "repertoire_id"})
        df_ids["repertoire_id"] = df_ids["repertoire_id"].astype(str)

        # Map repertoire_id -> filename inside test_dir
        def find_file_for(rid: str) -> Optional[str]:
            for ext in (".tsv", ".tsv.gz", ".tsv.bz2", ".tsv.xz"):
                p = os.path.join(test_dir, f"{rid}{ext}")
                if os.path.exists(p):
                    return os.path.basename(p)
            # relaxed fallback: anything containing the ID
            matches = glob.glob(os.path.join(test_dir, f"*{rid}*.tsv*"))
            return os.path.basename(matches[0]) if matches else None

        df_ids["filename"] = df_ids["repertoire_id"].map(find_file_for)
        missing = int(df_ids["filename"].isna().sum())
        if missing:
            _log(f"WARNING: {missing} IDs from sample_submissions had no matching files in {test_dir}.")
        df_ids = df_ids.dropna(subset=["filename"])

        if len(df_ids):
            _log(f"Mapped {len(df_ids)} IDs to files in {test_dir}.")
            return df_ids[["repertoire_id","filename"]]

        _log("WARNING: After mapping IDs to files, no matches remained; falling back to file inference.")

    # 3) Infer from files
    return _infer_from_files()


# --------------------------------- featurization ---------------------------------

def _kmer_counts(aa: str, k: int = 3) -> Counter:
    out = Counter()
    if not isinstance(aa, str): return out
    n = len(aa)
    for i in range(0, max(0, n - k + 1)):
        out[aa[i:i + k]] += 1
    return out


def _featurize_repertoire(df_rep: pd.DataFrame,
                          k: int,
                          top_kmers: Optional[List[str]],
                          v_vocab: Optional[List[str]],
                          j_vocab: Optional[List[str]]) -> dict:
    if df_rep is None or len(df_rep) == 0:
        return {"cdr3_len_mean": 0.0, "cdr3_len_std": 0.0, "shannon": 0.0}

    w = df_rep["duplicate_count"].astype(int)
    total_w = max(int(w.sum()), 1)

    # k-mers
    km = Counter()
    for s, wt in zip(df_rep["junction_aa"].astype(str), w):
        if not isinstance(s, str) or not s:
            continue
        c = _kmer_counts(s, k)
        if not c:
            continue
        for kk, vv in c.items():
            km[kk] += vv * wt
    if top_kmers is not None:
        km = {f"km{k}_{kk}": km.get(kk, 0) / total_w for kk in top_kmers}
    else:
        km = {f"km{k}_{kk}": vv / total_w for kk, vv in km.items()}

    # V/J usage
    v_counts = df_rep["v_call"].value_counts()
    j_counts = df_rep["j_call"].value_counts()
    vfeats = {f"v_{g}": v_counts.get(g, 0) / len(df_rep) for g in (v_vocab or v_counts.index)}
    jfeats = {f"j_{g}": j_counts.get(g, 0) / len(df_rep) for g in (j_vocab or j_counts.index)}

    # length + diversity proxy
    lens = df_rep["junction_aa"].astype(str).str.len()
    p = w / total_w
    shannon = float(-(p * np.log(p + 1e-12)).sum())

    feats = {
        "cdr3_len_mean": float(lens.mean() if len(lens) else 0.0),
        "cdr3_len_std": float(lens.std(ddof=0) if len(lens) else 0.0),
        "shannon": shannon
    }
    feats.update(vfeats); feats.update(jfeats); feats.update(km)
    return feats


# -------------------------------- diagnostics ------------------------------------

def _plot_prob_diagnostics(y_true: np.ndarray, prob: np.ndarray, out_dir: str, split_tag: str) -> Dict[str, Any]:
    """Save ROC, PR, calibration, histogram; return metrics."""
    diag = {}
    diag_dir = os.path.join(out_dir, "diagnostics")
    _safe_makedirs(diag_dir)

    # ROC
    try:
        auc = roc_auc_score(y_true, prob)
        diag["roc_auc"] = float(auc)
        RocCurveDisplay.from_predictions(y_true, prob)
        plt.title(f"ROC ({split_tag}) AUC={auc:.3f}")
        _savefig(os.path.join(diag_dir, f"roc_{split_tag}.png"))
    except Exception as e:
        _log(f"WARNING: ROC plotting failed ({split_tag}): {e}")

    # PR
    try:
        ap = average_precision_score(y_true, prob)
        diag["average_precision"] = float(ap)
        PrecisionRecallDisplay.from_predictions(y_true, prob)
        plt.title(f"Precision-Recall ({split_tag}) AP={ap:.3f}")
        _savefig(os.path.join(diag_dir, f"pr_{split_tag}.png"))
    except Exception as e:
        _log(f"WARNING: PR plotting failed ({split_tag}): {e}")

    # Calibration curve
    try:
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        plt.figure()
        plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title(f"Calibration ({split_tag})")
        _savefig(os.path.join(diag_dir, f"calibration_{split_tag}.png"))
        diag["brier"] = float(brier_score_loss(y_true, prob))
    except Exception as e:
        _log(f"WARNING: Calibration plotting failed ({split_tag}): {e}")

    # Histogram of probabilities
    try:
        plt.figure()
        plt.hist(prob, bins=30)
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.title(f"Score distribution ({split_tag})")
        _savefig(os.path.join(diag_dir, f"prob_hist_{split_tag}.png"))
    except Exception as e:
        _log(f"WARNING: Probability histogram failed ({split_tag}): {e}")

    return diag


def _plot_feature_magnitudes(model: CalibratedClassifierCV, feature_cols: List[str], out_dir: str) -> None:
    """
    Plot top positive/negative coefficients from the underlying LogisticRegression inside the pipeline.
    CalibratedClassifierCV wraps clones; we fetch one of the base estimators.
    """
    try:
        # Pull one fitted base estimator (they are clones per CV split)
        pipe = None
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            pipe = model.calibrated_classifiers_[0].estimator  # Pipeline
        elif hasattr(model, "base_estimator"):
            pipe = model.base_estimator

        if pipe is None:
            _log("INFO: Could not locate inner estimator for coefficient plotting.")
            return

        # Find the LogisticRegression step
        lr = None
        if isinstance(pipe, Pipeline):
            for _, step in pipe.steps:
                if isinstance(step, LogisticRegression):
                    lr = step
                    break
        if lr is None or not hasattr(lr, "coef_"):
            _log("INFO: LogisticRegression with coef_ not found; skipping coef plot.")
            return

        coefs = lr.coef_.ravel()
        if len(coefs) != len(feature_cols):
            _log("INFO: Coefficients length does not match feature columns; skipping coef plot.")
            return

        k = min(20, len(coefs))
        idx_pos = np.argsort(coefs)[-k:][::-1]
        idx_neg = np.argsort(coefs)[:k]

        diag_dir = os.path.join(out_dir, "diagnostics"); _safe_makedirs(diag_dir)

        # Positive
        plt.figure(figsize=(8, max(4, k * 0.25)))
        plt.barh([feature_cols[i] for i in idx_pos[::-1]], coefs[idx_pos[::-1]])
        plt.title("Top positive coefficients")
        plt.xlabel("Coefficient")
        _savefig(os.path.join(diag_dir, "coef_top_positive.png"))

        # Negative
        plt.figure(figsize=(8, max(4, k * 0.25)))
        plt.barh([feature_cols[i] for i in idx_neg], coefs[idx_neg])
        plt.title("Top negative coefficients")
        plt.xlabel("Coefficient")
        _savefig(os.path.join(diag_dir, "coef_top_negative.png"))

    except Exception as e:
        _log(f"WARNING: feature magnitude plotting failed: {e}")


def _plot_repertoire_summaries(train_meta: pd.DataFrame, train_dir: str, out_dir: str) -> None:
    """
    Quick sanity plots: per-repertoire total weight (depth), unique clonotypes,
    and shannon proxy from featurization (computed cheaply here).
    """
    try:
        rows = []
        for _, row in train_meta.iterrows():
            path = os.path.join(train_dir, row["filename"])
            try:
                df = _read_repertoire_tsv(path)
            except Exception as e:
                _log(f"WARNING: skipping summary for {path}: {e}")
                continue
            w = df["duplicate_count"].astype(int)
            total_w = int(w.sum())
            uniq = df.drop_duplicates(subset=["junction_aa", "v_call", "j_call"]).shape[0]
            p = (w / max(total_w, 1)).values
            shannon = float(-(p * np.log(p + 1e-12)).sum())
            rows.append({"repertoire_id": row["repertoire_id"], "total_w": total_w, "unique_triplets": uniq, "shannon": shannon})

        if not rows:
            return

        rep = pd.DataFrame(rows)
        diag_dir = os.path.join(out_dir, "diagnostics"); _safe_makedirs(diag_dir)

        plt.figure()
        plt.hist(rep["total_w"].values, bins=30)
        plt.xlabel("Total duplicate_count (depth proxy)")
        plt.ylabel("Repertoires")
        plt.title("Per-repertoire depth")
        _savefig(os.path.join(diag_dir, "depth_hist.png"))

        plt.figure()
        plt.hist(rep["unique_triplets"].values, bins=30)
        plt.xlabel("# unique (CDR3,V,J) triplets")
        plt.ylabel("Repertoires")
        plt.title("Unique clonotypes per repertoire")
        _savefig(os.path.join(diag_dir, "unique_triplets_hist.png"))

        plt.figure()
        plt.hist(rep["shannon"].values, bins=30)
        plt.xlabel("Shannon proxy")
        plt.ylabel("Repertoires")
        plt.title("Diversity (weight-based)")
        _savefig(os.path.join(diag_dir, "shannon_hist.png"))

        rep.to_csv(os.path.join(diag_dir, "repertoire_summaries.csv"), index=False)
        _log(f"Wrote repertoire summaries: {os.path.join(diag_dir, 'repertoire_summaries.csv')}")
    except Exception as e:
        _log(f"WARNING: repertoire summaries failed: {e}")


# ----------------------------------- model class ---------------------------------

class ImmuneStatePredictor:
    """
    Template-compatible predictor:
      - train(train_dir, out_dir)
      - predict(test_dir, out_dir) -> predictions.csv
      - rank_sequences(train_dir, out_dir, topk)
    """

    def __init__(self, n_jobs: int = 1, device: str = "cpu", random_state: int = 42):
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state

        self.k = 3
        self.top_kmers: List[str] = []
        self.v_vocab: List[str] = []
        self.j_vocab: List[str] = []
        self.feature_cols: List[str] = []
        self.model: Optional[CalibratedClassifierCV] = None

        # Training diagnostics
        self._train_metrics: Dict[str, Any] = {}

    # -------- internals --------
    def _collect_feature_vocab_from_meta(self, train_dir: str, meta_subset: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """Build vocabularies using ONLY the provided meta subset (to avoid leakage)."""
        _log(f"Building vocab from {len(meta_subset)} repertoires (train split).")
        kmer_counter = Counter(); v_set = Counter(); j_set = Counter()
        failed = 0
        for _, row in meta_subset.iterrows():
            fpath = os.path.join(train_dir, row["filename"])
            try:
                df_r = _read_repertoire_tsv(fpath)
            except Exception as e:
                failed += 1
                _log(f"WARNING: skipping {fpath}: {e}")
                continue
            for s, wt in zip(df_r["junction_aa"].astype(str), df_r["duplicate_count"].astype(int)):
                if not s:
                    continue
                kmer_counter.update({k: c * wt for k, c in _kmer_counts(s, self.k).items()})
            v_set.update(df_r["v_call"].astype(str).tolist())
            j_set.update(df_r["j_call"].astype(str).tolist())
        if failed:
            _log(f"WARNING: {failed} training files failed to parse for vocab.")
        top_kmers = [k for k, _ in kmer_counter.most_common(5000)]
        _log(f"Built vocabularies: {len(top_kmers)} top k-mers, {len(v_set)} V genes, {len(j_set)} J genes.")
        return top_kmers, sorted(v_set.keys()), sorted(j_set.keys())

    def _build_matrix(self, dataset_dir: str, meta: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str]]:
        rows = []; y = []
        failed = 0
        for _, row in meta.iterrows():
            fpath = os.path.join(dataset_dir, row["filename"])
            try:
                df_r = _read_repertoire_tsv(fpath)
            except Exception as e:
                failed += 1
                _log(f"WARNING: skipping {fpath}: {e}")
                continue
            feats = _featurize_repertoire(df_r, k=self.k,
                                          top_kmers=self.top_kmers,
                                          v_vocab=self.v_vocab,
                                          j_vocab=self.j_vocab)
            feats["repertoire_id"] = row["repertoire_id"]
            rows.append(feats)
            if "label_positive" in meta.columns:
                y.append(_to_bool(row["label_positive"]))
        if failed:
            _log(f"WARNING: {failed} files were skipped during matrix construction.")
        if not rows:
            raise RuntimeError(f"No features could be built from dir {dataset_dir}.")
        X = pd.DataFrame(rows).fillna(0.0)
        if not self.feature_cols:
            self.feature_cols = [c for c in X.columns if c != "repertoire_id"]
        X = X[["repertoire_id"] + self.feature_cols]
        return X, (np.array(y) if y else None), self.feature_cols

    def _stratified_70_15_15(self, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train70, val15, test15) stratified by label_positive."""
        if "label_positive" not in meta.columns:
            raise RuntimeError("metadata.csv lacks 'label_positive' for splitting.")
        y = meta["label_positive"].map(_to_bool).astype(int).values
        # First: test 15%
        meta_trainval, meta_test = train_test_split(
            meta, test_size=0.15, stratify=y, random_state=self.random_state)
        # Now: val is 15% of total => 0.15 / 0.85 of remaining
        y_trainval = meta_trainval["label_positive"].map(_to_bool).astype(int).values
        val_frac_of_remaining = 0.15 / 0.85  # ≈ 0.17647
        meta_train, meta_val = train_test_split(
            meta_trainval, test_size=val_frac_of_remaining, stratify=y_trainval, random_state=self.random_state)
        # Log counts
        def _count_pos(df: pd.DataFrame) -> Tuple[int, int]:
            n = len(df); p = int(df["label_positive"].map(_to_bool).sum()); return n, p
        ntr, ptr = _count_pos(meta_train); nva, pva = _count_pos(meta_val); nts, pts = _count_pos(meta_test)
        _log(f"Split (N,pos): train={ntr},{ptr}  val={nva},{pva}  test={nts},{pts}")
        return meta_train.reset_index(drop=True), meta_val.reset_index(drop=True), meta_test.reset_index(drop=True)

    # -------- API required by main.py --------
    def train(self, train_dir: str, out_dir: str) -> None:
        _safe_makedirs(out_dir)

        # Load full metadata
        full_meta = pd.read_csv(os.path.join(train_dir, "metadata.csv"))
        _log(f"Found {len(full_meta)} training repertoires in {train_dir}.")

        # Quick sanity plots (full set)
        _plot_repertoire_summaries(full_meta, train_dir, out_dir)

        # 70/15/15 split (stratified)
        meta_tr, meta_va, meta_te = self._stratified_70_15_15(full_meta)

        # Save split assignments
        split_df = pd.concat([
            meta_tr[["repertoire_id"]].assign(split="train"),
            meta_va[["repertoire_id"]].assign(split="val"),
            meta_te[["repertoire_id"]].assign(split="test"),
        ], ignore_index=True)
        split_path = os.path.join(out_dir, "split_assignments.csv")
        split_df.to_csv(split_path, index=False)
        _log(f"Wrote split assignments: {split_path}")

        # Build vocab ONLY from train split (avoid leakage)
        self.top_kmers, self.v_vocab, self.j_vocab = self._collect_feature_vocab_from_meta(train_dir, meta_tr)

        # Build matrices
        X_tr, y_tr, _ = self._build_matrix(train_dir, meta_tr)
        X_va, y_va, _ = self._build_matrix(train_dir, meta_va)
        X_te, y_te, _ = self._build_matrix(train_dir, meta_te)

        for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
            if y is None or len(y) == 0:
                raise RuntimeError(f"No labels for {name} split.")
        _log(f"Matrices: train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")

        # Model (fit on train70 only)
        base = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga",
                l1_ratio=0.5, C=1.0, max_iter=5000,
                n_jobs=self.n_jobs, random_state=self.random_state))
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        self.model.fit(X_tr[self.feature_cols].values, y_tr)
        _log("Model training (70%) + isotonic calibration complete.")

        # Evaluate splits
        metrics_all: Dict[str, Any] = {}

        def _eval_and_save(tag: str, X_df: pd.DataFrame, y_vec: np.ndarray):
            prob = self.model.predict_proba(X_df[self.feature_cols].values)[:, 1]
            # Save predictions with truth
            out_csv = os.path.join(out_dir, f"{tag}_predictions.csv")
            pd.DataFrame({
                "repertoire_id": X_df["repertoire_id"],
                "label_positive_probability": prob,
                "label_positive": y_vec
            }).to_csv(out_csv, index=False)
            _log(f"Wrote {tag} predictions: {out_csv}")
            # Plots + metrics
            m = _plot_prob_diagnostics(y_vec, prob, out_dir, split_tag=tag)
            metrics_all[tag] = m

        _eval_and_save("train", X_tr, y_tr)
        _eval_and_save("val",   X_va, y_va)
        _eval_and_save("test",  X_te, y_te)

        # Coefficient plots (best-effort)
        _plot_feature_magnitudes(self.model, self.feature_cols, out_dir)

        # Persist feature spec
        try:
            with open(os.path.join(out_dir, "feature_spec.json"), "w") as f:
                json.dump({
                    "k": self.k,
                    "top_kmers": self.top_kmers,
                    "v_vocab": self.v_vocab,
                    "j_vocab": self.j_vocab,
                    "feature_cols": self.feature_cols
                }, f)
            _log(f"Wrote feature spec: {os.path.join(out_dir, 'feature_spec.json')}")
        except Exception as e:
            _log(f"WARNING: failed to write feature_spec.json: {e}")

        # Save split metrics json
        try:
            with open(os.path.join(out_dir, "split_metrics.json"), "w") as f:
                json.dump(metrics_all, f, indent=2)
            _log(f"Wrote split metrics: {os.path.join(out_dir, 'split_metrics.json')}")
        except Exception as e:
            _log(f"WARNING: failed to write split_metrics.json: {e}")

    def predict(self, test_dir: str, out_dir: str) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() before predict().")

        _safe_makedirs(out_dir)
        try:
            meta = _build_test_meta(test_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))

        # Safety net: ensure required columns exist
        need = {"repertoire_id", "filename"}
        if not need.issubset(meta.columns):
            raise RuntimeError(f"Internal error: meta missing {need}. Got columns={list(meta.columns)}")

        _log(f"Predicting {len(meta)} repertoires from {test_dir}.")
        X, _, _ = self._build_matrix(test_dir, meta)
        probs = self.model.predict_proba(X[self.feature_cols].values)[:, 1]
        out = pd.DataFrame({"repertoire_id": X["repertoire_id"], "label_positive_probability": probs})
        out_path = os.path.join(out_dir, "predictions.csv")
        out.to_csv(out_path, index=False)
        _log(f"Wrote predictions: {out_path}")
        return out

    def rank_sequences(self, train_dir: str, out_dir: str, topk: int = 50000) -> pd.DataFrame:
        _safe_makedirs(out_dir)
        meta = pd.read_csv(os.path.join(train_dir, "metadata.csv"))
        rows = []
        failed = 0
        for _, row in meta.iterrows():
            rep_id = row["repertoire_id"]
            lab = _to_bool(row["label_positive"])
            fpath = os.path.join(train_dir, row["filename"])
            try:
                df_r = _read_repertoire_tsv(fpath)
            except Exception as e:
                failed += 1
                _log(f"WARNING: skipping {fpath}: {e}")
                continue
            df_r = df_r.drop_duplicates(subset=["junction_aa", "v_call", "j_call"])
            df_r["repertoire_id"] = rep_id
            df_r["label_positive"] = lab
            rows.append(df_r)
        if failed:
            _log(f"WARNING: {failed} files failed during rank_sequences data load.")
        if not rows:
            raise RuntimeError("No data available for rank_sequences.")

        pres = pd.concat(rows, ignore_index=True)

        pos_reps = pres.loc[pres["label_positive"] == True, "repertoire_id"].nunique()
        neg_reps = pres.loc[pres["label_positive"] == False, "repertoire_id"].nunique()

        grp = pres.groupby(["junction_aa", "v_call", "j_call", "label_positive"])["repertoire_id"].nunique()
        keys = pres[["junction_aa", "v_call", "j_call"]].drop_duplicates()

        out_rows = []
        for _, krow in keys.iterrows():
            jaa, v, j = krow["junction_aa"], krow["v_call"], krow["j_call"]
            a = int(grp.get((jaa, v, j, True), 0))
            c = int(grp.get((jaa, v, j, False), 0))
            b = pos_reps - a
            d = neg_reps - c
            if (a + b == 0) or (c + d == 0):
                continue
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            score = float(np.log(odds + 1e-9) * -np.log10(p + 1e-300))
            out_rows.append((jaa, v, j, score, odds, p, a, c))

        if not out_rows:
            _log("WARNING: No sequences produced a valid Fisher table.")
            out = pd.DataFrame(columns=["junction_aa", "v_call", "j_call", "score", "odds", "p", "pos_pres", "neg_pres"])
        else:
            out = pd.DataFrame(out_rows, columns=["junction_aa", "v_call", "j_call", "score", "odds", "p", "pos_pres", "neg_pres"])
            out = out.sort_values(["score", "pos_pres", "neg_pres"], ascending=[False, False, True]).head(topk)

        out_file = os.path.join(out_dir, "ranked_sequences.csv")
        out.to_csv(out_file, index=False)
        _log(f"Wrote ranked sequences: {out_file}")
        return out

