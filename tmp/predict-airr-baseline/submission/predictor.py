import os
import json
import math
import glob
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from scipy.stats import fisher_exact


def _to_bool(x) -> bool:
    """Robustly parse label_positive that might be True/False, 1/0, 'true'/'false', etc."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return x != 0
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


############################
# File helpers
############################

def _read_repertoire_tsv(path: str) -> pd.DataFrame:
    """Read a single repertoire TSV in AIRR-like format.
    Required columns: junction_aa, v_call, j_call
    Optional columns: d_call, templates or duplicate_count
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "templates" in df.columns and "duplicate_count" not in df.columns:
        df["duplicate_count"] = pd.to_numeric(df["templates"], errors="coerce").fillna(1).astype(int)
    elif "duplicate_count" in df.columns:
        df["duplicate_count"] = pd.to_numeric(df["duplicate_count"], errors="coerce").fillna(1).astype(int)
    else:
        df["duplicate_count"] = 1

    for col in ("junction_aa", "v_call", "j_call"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    return df[["junction_aa", "v_call", "j_call", "duplicate_count"]]


def _kmer_counts(aa: str, k: int = 3) -> Counter:
    out = Counter()
    if not isinstance(aa, str):
        return out
    n = len(aa)
    if n < k:
        return out
    for i in range(0, n - k + 1):
        out[aa[i:i+k]] += 1
    return out


def _featurize_repertoire(df_rep: pd.DataFrame,
                          k: int = 3,
                          top_kmers: Optional[List[str]] = None,
                          v_vocab: Optional[List[str]] = None,
                          j_vocab: Optional[List[str]] = None) -> dict:
    w = df_rep["duplicate_count"].astype(int)
    total_w = max(int(w.sum()), 1)

    # k-mers
    km = Counter()
    for s, wt in zip(df_rep["junction_aa"].astype(str), w):
        c = _kmer_counts(s, k)
        if c:
            for kk, vv in c.items():
                km[kk] += vv * wt
    if top_kmers is not None:
        km = {f"km{str(k)}_{kk}": km.get(kk, 0)/total_w for kk in top_kmers}
    else:
        km = {f"km{str(k)}_{kk}": vv/total_w for kk, vv in km.items()}

    # V/J usage one-hot
    v_counts = df_rep["v_call"].value_counts()
    j_counts = df_rep["j_call"].value_counts()
    if v_vocab is None:
        vfeats = {f"v_{k}": v_counts.get(k, 0)/len(df_rep) for k in v_counts.index}
    else:
        vfeats = {f"v_{k}": v_counts.get(k, 0)/len(df_rep) for k in v_vocab}
    if j_vocab is None:
        jfeats = {f"j_{k}": j_counts.get(k, 0)/len(df_rep) for k in j_counts.index}
    else:
        jfeats = {f"j_{k}": j_counts.get(k, 0)/len(df_rep) for k in j_vocab}

    # Length + diversity proxy
    lens = df_rep["junction_aa"].astype(str).str.len()
    p = w / total_w
    shannon = float((- (p * np.log(p + 1e-12))).sum())

    feats = {
        "cdr3_len_mean": float(lens.mean() if len(lens) else 0.0),
        "cdr3_len_std": float(lens.std(ddof=0) if len(lens) else 0.0),
        "shannon": shannon,
    }
    feats.update(vfeats)
    feats.update(jfeats)
    feats.update(km)
    return feats


############################
# Template-required class
############################

class ImmuneStatePredictor:
    """
    PredictAIRR-compatible baseline:
      - train(): build feature vocab (top k-mers, V/J vocab), train elastic-net LR with isotonic calibration
      - predict(): calibrated probabilities for test repertoires
      - rank_sequences(): Fisher exact test across the training dataset; output up to top-50k unique sequences
    """

    def __init__(self, n_jobs: int = 1, device: str = "cpu", random_state: int = 42):
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state

        self.k = 3
        self.top_kmers: List[str] = []
        self.v_vocab: List[str] = []
        self.j_vocab: List[str] = []

        self.model = None
        self.feature_cols: List[str] = []

    # ---- helpers ----
    def _collect_feature_vocab(self, train_dir: str) -> Tuple[List[str], List[str], List[str]]:
        meta = pd.read_csv(os.path.join(train_dir, "metadata.csv"))
        kmer_counter = Counter()
        v_set = Counter()
        j_set = Counter()

        for _, row in meta.iterrows():
            df_r = _read_repertoire_tsv(os.path.join(train_dir, row["filename"]))
            for s, wt in zip(df_r["junction_aa"].astype(str), df_r["duplicate_count"].astype(int)):
                kmer_counter.update({k: c*wt for k, c in _kmer_counts(s, self.k).items()})
            v_set.update(df_r["v_call"].astype(str).tolist())
            j_set.update(df_r["j_call"].astype(str).tolist())
        top_kmers = [k for k, _ in kmer_counter.most_common(5000)]
        v_vocab = sorted(v_set.keys())
        j_vocab = sorted(j_set.keys())
        return top_kmers, v_vocab, j_vocab

    def _build_matrix(self, dataset_dir: str, meta: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str]]:
        rows = []
        y = []
        for _, row in meta.iterrows():
            df_r = _read_repertoire_tsv(os.path.join(dataset_dir, row["filename"]))
            feats = _featurize_repertoire(df_r, k=self.k,
                                          top_kmers=self.top_kmers,
                                          v_vocab=self.v_vocab,
                                          j_vocab=self.j_vocab)
            feats["repertoire_id"] = row["repertoire_id"]
            rows.append(feats)
            if "label_positive" in meta.columns:
                y.append(_to_bool(row["label_positive"]))
        X = pd.DataFrame(rows).fillna(0.0)
        if not self.feature_cols:
            self.feature_cols = [c for c in X.columns if c != "repertoire_id"]
        X = X[["repertoire_id"] + self.feature_cols]
        return X, (np.array(y) if len(y) else None), self.feature_cols

    # ---- required API ----
    def train(self, train_dir: str, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.top_kmers, self.v_vocab, self.j_vocab = self._collect_feature_vocab(train_dir)
        meta = pd.read_csv(os.path.join(train_dir, "metadata.csv"))
        X, y, _ = self._build_matrix(train_dir, meta)
        X_mat = X[self.feature_cols].values

        base = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga",
                l1_ratio=0.5, C=1.0, max_iter=5000, n_jobs=self.n_jobs,
                random_state=self.random_state)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=cv)
        self.model.fit(X_mat, y)

        with open(os.path.join(out_dir, "feature_spec.json"), "w") as f:
            json.dump({
                "k": self.k,
                "top_kmers": self.top_kmers,
                "v_vocab": self.v_vocab,
                "j_vocab": self.j_vocab,
                "feature_cols": self.feature_cols,
            }, f)

    def predict(self, test_dir: str, out_dir: str) -> pd.DataFrame:
        os.makedirs(out_dir, exist_ok=True)
        meta = pd.read_csv(os.path.join(test_dir, "metadata.csv"))
        X, _, _ = self._build_matrix(test_dir, meta)
        probs = self.model.predict_proba(X[self.feature_cols].values)[:, 1]
        pred = pd.DataFrame({
            "repertoire_id": X["repertoire_id"],
            "label_positive_probability": probs,
        })
        pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        return pred

    def rank_sequences(self, train_dir: str, out_dir: str, topk: int = 50000) -> pd.DataFrame:
        os.makedirs(out_dir, exist_ok=True)
        meta = pd.read_csv(os.path.join(train_dir, "metadata.csv"))
        rows = []
        for _, row in meta.iterrows():
            rep_id = row["repertoire_id"]
            lab = _to_bool(row["label_positive"])
            df_r = _read_repertoire_tsv(os.path.join(train_dir, row["filename"]))
            df_r = df_r.drop_duplicates(subset=["junction_aa", "v_call", "j_call"])
            df_r["repertoire_id"] = rep_id
            df_r["label_positive"] = lab
            rows.append(df_r)
        pres = pd.concat(rows, ignore_index=True)

        pos_reps = pres.loc[pres["label_positive"] == True, "repertoire_id"].nunique()
        neg_reps = pres.loc[pres["label_positive"] == False, "repertoire_id"].nunique()

        grp = pres.groupby(["junction_aa", "v_call", "j_call", "label_positive"])['repertoire_id'].nunique()
        keys = pres[["junction_aa", "v_call", "j_call"]].drop_duplicates()

        out_rows = []
        for _, krow in keys.iterrows():
            jaa, v, j = krow["junction_aa"], krow["v_call"], krow["j_call"]
            a = int(grp.get((jaa, v, j, True), 0))
            c = int(grp.get((jaa, v, j, False), 0))
            b = pos_reps - a
            d = neg_reps - c
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            score = float(np.log(odds + 1e-9) * -np.log10(p + 1e-300))
            out_rows.append((jaa, v, j, score, odds, p, a, c))

        out = pd.DataFrame(out_rows, columns=["junction_aa", "v_call", "j_call",
                                              "score", "odds", "p", "pos_pres", "neg_pres"])
        out = out.sort_values(["score", "pos_pres", "neg_pres"], ascending=[False, False, True]).head(topk)
        out.to_csv(os.path.join(out_dir, "ranked_sequences.csv"), index=False)
        return out
