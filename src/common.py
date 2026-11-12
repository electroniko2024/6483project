"""Shared utilities for the EE6483 sentiment project.

Owned by member A. Other members import from here and do not edit it;
ask A first if something needs to change.

Usage from any script in src/:
    from common import load_data, get_split, evaluate, ...
"""
from __future__ import annotations

import html
import json
import os
import platform
import random
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_PATH = ROOT / "splits" / "val_idx.npy"
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_PATH = ROOT / "submission.csv"

GLOBAL_SEED = 42
N_TEST = 1851


# ---------------------------------------------------------------- randomness
def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Fix python, numpy and (if installed) torch / cudnn randomness."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------- data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df). train has columns reviews, sentiments;
    test has reviews. Row order is the original file order."""
    train_df = pd.read_json(DATA_DIR / "train.json")
    test_df = pd.read_json(DATA_DIR / "test.json")
    assert len(test_df) == N_TEST, len(test_df)
    return train_df, test_df


def normalize_text(s: str) -> str:
    """Grouping key for duplicate detection. Fixed definition, do not change:
    lower-case, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def text_groups(texts) -> np.ndarray:
    """Integer group id per row; identical normalized texts share an id."""
    return pd.factorize(pd.Series(list(texts)).map(normalize_text))[0]


def clean_text(s: str, lower: bool = True) -> str:
    """Light cleaning: HTML unescape, collapse whitespace, strip.
    Keeps negations (not, n't) and ! ? punctuation.
    Use lower=False for Transformers (keep original casing)."""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if lower else s


# ---------------------------------------------------------------- splits
def make_split(save: bool = True) -> np.ndarray:
    """Create the shared validation split. Only A runs this, once."""
    train_df, _ = load_data()
    groups = text_groups(train_df["reviews"])
    y = train_df["sentiments"].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    _, val_idx = next(iter(sgkf.split(train_df, y, groups)))
    val_idx = np.sort(val_idx).astype(np.int64)
    if save:
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(SPLIT_PATH, val_idx)
    return val_idx


def get_split() -> tuple[np.ndarray, np.ndarray]:
    """Return (tr_idx, val_idx): row numbers into train.json."""
    val_idx = np.load(SPLIT_PATH).astype(np.int64)
    n = len(load_data()[0])
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    return tr_idx, val_idx


def group_cv_folds(tr_idx: np.ndarray, k: int = 5, seed: int = GLOBAL_SEED):
    """Grouped + stratified CV inside the training part.
    Returns a list of (fit_idx, holdout_idx); both are row numbers into
    train.json (subsets of tr_idx)."""
    train_df, _ = load_data()
    tr_idx = np.asarray(tr_idx)
    sub = train_df.iloc[tr_idx]
    groups = text_groups(sub["reviews"])
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr_idx[a], tr_idx[b])
            for a, b in sgkf.split(sub, sub["sentiments"].values, groups)]


def positional_folds(folds, subset_idx: np.ndarray):
    """Convert folds of train.json row numbers into positions within
    subset_idx (what sklearn's cv= argument expects when X = X_all[subset_idx])."""
    pos = {r: i for i, r in enumerate(np.asarray(subset_idx))}
    return [(np.array([pos[r] for r in a]), np.array([pos[r] for r in b]))
            for a, b in folds]


# ---------------------------------------------------------------- metrics
def evaluate(y_true, prob, thr: float = 0.5) -> dict:
    """All metrics for P(positive)=prob at threshold thr."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= thr).astype(int)
    out = {
        "thr": float(thr),
        "acc": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "neg_f1": float(f1_score(y_true, pred, pos_label=0)),
        "pos_f1": float(f1_score(y_true, pred, pos_label=1)),
        "confusion": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def best_threshold(y_true, prob, metric: str = "acc") -> float:
    """Threshold in [0.01, 0.99] (step 0.005) maximising 'acc' or 'macro_f1'.
    Ties are broken towards 0.5."""
    assert metric in ("acc", "macro_f1")
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    grid = np.round(np.arange(0.01, 0.9901, 0.005), 3)
    best, best_score = 0.5, -1.0
    for t in sorted(grid, key=lambda t: abs(t - 0.5)):
        pred = (prob >= t).astype(int)
        score = (accuracy_score(y_true, pred) if metric == "acc"
                 else f1_score(y_true, pred, average="macro"))
        if score > best_score + 1e-12:
            best, best_score = float(t), score
    return best


def make_metrics(model: str, feature: str, y_val, val_prob, seeds=(GLOBAL_SEED,),
                 seed_macro_f1s=None, **extra) -> dict:
    """Build the metrics.json dict in the agreed format.
    val_prob: final validation probabilities (seed-averaged if several seeds).
    seed_macro_f1s: per-seed val Macro-F1 @0.5, for seed_std_macro_f1.
    extra: n_params, feature_dim, train_time_s, infer_time_s, peak_mem_mb,
           hardware, plus anything else worth recording."""
    m05 = evaluate(y_val, val_prob, 0.5)
    t_acc = best_threshold(y_val, val_prob, "acc")
    t_f1 = best_threshold(y_val, val_prob, "macro_f1")
    m_acc = evaluate(y_val, val_prob, t_acc)
    m_f1 = evaluate(y_val, val_prob, t_f1)
    out = {
        "model": model, "feature": feature, "seeds": list(seeds),
        "val_acc@0.5": m05["acc"], "val_macro_f1@0.5": m05["macro_f1"],
        "val_neg_f1@0.5": m05["neg_f1"], "val_auc": m05["auc"],
        "confusion@0.5": m05["confusion"],
        "thr_acc": t_acc, "val_acc@thr_acc": m_acc["acc"],
        "val_macro_f1@thr_acc": m_acc["macro_f1"],
        "thr_f1": t_f1, "val_macro_f1@thr_f1": m_f1["macro_f1"],
        "val_acc@thr_f1": m_f1["acc"],
        "seed_std_macro_f1": (float(np.std(seed_macro_f1s)) if seed_macro_f1s
                              is not None and len(seed_macro_f1s) > 1 else 0.0),
        "n_params": 0, "feature_dim": 0, "train_time_s": 0.0,
        "infer_time_s": 0.0, "peak_mem_mb": 0.0, "hardware": hardware(),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------- outputs
def save_outputs(model_name: str, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 metrics: dict, hparam_log: pd.DataFrame | None = None) -> Path:
    """Write outputs/<model_name>/{val_probs,test_probs}.csv, metrics.json
    and (optionally) hparam_log.csv.
    val_df: columns idx (train.json row), label, prob_pos.
    test_df: columns idx (test.json row 0..1850), prob_pos."""
    out = OUTPUT_DIR / model_name
    out.mkdir(parents=True, exist_ok=True)
    val_df = val_df[["idx", "label", "prob_pos"]].sort_values("idx")
    test_df = test_df[["idx", "prob_pos"]].sort_values("idx")
    assert len(test_df) == N_TEST and test_df["idx"].tolist() == list(range(N_TEST))
    val_df.to_csv(out / "val_probs.csv", index=False)
    test_df.to_csv(out / "test_probs.csv", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if hparam_log is not None:
        hparam_log.to_csv(out / "hparam_log.csv", index=False)
    return out


def write_submission(test_prob, thr: float, path=SUBMISSION_PATH) -> pd.DataFrame:
    """Write submission.csv (reviews, predictions) in test.json order and
    verify it by reading it back."""
    _, test_df = load_data()
    test_prob = np.asarray(test_prob, dtype=float)
    assert len(test_prob) == N_TEST
    sub = pd.DataFrame({"reviews": test_df["reviews"],
                        "predictions": (test_prob >= thr).astype(int)})
    sub.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert list(back.columns) == ["reviews", "predictions"], back.columns
    assert len(back) == N_TEST, len(back)
    assert set(back["predictions"].unique()) <= {0, 1}
    assert (back["reviews"] == test_df["reviews"]).all(), "row order mismatch"
    print(f"wrote {path}: {len(back)} rows, positive rate "
          f"{back['predictions'].mean():.3f}, thr={thr}")
    return back


# ---------------------------------------------------------------- resources
def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    desc = f"CPU: {cpu} ({os.cpu_count()} threads)"
    try:
        import torch
        if torch.cuda.is_available():
            desc += f"; GPU: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return desc


class ResourceMeter:
    """Context manager that records wall time and peak memory.

        with ResourceMeter() as rm:
            model.fit(X, y)
        rm.seconds, rm.peak_mb

    On GPU (gpu=True and CUDA available) peak_mb is
    torch.cuda.max_memory_allocated; otherwise it is the peak process RSS
    sampled every 50 ms (psutil), i.e. total process memory during the block.
    """

    def __init__(self, gpu: bool = False, interval: float = 0.05):
        self.gpu = gpu
        self.interval = interval
        self.seconds = 0.0
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def _use_cuda(self) -> bool:
        if not self.gpu:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _sample(self):
        import psutil
        proc = psutil.Process()
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, proc.memory_info().rss / 2**20)
            time.sleep(self.interval)

    def __enter__(self):
        self._cuda = self._use_cuda()
        if self._cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        else:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._cuda:
            import torch
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._t0
        if self._cuda:
            import torch
            self.peak_mb = torch.cuda.max_memory_allocated() / 2**20
        else:
            self._stop.set()
            self._thread.join()
        return False


if __name__ == "__main__":
    # Sanity check / one-off split creation:  python src/common.py [--make-split]
    import sys
    if "--make-split" in sys.argv:
        idx = make_split(save=True)
        print("saved", SPLIT_PATH, len(idx))
    tr_idx, val_idx = get_split()
    train_df, test_df = load_data()
    y = train_df["sentiments"].values
    g = text_groups(train_df["reviews"])
    print(f"train part {len(tr_idx)}, val {len(val_idx)}, "
          f"val negatives {(y[val_idx] == 0).sum()}, "
          f"group overlap {len(set(g[tr_idx]) & set(g[val_idx]))}")
