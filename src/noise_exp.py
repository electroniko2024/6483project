"""Noisy-label experiment for question (i) (member A).

Labels in the training part are corrupted; the validation split stays clean.
Base model: TF-IDF (word 1-2gram, min_df=2, sublinear tf) + Logistic
Regression (C=10, liblinear). The TF-IDF vocabulary/IDF is fitted once on the
training-part texts (unsupervised, so label noise does not affect it).

Noise settings (3 seeds each):
  symmetric : every training label flipped independently with prob r,
              r in {0, 0.05, 0.10, 0.20}
  pos2neg   : only positive labels flipped to negative with prob r,
              r in {0.05, 0.10, 0.20} (e.g. low score on a positive review)

Methods:
  M0 baseline      train on the noisy labels.
  M1 filter        Confident-Learning style pruning (Northcutt et al. 2021):
                   5-fold out-of-fold probabilities; per-class threshold
                   t_j = mean P(j) over samples whose given label is j; a
                   sample is flagged when the most likely class among those
                   with P(k) >= t_k differs from its given label; flagged
                   samples are removed and the model is retrained.
  M2 relabel       samples whose out-of-fold confidence for the opposite
                   label is > 0.9 get their label flipped, then retrain.
  M3 reweight      sample_weight = out-of-fold P(given label), then retrain.

Run from the project root:   python src/noise_exp.py
Outputs: outputs/noise_exp/{results_raw.csv, summary.csv, summary_table.md,
         noise_symmetric.png, noise_pos2neg.png} (figures also copied to
         report/figures/).
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUTPUT_DIR, ROOT, clean_text, evaluate, get_split,  # noqa: E402
                    group_cv_folds, load_data, positional_folds, set_seed)

SETTINGS = ([("symmetric", r) for r in (0.0, 0.05, 0.10, 0.20)]
            + [("pos2neg", r) for r in (0.05, 0.10, 0.20)])
SEEDS = (42, 43, 44)
METHODS = ("M0 baseline", "M1 filter (CL)", "M2 relabel", "M3 reweight")
RELABEL_CONF = 0.9


def new_clf():
    return LogisticRegression(C=10, solver="liblinear", max_iter=2000, random_state=42)


def corrupt(y, kind, rate, rng):
    y_noisy = y.copy()
    u = rng.random(len(y))
    if kind == "symmetric":
        flip = u < rate
    elif kind == "pos2neg":
        flip = (y == 1) & (u < rate)
    else:
        raise ValueError(kind)
    y_noisy[flip] = 1 - y_noisy[flip]
    return y_noisy, flip


def oof_probs(X, y_noisy, folds):
    """Out-of-fold P(positive) for every training-part row."""
    p = np.zeros(len(y_noisy))
    for a, b in folds:
        clf = new_clf().fit(X[a], y_noisy[a])
        p[b] = clf.predict_proba(X[b])[:, 1]
    return p


def cl_flags(p_pos, y_noisy):
    """Confident-Learning off-diagonal samples for the binary case."""
    P = np.column_stack([1 - p_pos, p_pos])
    t = np.array([P[y_noisy == j, j].mean() for j in (0, 1)])
    above = P >= t
    masked = np.where(above, P, -1.0)
    assigned = masked.argmax(1)
    has = above.any(1)
    return has & (assigned != y_noisy)


def main():
    t0 = time.time()
    set_seed(42)
    train_df, _ = load_data()
    texts = np.array([clean_text(s) for s in train_df["reviews"]], dtype=object)
    y_all = train_df["sentiments"].values.astype(int)
    tr_idx, val_idx = get_split()

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          token_pattern=r"(?u)\b\w+\b|[!?]", lowercase=False)
    Xtr = vec.fit_transform(texts[tr_idx])
    Xval = vec.transform(texts[val_idx])
    y_tr, y_val = y_all[tr_idx], y_all[val_idx]

    rows = []
    for seed in SEEDS:
        folds = positional_folds(group_cv_folds(tr_idx, k=5, seed=seed), tr_idx)
        for kind, rate in SETTINGS:
            rng = np.random.default_rng(seed * 1000 + int(rate * 100) + (500 if kind == "pos2neg" else 0))
            y_noisy, flipped = corrupt(y_tr, kind, rate, rng)
            neg_lab = y_noisy == 0
            neg_label_precision = float((y_tr[neg_lab] == 0).mean())
            p = oof_probs(Xtr, y_noisy, folds)
            p_given = np.where(y_noisy == 1, p, 1 - p)

            variants = {}
            variants[METHODS[0]] = (np.ones(len(y_tr), bool), y_noisy, None)
            flags = cl_flags(p, y_noisy)
            variants[METHODS[1]] = (~flags, y_noisy, None)
            y_rel = y_noisy.copy()
            relabel = (1 - p_given) > RELABEL_CONF
            y_rel[relabel] = 1 - y_rel[relabel]
            variants[METHODS[2]] = (np.ones(len(y_tr), bool), y_rel, None)
            variants[METHODS[3]] = (np.ones(len(y_tr), bool), y_noisy, p_given)

            for m, (keep, yy, w) in variants.items():
                clf = new_clf()
                clf.fit(Xtr[keep], yy[keep], sample_weight=None if w is None else w[keep])
                ev = evaluate(y_val, clf.predict_proba(Xval)[:, 1], 0.5)
                row = {"noise": kind, "rate": rate, "seed": seed, "method": m,
                       "val_acc": ev["acc"], "val_macro_f1": ev["macro_f1"],
                       "val_neg_f1": ev["neg_f1"], "val_auc": ev["auc"],
                       "n_flipped": int(flipped.sum()),
                       "neg_label_precision": neg_label_precision,
                       "n_train_used": int(keep.sum())}
                if m == METHODS[1]:
                    row["removed"] = int(flags.sum())
                    row["removed_precision"] = (float(flipped[flags].mean())
                                                if flags.any() else float("nan"))
                    row["flip_recall"] = (float(flags[flipped].mean())
                                          if flipped.any() else float("nan"))
                if m == METHODS[2]:
                    row["relabeled"] = int(relabel.sum())
                    row["relabel_precision"] = (float(flipped[relabel].mean())
                                                if relabel.any() else float("nan"))
                rows.append(row)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)")

    out = OUTPUT_DIR / "noise_exp"
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "results_raw.csv", index=False)

    agg_cols = ["val_acc", "val_macro_f1", "val_neg_f1", "val_auc",
                "neg_label_precision", "n_train_used", "removed",
                "removed_precision", "flip_recall", "relabeled", "relabel_precision"]
    agg_cols = [c for c in agg_cols if c in raw]
    summ = raw.groupby(["noise", "rate", "method"])[agg_cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "summary.csv", index=False)

    # markdown table: macro-F1 / accuracy, mean ± std
    lines = ["| Noise | Rate | Neg-label precision | " + " | ".join(METHODS) + " |",
             "|---|---|---|" + "---|" * len(METHODS)]
    for kind, rate in SETTINGS:
        s = summ[(summ.noise == kind) & (summ.rate == rate)].set_index("method")
        cells = [f"{s.loc[m, 'val_macro_f1_mean']:.3f}±{s.loc[m, 'val_macro_f1_std']:.3f} / "
                 f"{s.loc[m, 'val_acc_mean']:.3f}" for m in METHODS]
        lines.append(f"| {kind} | {rate:.0%} | {s['neg_label_precision_mean'].iloc[0]:.2f} | "
                     + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells: validation Macro-F1 mean±std over 3 seeds / accuracy, threshold 0.5, "
                 "clean validation split (1492 rows).")
    lines.append("")
    lines.append("Threshold-free view (validation ROC-AUC, mean±std). At threshold 0.5 part of "
                 "the Macro-F1 change comes from the decision boundary shifting with the label "
                 "prior, AUC isolates ranking quality.")
    lines.append("")
    lines.append("| Noise | Rate | " + " | ".join(METHODS) + " |")
    lines.append("|---|---|" + "---|" * len(METHODS))
    for kind, rate in SETTINGS:
        s = summ[(summ.noise == kind) & (summ.rate == rate)].set_index("method")
        lines.append(f"| {kind} | {rate:.0%} | " + " | ".join(
            f"{s.loc[m, 'val_auc_mean']:.4f}±{s.loc[m, 'val_auc_std']:.4f}" for m in METHODS) + " |")
    m1 = summ[summ.method == METHODS[1]]
    lines.append("")
    lines.append("| Noise | Rate | M1 removed | M1 removed that were flipped | M1 recall of flipped |")
    lines.append("|---|---|---|---|---|")
    for _, r in m1.iterrows():
        lines.append(f"| {r.noise} | {r.rate:.0%} | {r.removed_mean:.0f} | "
                     f"{r.removed_precision_mean:.2f} | {r.flip_recall_mean:.2f} |")
    (out / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")

    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    titles = {"symmetric": "Symmetric label noise", "pos2neg": "Positive→negative label noise"}
    for kind in ("symmetric", "pos2neg"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        sk = summ[summ.noise == kind]
        if kind == "pos2neg":  # include the clean point for reference
            sk = pd.concat([summ[(summ.noise == "symmetric") & (summ.rate == 0)].assign(noise="pos2neg"), sk])
        for ax, col, lab in ((axes[0], "val_macro_f1", "Validation Macro-F1 (thr 0.5)"),
                             (axes[1], "val_auc", "Validation ROC-AUC")):
            for m in METHODS:
                s = sk[sk.method == m].sort_values("rate")
                ax.errorbar(s.rate * 100, s[f"{col}_mean"], yerr=s[f"{col}_std"],
                            marker="o", capsize=3, label=m)
            ax.set_xlabel("Flip rate in training part (%)")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        fig.suptitle(titles[kind])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fn = out / f"noise_{kind}.png"
        fig.savefig(fn, dpi=200)
        plt.close(fig)
        shutil.copy(fn, fig_dir / fn.name)

    print("\n".join(lines))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
