"""L1: TF-IDF features + linear classifiers (member A).

Grid search with grouped, stratified 5-fold CV inside the training part
(common.group_cv_folds). Configurations are selected on CV only; the shared
validation split is used to report results and to pick thresholds.

For each classifier family (LR, calibrated LinearSVC, ComplementNB):
  outputs/tfidf_<family>/val_probs.csv   model trained on the train part (5909)
  outputs/tfidf_<family>/test_probs.csv  same config refit on all 7401 rows
  outputs/tfidf_<family>/metrics.json, hparam_log.csv
The TF-IDF+LR model also writes the baseline submission.csv (thr_acc).

Run from the project root:
    python src/tfidf_models.py          # full grid
    python src/tfidf_models.py --quick  # small grid for a fast check
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (GLOBAL_SEED, ResourceMeter, best_threshold, clean_text,  # noqa: E402
                    evaluate, get_split, group_cv_folds, load_data,
                    make_metrics, save_outputs, set_seed, text_groups,
                    write_submission)

# Keep single-character tokens and ! ?  (default pattern drops them).
TOKEN_PATTERN = r"(?u)\b\w+\b|[!?]"


# ---------------------------------------------------------------- grids
def feature_grid(quick: bool):
    if quick:
        return [dict(min_df=2, max_features=None, char=c) for c in (False, True)]
    return [dict(min_df=m, max_features=f, char=c)
            for m, f, c in itertools.product((1, 2, 3),
                                             (10_000, 30_000, 50_000, None),
                                             (False, True))]


def clf_grid(quick: bool):
    cws = (None, "balanced")
    if quick:
        return ([("lr", dict(C=10, class_weight=None))]
                + [("svm", dict(C=1, class_weight=None))]
                + [("cnb", dict(alpha=0.3))])
    return ([("lr", dict(C=C, class_weight=w)) for C in (0.1, 1, 10) for w in cws]
            + [("svm", dict(C=C, class_weight=w)) for C in (0.1, 1) for w in cws]
            + [("cnb", dict(alpha=a)) for a in (0.1, 0.3, 1.0)])


def fcfg_name(f):
    mf = "all" if f["max_features"] is None else f"{f['max_features'] // 1000}k"
    return f"word12_mindf{f['min_df']}_max{mf}" + ("+char25" if f["char"] else "")


def ccfg_name(fam, c):
    return f"{fam}(" + ",".join(f"{k}={v}" for k, v in c.items()) + ")"


# ---------------------------------------------------------------- building blocks
def build_features(fcfg, fit_texts, *other_texts):
    """Fit TF-IDF on fit_texts only; transform the others with it."""
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=fcfg["min_df"],
                           max_features=fcfg["max_features"], sublinear_tf=True,
                           token_pattern=TOKEN_PATTERN, lowercase=False)
    mats = [word.fit_transform(fit_texts)] + [word.transform(t) for t in other_texts]
    if fcfg["char"]:
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                               sublinear_tf=True, lowercase=False)
        cm = [char.fit_transform(fit_texts)] + [char.transform(t) for t in other_texts]
        mats = [hstack([a, b]).tocsr() for a, b in zip(mats, cm)]
    return mats


def inner_folds(texts, y, k=3):
    """Grouped folds inside a fit set, for CalibratedClassifierCV."""
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=GLOBAL_SEED)
    return list(sgkf.split(np.zeros(len(y)), y, text_groups(texts)))


def make_clf(fam, c, fit_texts=None, y_fit=None):
    if fam == "lr":
        return LogisticRegression(C=c["C"], class_weight=c["class_weight"],
                                  solver="liblinear", max_iter=2000,
                                  random_state=GLOBAL_SEED)
    if fam == "svm":
        base = LinearSVC(C=c["C"], class_weight=c["class_weight"],
                         max_iter=5000, random_state=GLOBAL_SEED)
        return CalibratedClassifierCV(estimator=base, method="sigmoid",
                                      cv=inner_folds(fit_texts, y_fit))
    if fam == "cnb":
        return ComplementNB(alpha=c["alpha"])
    raise ValueError(fam)


def run_task(fi, fcfg, fold_id, fit_idx, hold_idx, texts, y, clfs):
    """Vectorize once for (feature config, fold), fit every classifier."""
    Xf, Xh = build_features(fcfg, texts[fit_idx], texts[hold_idx])
    out = {}
    for ci, (fam, c) in enumerate(clfs):
        clf = make_clf(fam, c, texts[fit_idx], y[fit_idx])
        clf.fit(Xf, y[fit_idx])
        out[ci] = clf.predict_proba(Xh)[:, 1]
    return fi, fold_id, hold_idx, out


def n_params(clf, dim):
    if isinstance(clf, ComplementNB):
        return int(clf.feature_log_prob_.size)
    if isinstance(clf, CalibratedClassifierCV):
        return int(sum(cc.estimator.coef_.size + 1 + 2 for cc in clf.calibrated_classifiers_))
    return int(clf.coef_.size + clf.intercept_.size)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    set_seed(GLOBAL_SEED)

    train_df, test_df = load_data()
    texts = np.array([clean_text(s, lower=True) for s in train_df["reviews"]], dtype=object)
    test_texts = np.array([clean_text(s, lower=True) for s in test_df["reviews"]], dtype=object)
    y = train_df["sentiments"].values.astype(int)
    tr_idx, val_idx = get_split()
    folds = group_cv_folds(tr_idx, k=5)
    # fold id 5 = (train part -> validation): logged, never used for selection
    all_folds = folds + [(tr_idx, val_idx)]

    fgrid, clfs = feature_grid(args.quick), clf_grid(args.quick)
    print(f"{len(fgrid)} feature configs x {len(clfs)} classifier configs, "
          f"{len(fgrid) * len(all_folds)} vectorize tasks")
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(fi, f, k, a, b, texts, y, clfs)
        for fi, f in enumerate(fgrid) for k, (a, b) in enumerate(all_folds))
    print(f"grid done in {time.time() - t0:.0f}s")

    # out-of-fold probabilities per (feature, classifier) config
    oof = {}
    valp = {}
    for fi, k, hold, out in results:
        for ci, p in out.items():
            if k == 5:
                valp[(fi, ci)] = p
            else:
                oof.setdefault((fi, ci), np.full(len(y), np.nan))[hold] = p

    rows = []
    for (fi, ci), p_all in oof.items():
        p = p_all[tr_idx]
        yt = y[tr_idx]
        fam, c = clfs[ci]
        t_f1 = best_threshold(yt, p, "macro_f1")
        t_acc = best_threshold(yt, p, "acc")
        cv05, cvf1, cvacc = evaluate(yt, p, 0.5), evaluate(yt, p, t_f1), evaluate(yt, p, t_acc)
        v05 = evaluate(y[val_idx], valp[(fi, ci)], 0.5)
        vt = evaluate(y[val_idx], valp[(fi, ci)], t_f1)
        rows.append({
            "family": fam, "features": fcfg_name(fgrid[fi]), "classifier": ccfg_name(fam, c),
            **{f"f_{k}": v for k, v in fgrid[fi].items()},
            **{f"c_{k}": v for k, v in c.items()},
            "cv_auc": cv05["auc"], "cv_macro_f1@0.5": cv05["macro_f1"],
            "cv_acc@0.5": cv05["acc"], "cv_thr_f1": t_f1,
            "cv_macro_f1@thr": cvf1["macro_f1"], "cv_acc@thr_acc": cvacc["acc"],
            "val_macro_f1@0.5": v05["macro_f1"], "val_acc@0.5": v05["acc"],
            "val_macro_f1@cv_thr": vt["macro_f1"], "val_acc@cv_thr": vt["acc"],
            "val_auc": v05["auc"], "_fi": fi, "_ci": ci,
        })
    log = pd.DataFrame(rows).sort_values("cv_macro_f1@thr", ascending=False)

    names = {"lr": ("tfidf_lr", "TF-IDF + Logistic Regression"),
             "svm": ("tfidf_svm", "TF-IDF + LinearSVC (sigmoid-calibrated)"),
             "cnb": ("tfidf_cnb", "TF-IDF + Complement Naive Bayes")}
    summary = []
    for fam, (out_name, desc) in names.items():
        flog = log[log["family"] == fam]
        if flog.empty:
            continue
        best = flog.iloc[0]
        fi, ci = int(best["_fi"]), int(best["_ci"])
        fcfg, (_, c) = fgrid[fi], clfs[ci]

        # 1) train part -> validation (reported metrics, thresholds)
        set_seed(GLOBAL_SEED)
        with ResourceMeter() as rm_tr:
            Xtr, Xval = build_features(fcfg, texts[tr_idx], texts[val_idx])
            clf = make_clf(fam, c, texts[tr_idx], y[tr_idx])
            clf.fit(Xtr, y[tr_idx])
        with ResourceMeter() as rm_inf:
            pv = clf.predict_proba(Xval)[:, 1]
        dim = Xtr.shape[1]
        params = n_params(clf, dim)

        # 2) same config refit on all 7401 rows -> test probabilities
        Xall, Xte = build_features(fcfg, texts, test_texts)
        clf_all = make_clf(fam, c, texts, y)
        clf_all.fit(Xall, y)
        pt = clf_all.predict_proba(Xte)[:, 1]

        metrics = make_metrics(
            desc, f"TF-IDF {fcfg_name(fcfg)} (sparse)", y[val_idx], pv,
            seeds=[GLOBAL_SEED], n_params=params, feature_dim=int(dim),
            train_time_s=round(rm_tr.seconds, 3),
            infer_time_s=round(rm_inf.seconds, 4),
            peak_mem_mb=round(rm_tr.peak_mb, 1),
            selected_config={"features": fcfg, "classifier": c},
            cv_macro_f1_at_thr=float(best["cv_macro_f1@thr"]),
            cv_auc=float(best["cv_auc"]),
            sparse=True,
            test_model="same config refit on all 7401 training rows; "
                       "thresholds taken from the validation split",
            note="deterministic; seed_std is 0 by construction",
        )
        save_outputs(
            out_name,
            pd.DataFrame({"idx": val_idx, "label": y[val_idx], "prob_pos": pv}),
            pd.DataFrame({"idx": np.arange(len(pt)), "prob_pos": pt}),
            metrics,
            flog.drop(columns=["_fi", "_ci"]).reset_index(drop=True),
        )
        summary.append((out_name, metrics, pt))
        print(f"\n[{out_name}] {fcfg_name(fcfg)} {ccfg_name(fam, c)}\n"
              f"  CV macro-F1@thr {best['cv_macro_f1@thr']:.4f} | val acc@0.5 "
              f"{metrics['val_acc@0.5']:.4f} macroF1@0.5 {metrics['val_macro_f1@0.5']:.4f} "
              f"AUC {metrics['val_auc']:.4f} | thr_acc {metrics['thr_acc']} -> acc "
              f"{metrics['val_acc@thr_acc']:.4f} | thr_f1 {metrics['thr_f1']} -> macroF1 "
              f"{metrics['val_macro_f1@thr_f1']:.4f} | dim {dim} train {rm_tr.seconds:.1f}s")

    log.drop(columns=["_fi", "_ci"]).to_csv(
        Path(__file__).resolve().parents[1] / "outputs" / "tfidf_grid_all.csv", index=False)

    # baseline submission from TF-IDF + LR
    for out_name, metrics, pt in summary:
        if out_name == "tfidf_lr":
            write_submission(pt, metrics["thr_acc"])


if __name__ == "__main__":
    main()
