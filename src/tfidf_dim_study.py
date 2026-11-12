"""Feature-dimension study for question (g) (member A).

Fixed classifier (LR, C=10, class_weight=balanced, the selected TF-IDF+LR
setting); varies the TF-IDF vocabulary size and the optional character
n-grams. Trained on the training part, evaluated on the validation split.
Records feature dimension, non-zeros per review, sparse-matrix memory,
fit time, validation OOV token rate and accuracy.

    python src/tfidf_dim_study.py
Output: outputs/resource/tfidf_dim_study.csv / .md
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT_DIR, best_threshold, clean_text, evaluate, get_split, load_data  # noqa: E402
from tfidf_models import TOKEN_PATTERN, build_features  # noqa: E402


def main():
    train_df, _ = load_data()
    texts = np.array([clean_text(s) for s in train_df["reviews"]], dtype=object)
    y = train_df["sentiments"].values.astype(int)
    tr_idx, val_idx = get_split()

    tok = re.compile(TOKEN_PATTERN)
    tr_vocab = {}
    for t in texts[tr_idx]:
        for w in tok.findall(t):
            tr_vocab[w] = tr_vocab.get(w, 0) + 1
    val_tokens = [w for t in texts[val_idx] for w in tok.findall(t)]

    rows = []
    for char in (False, True):
        for mf in (1_000, 5_000, 10_000, 30_000, 50_000, None):
            f = dict(min_df=1, max_features=mf, char=char)
            t0 = time.perf_counter()
            Xtr, Xval = build_features(f, texts[tr_idx], texts[val_idx])
            clf = LogisticRegression(C=10, class_weight="balanced", solver="liblinear",
                                     max_iter=2000, random_state=42).fit(Xtr, y[tr_idx])
            fit_s = time.perf_counter() - t0
            p = clf.predict_proba(Xval)[:, 1]
            e05 = evaluate(y[val_idx], p, 0.5)
            tf1 = best_threshold(y[val_idx], p, "macro_f1")
            ef1 = evaluate(y[val_idx], p, tf1)
            mb = (Xtr.data.nbytes + Xtr.indices.nbytes + Xtr.indptr.nbytes) / 2**20
            rows.append({
                "char_2_5": char, "max_features": "all" if mf is None else mf,
                "feature_dim": Xtr.shape[1],
                "nnz_per_review": round(Xtr.nnz / Xtr.shape[0], 1),
                "density_%": round(100 * Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1]), 3),
                "train_matrix_mb": round(mb, 1),
                "vectorize+fit_s": round(fit_s, 2),
                "val_acc@0.5": round(e05["acc"], 4),
                "val_macro_f1@0.5": round(e05["macro_f1"], 4),
                "val_macro_f1@thr_f1": round(ef1["macro_f1"], 4),
                "val_auc": round(e05["auc"], 4),
            })
            print(rows[-1])

    oov_types = 1 - np.mean([w in tr_vocab for w in set(val_tokens)])
    oov_tokens = 1 - np.mean([w in tr_vocab for w in val_tokens])
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "resource"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tfidf_dim_study.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    md += ["", f"Unigram vocabulary of the training part: {len(tr_vocab)} types. "
               f"Validation OOV rate w.r.t. that vocabulary: {oov_tokens:.4f} of tokens, "
               f"{oov_types:.4f} of types."]
    (out / "tfidf_dim_study.md").write_text("\n".join(md), encoding="utf-8")
    print(md[-1])


if __name__ == "__main__":
    main()
