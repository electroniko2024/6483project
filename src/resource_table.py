"""Resource vs accuracy comparison for question (g) (member A).

Reads every outputs/<model>/metrics.json and writes
  outputs/resource/resource_table.csv / .md
  outputs/resource/f1_vs_train_time.png  (also copied to report/figures/)

Run again whenever a member pushes new outputs:
    python src/resource_table.py
Optional fields picked up if present in metrics.json: oov_rate (B),
sparse (bool), seed_std_macro_f1.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT_DIR, ROOT  # noqa: E402

COLS = ["name", "model", "feature", "feature_dim", "sparse", "n_params",
        "val_acc@0.5", "val_macro_f1@0.5", "val_auc", "thr_acc", "val_acc@thr_acc",
        "thr_f1", "val_macro_f1@thr_f1", "seed_std_macro_f1", "train_time_s",
        "infer_time_s", "peak_mem_mb", "oov_rate", "hardware"]


def main():
    rows = []
    for f in sorted(OUTPUT_DIR.glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        m["name"] = f.parent.name
        rows.append(m)
    if not rows:
        print("no metrics.json found under outputs/")
        return
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df:
            df[c] = None
    df = df[COLS].sort_values("val_macro_f1@thr_f1", ascending=False)

    out = OUTPUT_DIR / "resource"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "resource_table.csv", index=False)

    def fmt(v, nd=4):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "–"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    md = ["| Model | Feature dim | Sparse | Params | Val Acc@0.5 | Val Macro-F1@0.5 | AUC | "
          "Acc@thr_acc | Macro-F1@thr_f1 | Seed std | Train s | Infer s | Peak MB | OOV | Hardware |",
          "|" + "---|" * 15]
    for _, r in df.iterrows():
        md.append("| " + " | ".join([
            r["name"], fmt(r["feature_dim"]), fmt(r["sparse"]), fmt(r["n_params"]),
            fmt(r["val_acc@0.5"]), fmt(r["val_macro_f1@0.5"]), fmt(r["val_auc"]),
            fmt(r["val_acc@thr_acc"]), fmt(r["val_macro_f1@thr_f1"]),
            fmt(r["seed_std_macro_f1"]), fmt(r["train_time_s"], 1),
            fmt(r["infer_time_s"], 3), fmt(r["peak_mem_mb"], 0), fmt(r["oov_rate"], 3),
            fmt(r["hardware"])]) + " |")
    (out / "resource_table.md").write_text("\n".join(md), encoding="utf-8")

    plot = df.dropna(subset=["train_time_s"])
    plot = plot[plot["train_time_s"].astype(float) > 0]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(plot["train_time_s"].astype(float), plot["val_macro_f1@thr_f1"].astype(float))
    for _, r in plot.iterrows():
        ax.annotate(r["name"], (float(r["train_time_s"]), float(r["val_macro_f1@thr_f1"])),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Validation Macro-F1 (tuned threshold)")
    ax.set_title("Accuracy vs training cost")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fn = out / "f1_vs_train_time.png"
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    fig_dir = ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fn, fig_dir / fn.name)
    print("\n".join(md))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
