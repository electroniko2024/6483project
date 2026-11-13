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
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT_DIR, ROOT, evaluate, make_metrics  # noqa: E402

COLS = ["name", "model", "feature", "feature_dim", "sparse", "n_params",
        "val_acc@0.5", "val_macro_f1@0.5", "val_auc", "thr_acc", "val_acc@thr_acc",
        "thr_f1", "val_macro_f1@thr_f1", "seed_std_macro_f1", "train_time_s",
        "infer_time_s", "peak_mem_mb", "oov_rate", "hardware"]


def main():
    rows = []
    folders = sorted(f.parent for f in OUTPUT_DIR.glob("*/metrics.json"))
    seed_runs = {}  # base name -> [folders]; C stores seeds as X, X_seed43, X_seed44
    for d in folders:
        base = re.sub(r"_seed\d+$", "", d.name)
        seed_runs.setdefault(base, []).append(d)
    for base, ds in seed_runs.items():
        if len(ds) == 1:
            m = json.loads((ds[0] / "metrics.json").read_text(encoding="utf-8"))
            m["name"] = base
            # report time per seed for multi-seed folders (B stores the total)
            if "mean_single_seed_train_time_s" in m:
                m["train_time_s"] = m["mean_single_seed_train_time_s"]
                m["infer_time_s"] = m.get("mean_single_seed_infer_time_s", m.get("infer_time_s"))
            rows.append(m)
            continue
        # several single-seed folders: average their probabilities into one row
        ms = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in ds]
        vals = [pd.read_csv(d / "val_probs.csv").sort_values("idx") for d in ds]
        y = vals[0]["label"].values
        prob = sum(v["prob_pos"].values for v in vals) / len(vals)
        per_seed = [evaluate(y, v["prob_pos"].values, 0.5)["macro_f1"] for v in vals]
        m = make_metrics(
            ms[0]["model"], ms[0].get("feature", ""), y, prob,
            seeds=[s for mm in ms for s in mm.get("seeds", [])],
            seed_macro_f1s=per_seed,
            n_params=ms[0].get("n_params"), feature_dim=ms[0].get("feature_dim"),
            train_time_s=sum(mm.get("train_time_s", 0) for mm in ms) / len(ms),
            infer_time_s=sum(mm.get("infer_time_s", 0) for mm in ms) / len(ms),
            peak_mem_mb=max(mm.get("peak_mem_mb", 0) for mm in ms),
            hardware=ms[0].get("hardware"))
        m["name"] = f"{base} (avg of {len(ds)} seeds)"
        rows.append(m)
        print(f"{base}: averaged {[d.name for d in ds]}; per-seed val Macro-F1@0.5 "
              + ", ".join(f"{x:.4f}" for x in per_seed) + "; train_time_s is per seed")
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
