# EE6483 Mini Project (Option 1): Review Sentiment Classification

Deadline: 2026-11-13 23:59 (NTULearn). Full plan: [项目实施方案.md](项目实施方案.md).

## Layout
```text
data/        train.json test.json (in git, only ~3 MB)
splits/      val_idx.npy                 shared validation indices (do not regenerate)
src/         common.py, tfidf_models.py, noise_exp.py, resource_table.py (A)
             glove_bilstm.py (B)
             transformer_ft.py, ensemble.py (C)
outputs/     <model_name>/ val_probs.csv test_probs.csv metrics.json hparam_log.csv
             noise_exp/  results + figures for (i)
             resource/   resource-vs-accuracy table + figure for (g)
report/      figures/ refs.bib
docs/        assignment PDF and example notebook
submission.csv
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```
GloVe (member B) is not in git: download `glove.6B.zip` from
https://nlp.stanford.edu/projects/glove/ and put `glove.6B.300d.txt` in `data/`.

## Shared conventions (see plan section 2)
- Always split with `common.get_split()` (train part 5909 / validation 1492,
  grouped by normalized text so duplicates never cross the split) and
  `common.group_cv_folds()` for CV inside the train part. Never re-split.
- Write outputs with `common.save_outputs()` and build metrics with
  `common.make_metrics()`; write the submission only with
  `common.write_submission()`.
- Global seed 42; neural models run seeds 42, 43, 44 and report mean ± std.

Quick check that the environment and split are fine:
```bash
python src/common.py
```
Expected: `train part 5909, val 1492, val negatives 218, group overlap 0`.

## Run instructions
All commands are run from the project root.

| Step | Command | Output | Runtime / hardware |
|---|---|---|---|
| TF-IDF baselines (grid search with grouped 5-fold CV) | `python src/tfidf_models.py` | `outputs/tfidf_lr/`, `outputs/tfidf_svm/`, `outputs/tfidf_cnb/`, baseline `submission.csv` | ~6 min (10 parallel jobs); `--quick` <1 min |
| Noise-label experiment, question (i) | `python src/noise_exp.py` | `outputs/noise_exp/` | a few minutes |
| GloVe + BiLSTM (B) | `python src/glove_bilstm.py` | `outputs/glove_lr/`, `outputs/bilstm/` | CPU or GPU |
| RoBERTa fine-tuning (C) | See the commands below | `outputs/roberta/` (seed 42), `outputs/roberta_seed43/`, `outputs/roberta_seed44/` | CUDA GPU |
| Generate final predictions (C) | `python src/ensemble.py` | `submission_candidate.csv` (copied to `submission.csv` for the final submission) | seconds |
| Resource vs accuracy, question (g) | `python src/resource_table.py` | `outputs/resource/` | seconds |
| TF-IDF dimension study, question (g) | `python src/tfidf_dim_study.py` | `outputs/resource/tfidf_dim_study.*` | ~1 min |

`python src/tfidf_models.py --quick` runs a reduced grid for a fast check.
For member B, run `python src/glove_bilstm.py --model lr` for the lightweight baseline or
`python src/glove_bilstm.py --model bilstm` for BiLSTM alone. The full run tunes
BiLSTM sequentially and then averages probabilities from seeds 42, 43, and 44.
It selects CUDA when available. For a shorter run use `--default-config` to
skip tuning, and `--epochs` to cap training. Both models use the shared
5,909-row training portion for fits and export positive probabilities
in original row order. Coverage
statistics are recorded in each model's `metrics.json`.
The archived B runs used Python 3.12, PyTorch 2.6.0 CPU, NumPy 2.3.5,
and scikit-learn 1.8.0. Timing excludes vocabulary and embedding preparation.

### GloVe and BiLSTM

Run `python src/glove_bilstm.py` from the project root after placing
`glove.6B.300d.txt` in `data/`. The script is on `member-b` and is
available in the project root after that branch is integrated.

### RoBERTa and final predictions

To reproduce the three training runs, activate a CUDA-enabled Python
environment and run the following from the project root in PowerShell:

```powershell
foreach ($seed in 42, 43, 44) {
    python src/transformer_ft.py --model roberta --epochs 1 --batch-size 4 --max-length 256 --lr 2e-5 --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed" }
}
```

The archived seed 42 results are in `outputs/roberta/`. The current
training script writes a new seed 42 run to `outputs/roberta_seed42/`,
while `ensemble.py` reads `outputs/roberta/`. If you retrain, copy the
new seed 42 outputs into the directory used by `ensemble.py`:

```powershell
New-Item -ItemType Directory -Force outputs/roberta | Out-Null
Copy-Item outputs/roberta_seed42/* outputs/roberta/ -Force
```

With the archived predictions, run `python src/ensemble.py` directly.
It averages the three positive-class probabilities and applies the
validation threshold of 0.680. Check that `submission_candidate.csv`
has 1,851 rows with columns `reviews,predictions`. To replace the
official submission after checking the candidate, run
`Copy-Item submission_candidate.csv submission.csv -Force`.

## Report

Build `report/main.tex` with XeLaTeX. The report uses Times New Roman.
From the `report/` directory, run:

```bash
xelatex -halt-on-error main.tex
bibtex main
xelatex -halt-on-error main.tex
xelatex -halt-on-error main.tex
```

Open `report/main.pdf` to check the compiled report.
