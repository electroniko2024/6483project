# EE6483 Mini Project (Option 1): Review Sentiment Classification

Deadline: 2026-11-13 23:59 (NTULearn). Full plan: [项目实施方案.md](项目实施方案.md).

## Layout
```text
data/        train.json test.json (in git, only ~3 MB)
splits/      val_idx.npy                 shared validation indices (do not regenerate)
src/         common.py, tfidf_models.py, noise_exp.py, resource_table.py (A)
             glove_bilstm.py (B)
             transformer_ft.py, transformer_ft_warmup.py, ensemble.py (C)
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
| RoBERTa fine-tuning (C) | See the commands below | `outputs/roberta_seed42_warmup/`, `outputs/roberta_seed43_warmup/`, `outputs/roberta_seed44_warmup/` | CUDA GPU |
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

The final submission uses the average positive-class probabilities
from three RoBERTa-base runs with seeds 42, 43, and 44. Each run used
one epoch, 10% learning-rate warmup followed by linear decay, and a
peak learning rate of `2e-5`.

To reproduce the training runs, activate a CUDA-enabled Python
environment and run the following in PowerShell from the project root:

```powershell
foreach ($seed in 42, 43, 44) {
    python src/transformer_ft_warmup.py --model roberta --epochs 1 --batch-size 4 --max-length 256 --lr 2e-5 --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed" }
}
```

To regenerate predictions from the saved warmup outputs, run:

```powershell
python src/ensemble.py
```

The script averages the three sets of probabilities and applies the
validation-selected threshold of `0.590`. It writes
`submission_candidate.csv` with 1,851 rows and the columns
`reviews,predictions`. The saved runs yield 1,568 positive and 283
negative predictions.

To update the final submission after checking the candidate, run:

```powershell
Copy-Item -LiteralPath .\submission_candidate.csv -Destination .\submission.csv -Force
```

The TF-IDF baseline script also writes `submission.csv`. If you run it,
regenerate the RoBERTa submission afterward.

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
