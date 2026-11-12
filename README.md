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

| Step | Command | Output | Time (CPU) |
|---|---|---|---|
| TF-IDF baselines (grid search with grouped 5-fold CV) | `python src/tfidf_models.py` | `outputs/tfidf_lr/`, `outputs/tfidf_svm/`, `outputs/tfidf_cnb/`, baseline `submission.csv` | ~6 min (10 parallel jobs); `--quick` <1 min |
| Noise-label experiment, question (i) | `python src/noise_exp.py` | `outputs/noise_exp/` | a few minutes |
| GloVe + BiLSTM (B) | TODO | `outputs/glove_lr/`, `outputs/bilstm/` | |
| RoBERTa fine-tuning (C) | TODO | `outputs/roberta/` | GPU |
| Ensemble + final submission (C) | TODO | `submission.csv` | |
| Resource vs accuracy, question (g) | `python src/resource_table.py` | `outputs/resource/` | seconds |

`python src/tfidf_models.py --quick` runs a reduced grid for a fast check.
