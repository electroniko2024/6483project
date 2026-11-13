from pathlib import Path

import pandas as pd

from common import evaluate, load_data, write_submission


root = Path(__file__).resolve().parents[1]
folders = [
    "roberta_seed42_warmup",
    "roberta_seed43_warmup",
    "roberta_seed44_warmup",
]
threshold = 0.590


def read_predictions(filename):
    frames = []
    for folder in folders:
        path = root / "outputs" / folder / filename
        frame = pd.read_csv(path).set_index("idx", verify_integrity=True)
        frame = frame.sort_index()
        if not frame["prob_pos"].between(0, 1).all():
            raise ValueError(f"Invalid probabilities in {path}")
        frames.append(frame)
    return frames


val = read_predictions("val_probs.csv")
test = read_predictions("test_probs.csv")
_, test_data = load_data()

for frame in val[1:]:
    if not val[0]["label"].equals(frame["label"]):
        raise ValueError("Validation IDs or labels do not match")

for frame in test:
    if not frame.index.equals(test_data.index):
        raise ValueError("Test IDs do not match test.json")

val_prob = pd.concat([frame["prob_pos"] for frame in val], axis=1).mean(axis=1)
test_prob = pd.concat([frame["prob_pos"] for frame in test], axis=1).mean(axis=1)

scores = evaluate(val[0]["label"], val_prob, threshold)
print(f"Validation accuracy: {scores['acc']:.2%}")
print(f"Validation macro-F1: {scores['macro_f1']:.2%}")

submission = write_submission(
    test_prob, threshold, path=root / "submission_candidate.csv"
)
print("Positive:", int((submission["predictions"] == 1).sum()))
print("Negative:", int((submission["predictions"] == 0).sum()))