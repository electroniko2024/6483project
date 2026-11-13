from pathlib import Path

import pandas as pd

from common import load_data


root = Path(__file__).resolve().parents[1]
outputs = root / "outputs"
train, test = load_data()

roberta_threshold = 0.680
svm_threshold = 0.625


def load_probs(folder):
    path = outputs / folder / "test_probs.csv"
    frame = pd.read_csv(path).set_index("idx", verify_integrity=True)
    prob = frame["prob_pos"].sort_index()
    if not prob.index.equals(test.index) or not prob.between(0, 1).all():
        raise ValueError(f"Invalid test predictions in {path}")
    return prob


p42 = load_probs("roberta")
p43 = load_probs("roberta_seed43")
p44 = load_probs("roberta_seed44")
roberta = (p42 + p43 + p44) / 3
svm = load_probs("tfidf_svm")

submission = pd.read_csv(root / "submission.csv")
final_labels = roberta.ge(roberta_threshold).astype(int)
if not submission["reviews"].equals(test["reviews"]):
    raise ValueError("submission.csv does not match the test-set order")
if not submission["predictions"].eq(final_labels).all():
    raise ValueError("RoBERTa labels do not match submission.csv")

pool = test.loc[~test["reviews"].isin(train["reviews"])].copy()
pool["distance"] = (roberta - roberta_threshold).abs()
pool["disagree"] = final_labels.ne(svm.ge(svm_threshold).astype(int))

near = pool.nsmallest(15, "distance")
pool = pool.drop(near.index)

disagree = pool.loc[pool["disagree"]].sample(15, random_state=42)
pool = pool.drop(disagree.index)

control = pool.sample(10, random_state=43)

sample = pd.concat([near, disagree, control])
sample = sample.sample(frac=1, random_state=44)

if len(sample) != 40 or not sample["reviews"].is_unique:
    raise ValueError("Expected 40 distinct reviews")

sheet = sample[["reviews"]].rename_axis("idx").reset_index()
sheet["manual_label"] = ""
sheet["note"] = ""
path = outputs / "test_for_annotation.csv"
sheet.to_csv(path, index=False, encoding="utf-8-sig")

print(f"Excluded train duplicates: {len(test) - len(test.loc[~test['reviews'].isin(train['reviews'])])}")
print(f"Saved {len(sheet)} reviews to {path}")