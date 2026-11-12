"""Train member B's GloVe models on the shared EE6483 split."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (DATA_DIR, ResourceMeter, clean_text, evaluate, get_split,
                    load_data, make_metrics, save_outputs, set_seed)

DIM = 300
MAX_LEN = 256
SEEDS = (42, 43, 44)
TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[!?]", re.UNICODE)


def tokenize(text):
    return TOKEN.findall(clean_text(text))[:MAX_LEN]


def load_glove(path, vocab):
    matrix = np.zeros((len(vocab), DIM), dtype=np.float32)
    covered = np.zeros(len(vocab), dtype=bool)
    with path.open(encoding="utf-8") as source:
        for line in source:
            word, _, values = line.partition(" ")
            idx = vocab.get(word)
            if idx is not None and idx >= 2:
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if len(vector) != DIM or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid {DIM}-dimensional GloVe vector for {word}")
                matrix[idx] = vector
                covered[idx] = True
    if not covered.any():
        raise ValueError("No training vocabulary entries matched the GloVe file")
    return matrix, covered


def oov_stats(sequences, covered):
    ids = np.fromiter((idx for row in sequences for idx in row), dtype=np.int64)
    return {
        "tokens": len(ids),
        "vocab_oov_rate": float((ids == 1).sum() / max(len(ids), 1)),
        "glove_uncovered_rate": float((~covered[ids]).sum() / max(len(ids), 1)),
    }


def mean_vectors(sequences, matrix, covered):
    result = np.zeros((len(sequences), DIM), dtype=np.float32)
    for i, row in enumerate(sequences):
        ids = np.asarray(row, dtype=np.int64)
        ids = ids[covered[ids]]
        if len(ids):
            result[i] = matrix[ids].mean(axis=0)
    return result


def export(name, label, feature, val_idx, y_val, val_prob, test_prob, log, **extra):
    metrics = make_metrics(
        label, feature, y_val, val_prob, sparse=False,
        oov_rate=extra["coverage"]["val"]["glove_uncovered_rate"], **extra)
    for key in ("acc", "f1"):
        threshold = metrics[f"thr_{key}"]
        metrics[f"at_thr_{key}"] = evaluate(y_val, val_prob, threshold)
    val = pd.DataFrame(dict(idx=val_idx, label=y_val, prob_pos=val_prob))
    test = pd.DataFrame(dict(idx=np.arange(len(test_prob)), prob_pos=test_prob))
    path = save_outputs(name, val, test, metrics, pd.DataFrame(log))
    print(f"{path}: Macro-F1={metrics['val_macro_f1@0.5']:.4f}", flush=True)


def run_lr(sequences, matrix, covered, y_train, val_idx, y_val, coverage):
    train, val, test = (mean_vectors(rows, matrix, covered) for rows in sequences)
    log = []
    best = None
    for c in (0.1, 1., 10.):
        for weight in (None, "balanced"):
            model = LogisticRegression(C=c, class_weight=weight, max_iter=1000)
            with ResourceMeter() as fit:
                model.fit(train, y_train)
            prob = model.predict_proba(val)[:, 1]
            row = dict(C=c, class_weight=weight or "none", **evaluate(y_val, prob))
            log.append(row)
            if best is None or row["macro_f1"] > best[0]["macro_f1"]:
                best = row, model, fit, prob
    config, model, fit, prob = best
    with ResourceMeter() as infer:
        test_prob = model.predict_proba(test)[:, 1]
    export("glove_lr", "Mean GloVe + Logistic Regression", "GloVe 300d mean (dense)",
           val_idx, y_val, prob, test_prob, log, feature_dim=DIM,
           n_params=model.coef_.size + model.intercept_.size,
           train_time_s=fit.seconds, infer_time_s=infer.seconds,
           peak_mem_mb=max(fit.peak_mb, infer.peak_mb), coverage=coverage,
           selected_config={k: config[k] for k in ("C", "class_weight")})


@dataclass(frozen=True)
class Config:
    lr: float = 0.001
    hidden: int = 128
    dropout: float = 0.3
    freeze: bool = False
    weighted: bool = False


def run_bilstm(sequences, matrix, covered, y_train, val_idx, y_val, coverage, args):
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BiLSTM device: {device}", flush=True)

    class BiLSTM(nn.Module):
        def __init__(self, config):
            super().__init__()
            weights = np.random.normal(0, .05, matrix.shape).astype(np.float32)
            weights[covered] = matrix[covered]
            weights[0] = 0
            self.embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(weights), freeze=config.freeze, padding_idx=0)
            self.layers = nn.ModuleList([
                nn.ModuleList([nn.LSTM(width, config.hidden, batch_first=True)
                               for _ in range(2)])
                for width in (DIM, 2 * config.hidden)
            ])
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(4 * config.hidden, 2)

        def forward(self, ids, lengths):
            lengths = lengths.to(ids.device)
            position = torch.arange(ids.size(1), device=ids.device)[None, :]
            mask = position < lengths[:, None]
            # Reverse each valid prefix before the backward LSTM. Both directions
            # then see real tokens before padding, including in the second layer.
            reverse = torch.where(mask, lengths[:, None] - 1 - position, position)

            def reverse_prefix(values):
                return values.gather(1, reverse[..., None].expand_as(values))

            values = self.embedding(ids)
            for forward, backward in self.layers:
                left, _ = forward(values)
                right, _ = backward(reverse_prefix(values))
                values = torch.cat([left, reverse_prefix(right)], dim=2)
            maximum = values.masked_fill(~mask[..., None], -torch.inf).amax(1)
            mean = (values * mask[..., None]).sum(1) / lengths[:, None]
            return self.output(self.dropout(torch.cat([maximum, mean], dim=1)))

    def collate(batch):
        ids, labels = zip(*batch)
        return (pad_sequence(ids, batch_first=True),
                torch.tensor([len(row) for row in ids]), torch.tensor(labels))

    def dataset(rows, labels):
        return [(torch.tensor(row or [1], dtype=torch.long), int(label))
                for row, label in zip(rows, labels)]

    train, val, test = sequences
    training = dataset(train, y_train)
    validation = DataLoader(dataset(val, y_val), batch_size=64, collate_fn=collate)
    testing = DataLoader(dataset(test, np.zeros(len(test))), batch_size=64, collate_fn=collate)
    weights = torch.tensor(len(y_train) / (2 * np.bincount(y_train)),
                           dtype=torch.float32, device=device)

    def predict(model, loader):
        model.eval()
        with torch.inference_mode():
            return np.concatenate([
                model(ids.to(device), lengths).softmax(1)[:, 1].cpu().numpy()
                for ids, lengths, _ in loader])

    log = []

    def fit(config, seed, phase, include_test=False):
        set_seed(seed)
        # Shuffle within coarse length buckets so each batch pads less.
        by_length = sorted(range(len(training)), key=lambda i: len(training[i][0]))
        pools = [by_length[i:i + 512] for i in range(0, len(by_length), 512)]
        rng = np.random.default_rng(seed)
        model = BiLSTM(config).to(device)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss(weight=weights if config.weighted else None)
        best_score, stale = -1., 0
        best_state = None
        with ResourceMeter(gpu=device.type == "cuda") as train_meter:
            for epoch in range(1, args.epochs + 1):
                model.train()
                batches = []
                for pool in pools:
                    order = rng.permutation(pool)
                    batches.extend(order[i:i + 64] for i in range(0, len(order), 64))
                rng.shuffle(batches)
                train_loader = DataLoader(training, batch_sampler=batches, collate_fn=collate)
                for ids, lengths, labels in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(ids.to(device), lengths), labels.to(device))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                prob = predict(model, validation)
                metrics = evaluate(y_val, prob)
                score = metrics["macro_f1"]
                print(f"{phase} seed={seed} epoch={epoch} f1={score:.4f}", flush=True)
                if score > best_score:
                    best_score, stale = score, 0
                    best_prob, best_metrics, best_epoch = prob, metrics, epoch
                    if include_test:
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale == 2:
                        break
        row = dict(phase=phase, seed=seed, **asdict(config), epoch=best_epoch,
                   epochs_run=epoch, train_time_s=train_meter.seconds, **best_metrics)
        log.append(row)
        if not include_test:
            return best_score
        model.load_state_dict(best_state)
        with ResourceMeter(gpu=device.type == "cuda") as infer_meter:
            test_prob = predict(model, testing)
        return (best_prob, test_prob, best_score, train_meter.seconds,
                infer_meter.seconds, max(train_meter.peak_mb, infer_meter.peak_mb),
                params, trainable)

    config = Config()
    if not args.default_config:
        best_score = fit(config, SEEDS[0], "initial")
        for field, choices in (("lr", (.001, .0005)), ("hidden", (64, 128, 256)),
                               ("dropout", (.3, .5)), ("freeze", (False, True)),
                               ("weighted", (False, True))):
            incumbent = config
            for value in choices:
                trial = replace(incumbent, **{field: value})
                if trial != incumbent:
                    score = fit(trial, SEEDS[0], field)
                    if score > best_score:
                        config, best_score = trial, score

    runs = [fit(config, seed, "final", include_test=True) for seed in SEEDS]
    val_p, test_p, f1, times, infers, peaks, params, trainable = zip(*runs)
    export("bilstm", "GloVe + BiLSTM", "GloVe 300d sequence (dense)",
           val_idx, y_val, np.mean(val_p, axis=0), np.mean(test_p, axis=0), log,
           seeds=SEEDS, seed_macro_f1s=f1, seed_mean_macro_f1=float(np.mean(f1)),
           per_seed_macro_f1=list(f1), feature_dim=DIM, coverage=coverage,
           n_params=params[0], trainable_params=trainable[0], device=str(device),
           train_time_s=sum(times), infer_time_s=sum(infers), peak_mem_mb=max(peaks),
           mean_single_seed_train_time_s=float(np.mean(times)),
           mean_single_seed_infer_time_s=float(np.mean(infers)),
           train_time_note="sum across final seeds; excludes tuning and preprocessing",
           selected_config=asdict(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove", type=Path, default=DATA_DIR / "glove.6B.300d.txt")
    parser.add_argument("--model", choices=("all", "lr", "bilstm"), default="all")
    parser.add_argument("--epochs", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--default-config", action="store_true",
                        help="Skip sequential tuning and run the default BiLSTM on all three seeds.")
    args = parser.parse_args()
    if not args.glove.is_file():
        parser.error(f"GloVe file missing: {args.glove}")
    set_seed()
    train, test = load_data()
    tr_idx, val_idx = get_split()
    y = train.sentiments.to_numpy(dtype=np.int64)
    train_tokens = [tokenize(text) for text in train.reviews.iloc[tr_idx]]
    counts = Counter(token for row in train_tokens for token in row)
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({word: i for i, word in enumerate(
        sorted(word for word, count in counts.items() if count >= 2), start=2)})
    matrix, covered = load_glove(args.glove, vocab)
    val_tokens = [tokenize(text) for text in train.reviews.iloc[val_idx]]
    test_tokens = [tokenize(text) for text in test.reviews]
    sequences = [
        [[vocab.get(token, 1) for token in row] for row in tokens]
        for tokens in (train_tokens, val_tokens, test_tokens)
    ]
    coverage = {"vocab_size": len(vocab), "covered_types": int(covered.sum()),
                **{name: oov_stats(rows, covered)
                   for name, rows in zip(("train", "val", "test"), sequences)}}
    print(f"Coverage: {coverage}", flush=True)
    if args.model in ("all", "lr"):
        run_lr(sequences, matrix, covered, y[tr_idx], val_idx, y[val_idx], coverage)
    if args.model in ("all", "bilstm"):
        run_bilstm(sequences, matrix, covered, y[tr_idx], val_idx, y[val_idx],
                   coverage, args)


if __name__ == "__main__":
    main()
