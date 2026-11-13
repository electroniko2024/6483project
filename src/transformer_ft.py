import argparse
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import clean_text, evaluate, get_split, load_data, make_metrics, save_outputs, set_seed


MODEL_IDS = {
    "distilbert": "distilbert/distilbert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Check the PyTorch environment first.")

    set_seed(args.seed)
    device = torch.device("cuda")
    train_df, test_df = load_data()
    train_idx, val_idx = get_split()
    train_rows = train_df.iloc[train_idx]
    val_rows = train_df.iloc[val_idx]

    if args.smoke:
        train_rows = train_rows.iloc[:32]
        test_df = test_df.iloc[:8]

    checkpoint = MODEL_IDS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2
    ).to(device)

    def collate(rows):
        texts, labels = zip(*rows)
        inputs = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt"
        )
        return inputs, torch.tensor(labels, dtype=torch.long)

    def loader(frame, labeled, shuffle=False):
        labels = frame["sentiments"].tolist() if labeled else [-1] * len(frame)
        texts = [clean_text(text, lower=False) for text in frame["reviews"]]
        return DataLoader(
            list(zip(texts, labels)), batch_size=args.batch_size,
            shuffle=shuffle, collate_fn=collate
        )

    @torch.inference_mode()
    def predict(batches):
        model.eval()
        probabilities = []
        for inputs, _ in batches:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**inputs.to(device)).logits
            probabilities.extend(logits.float().softmax(dim=1)[:, 1].tolist())
        return np.asarray(probabilities)

    training = loader(train_rows, labeled=True, shuffle=True)
    validation = loader(val_rows, labeled=True)
    testing = loader(test_df, labeled=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(training)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    train_time = 0.0
    best_f1 = -1.0

    print(f"Training {args.model} on {len(train_rows)} rows", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = perf_counter()
        for step, (inputs, labels) in enumerate(training, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs.to(device), labels=labels.to(device)).loss
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            if step == 1 or step % 50 == 0:
                print(f"epoch {epoch}, batch {step}/{len(training)}, loss={loss.item():.4f}",
                      flush=True)
        torch.cuda.synchronize()
        train_time += perf_counter() - started

        if args.smoke:
            print("SMOKE OK. Test positive probabilities:", predict(testing).round(4))
            return

        val_prob = predict(validation)
        scores = evaluate(val_rows["sentiments"], val_prob, 0.5)
        print(f"epoch {epoch}: accuracy={scores['acc']:.4f}, "
              f"macro-F1={scores['macro_f1']:.4f}", flush=True)
        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            best_epoch = epoch
            best_val_prob = val_prob
            started = perf_counter()
            best_test_prob = predict(testing)
            infer_time = perf_counter() - started

    labels = val_rows["sentiments"].to_numpy()
    metrics = make_metrics(
        args.model, "pretrained Transformer subword features", labels, best_val_prob,
        seeds=[args.seed], n_params=sum(p.numel() for p in model.parameters()),
        feature_dim=model.config.hidden_size, train_time_s=train_time,
        infer_time_s=infer_time, peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20,
        best_epoch=best_epoch,
    )
    val_output = pd.DataFrame({"idx": val_idx, "label": labels, "prob_pos": best_val_prob})
    test_output = pd.DataFrame({"idx": test_df.index, "prob_pos": best_test_prob})
    log = pd.DataFrame([{
        "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
        "max_length": args.max_length, "epochs": args.epochs,
        "best_epoch": best_epoch, "loss": "unweighted cross-entropy",
        "val_acc@0.5": metrics["val_acc@0.5"],
        "val_macro_f1@0.5": metrics["val_macro_f1@0.5"],
    }])
    print("Saved:", save_outputs(args.model, val_output, test_output, metrics, log))


if __name__ == "__main__":
    main()