import argparse
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import clean_text, evaluate, get_split, load_data, make_metrics, save_outputs, set_seed


MODELS = {
    "distilbert": "distilbert/distilbert-base-uncased",
    "roberta": "FacebookAI/roberta-base",
}


def encode(tokenizer, texts, max_length):
    return tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to("cuda")


def predict(model, tokenizer, texts, batch_size, max_length):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = encode(tokenizer, batch, max_length)
            logits = model(**inputs).logits
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.array(probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 4 or args.lr <= 0:
        parser.error("Use positive epochs, batch size and learning rate, and max length >= 4.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Use the project's CUDA Python environment.")

    set_seed(args.seed)
    train_data, test_data = load_data()
    train_idx, val_idx = get_split()
    if args.smoke:
        train_idx = train_idx[:32]
        test_data = test_data.iloc[:8]

    train = train_data.iloc[train_idx]
    val = train_data.iloc[val_idx]
    train_texts = [clean_text(text, lower=False) for text in train["reviews"]]
    val_texts = [clean_text(text, lower=False) for text in val["reviews"]]
    test_texts = [clean_text(text, lower=False) for text in test_data["reviews"]]
    train_labels = train["sentiments"].tolist()
    val_labels = val["sentiments"].to_numpy()

    checkpoint = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    torch.cuda.reset_peak_memory_stats()
    train_time = 0.0
    best_f1 = -1.0
    print(f"Training {args.model} on {len(train_texts)} rows", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_texts)).tolist()
        started = perf_counter()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            texts = [train_texts[i] for i in indices]
            labels = torch.tensor([train_labels[i] for i in indices], device="cuda")
            inputs = encode(tokenizer, texts, args.max_length)

            optimizer.zero_grad()
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()

            step = start // args.batch_size + 1
            if step == 1 or step % 50 == 0:
                print(f"Epoch {epoch}, batch {step}, loss={loss.item():.4f}", flush=True)

        torch.cuda.synchronize()
        train_time += perf_counter() - started

        if args.smoke:
            probabilities = predict(model, tokenizer, test_texts, args.batch_size, args.max_length)
            print("SMOKE OK. Test positive probabilities:", probabilities.round(4))
            return

        val_prob = predict(model, tokenizer, val_texts, args.batch_size, args.max_length)
        scores = evaluate(val_labels, val_prob, 0.5)
        f1 = scores["macro_f1"]
        print(f"Epoch {epoch}: accuracy={scores['acc']:.4f}, macro-F1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_val_prob = val_prob
            started = perf_counter()
            best_test_prob = predict(model, tokenizer, test_texts, args.batch_size, args.max_length)
            infer_time = perf_counter() - started

    metrics = make_metrics(
        args.model, "pretrained Transformer subword features", val_labels, best_val_prob,
        seeds=[args.seed], n_params=sum(p.numel() for p in model.parameters()),
        feature_dim=model.config.hidden_size, train_time_s=train_time,
        infer_time_s=infer_time, peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20,
        best_epoch=best_epoch, precision="fp32", lr_schedule="constant",
    )
    val_output = pd.DataFrame({"idx": val_idx, "label": val_labels, "prob_pos": best_val_prob})
    test_output = pd.DataFrame({"idx": test_data.index, "prob_pos": best_test_prob})
    log = pd.DataFrame([{
        "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
        "max_length": args.max_length, "epochs": args.epochs, "best_epoch": best_epoch,
        "loss": "unweighted cross-entropy", "precision": "fp32", "lr_schedule": "constant",
        "val_acc@0.5": metrics["val_acc@0.5"], "val_macro_f1@0.5": metrics["val_macro_f1@0.5"],
    }])
    out = save_outputs(args.model, val_output, test_output, metrics, log)
    print("Saved:", out)


if __name__ == "__main__":
    main()