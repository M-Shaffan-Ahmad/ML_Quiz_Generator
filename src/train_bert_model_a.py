import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd


OPTION_COLUMNS = ["A", "B", "C", "D"]


def require_bert_dependencies():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "BERT training requires optional dependencies. Install them with "
            "`pip install -r bts/requirements-bert.txt`."
        ) from exc
    return torch, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def to_long(df):
    rows = []
    for _, row in df.iterrows():
        for option_letter in OPTION_COLUMNS:
            rows.append(
                {
                    "article": row["article"],
                    "question_option": f"{row['question']} [SEP] {row[option_letter]}",
                    "label": int(row["answer"] == option_letter),
                }
            )
    return pd.DataFrame(rows)


class RaceOptionDataset:
    def __init__(self, frame, tokenizer, max_length):
        self.labels = frame["label"].astype(int).tolist()
        self.encodings = tokenizer(
            frame["article"].fillna("").astype(str).tolist(),
            frame["question_option"].fillna("").astype(str).tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_split(path, max_questions=None, random_state=42):
    df = pd.read_csv(path)
    if max_questions is not None and len(df) > max_questions:
        df = df.sample(n=max_questions, random_state=random_state).reset_index(drop=True)
    return to_long(df)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune optional BERT Model A answer verifier.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="models/model_a/neural/bert")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--max-dev-questions", type=int, default=2000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use mixed precision; defaults to enabled when CUDA is available.",
    )
    args = parser.parse_args()

    torch, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments = require_bert_dependencies()
    use_cuda = torch.cuda.is_available()
    fp16 = use_cuda if args.fp16 is None else args.fp16
    if fp16 and not use_cuda:
        raise RuntimeError("--fp16 was requested, but CUDA is not available.")

    device_label = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Training device: {device_label}")
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    print("Loading RACE splits...")
    train_long = load_split(args.train_csv, args.max_train_questions)
    dev_long = load_split(args.dev_csv, args.max_dev_questions)

    print(f"Train option rows: {len(train_long)}")
    print(f"Dev option rows:   {len(dev_long)}")

    train_dataset = RaceOptionDataset(train_long, tokenizer, args.max_length)
    dev_dataset = RaceOptionDataset(dev_long, tokenizer, args.max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == labels).mean())
        return {"accuracy": accuracy}

    training_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "fp16": fp16,
        "save_total_limit": 2,
        "report_to": [],
    }
    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved BERT Model A checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
