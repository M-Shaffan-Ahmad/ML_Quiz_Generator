import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd


OPTION_COLUMNS = ["A", "B", "C", "D"]
ANSWER_TO_INDEX = {letter: index for index, letter in enumerate(OPTION_COLUMNS)}


def require_bert_dependencies():
    try:
        import torch
        from transformers import (
            AutoModelForMultipleChoice,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "BERT multiple-choice training requires optional dependencies. Install them with "
            "`pip install -r bts/requirements-bert.txt`."
        ) from exc
    return torch, AutoTokenizer, AutoModelForMultipleChoice, EarlyStoppingCallback, Trainer, TrainingArguments


class RaceMultipleChoiceDataset:
    def __init__(self, frame, tokenizer, max_length):
        self.labels = frame["answer"].map(ANSWER_TO_INDEX).astype(int).tolist()

        first_sentences = []
        second_sentences = []
        for _, row in frame.iterrows():
            article = str(row["article"])
            for option_letter in OPTION_COLUMNS:
                first_sentences.append(article)
                second_sentences.append(f"{row['question']} [SEP] {row[option_letter]}")

        encoded = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.encodings = {
            key: [
                value[index : index + len(OPTION_COLUMNS)]
                for index in range(0, len(value), len(OPTION_COLUMNS))
            ]
            for key, value in encoded.items()
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch

        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_split(
    path,
    max_questions=None,
    random_state=42,
    exclude_sample_size=0,
    exclude_random_state=None,
):
    frame = pd.read_csv(path).reset_index(drop=True)
    if exclude_sample_size:
        if exclude_sample_size >= len(frame):
            raise ValueError(
                f"Cannot exclude {exclude_sample_size} rows from {path}; "
                f"the split only has {len(frame)} rows."
            )
        previous_seed = random_state if exclude_random_state is None else exclude_random_state
        previous_indices = frame.sample(
            n=exclude_sample_size,
            random_state=previous_seed,
        ).index
        frame = frame.drop(index=previous_indices).reset_index(drop=True)

    if max_questions is not None and len(frame) > max_questions:
        frame = frame.sample(n=max_questions, random_state=random_state).reset_index(drop=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT Model A as a 4-way multiple-choice model.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--model-name", default="models/model_a/neural/bert")
    parser.add_argument("--output-dir", default="models/model_a/neural/bert_mc")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--max-dev-questions", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--exclude-train-sample-size",
        type=int,
        default=0,
        help=(
            "Exclude a previous random train sample before selecting the current train sample. "
            "Use this to continue training on fresh examples after an earlier max-train run."
        ),
    )
    parser.add_argument(
        "--exclude-train-random-state",
        type=int,
        default=None,
        help=(
            "Random seed used by the previous train sample being excluded; defaults to "
            "--random-state."
        ),
    )
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use mixed precision; defaults to enabled when CUDA is available.",
    )
    args = parser.parse_args()

    (
        torch,
        AutoTokenizer,
        AutoModelForMultipleChoice,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    ) = require_bert_dependencies()
    use_cuda = torch.cuda.is_available()
    fp16 = use_cuda if args.fp16 is None else args.fp16
    if fp16 and not use_cuda:
        raise RuntimeError("--fp16 was requested, but CUDA is not available.")

    device_label = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Training device: {device_label}")
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and multiple-choice model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name,
        ignore_mismatched_sizes=True,
    )

    print("Loading RACE splits...")
    train_frame = load_split(
        args.train_csv,
        args.max_train_questions,
        args.random_state,
        exclude_sample_size=args.exclude_train_sample_size,
        exclude_random_state=args.exclude_train_random_state,
    )
    dev_frame = load_split(args.dev_csv, args.max_dev_questions, args.random_state)
    if args.exclude_train_sample_size:
        previous_seed = (
            args.random_state
            if args.exclude_train_random_state is None
            else args.exclude_train_random_state
        )
        print(
            "Excluded previous train sample: "
            f"{args.exclude_train_sample_size} questions sampled with random_state={previous_seed}"
        )
    print(f"Train questions: {len(train_frame)}")
    print(f"Dev questions:   {len(dev_frame)}")

    train_dataset = RaceMultipleChoiceDataset(train_frame, tokenizer, args.max_length)
    dev_dataset = RaceMultipleChoiceDataset(dev_frame, tokenizer, args.max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": float((predictions == labels).mean())}

    strategy = "steps" if args.eval_steps else "epoch"
    training_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": strategy,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "fp16": fp16,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "report_to": [],
    }
    if args.eval_steps:
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs["save_steps"] = args.eval_steps
    if args.label_smoothing:
        training_kwargs["label_smoothing_factor"] = args.label_smoothing

    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = strategy
    else:
        training_kwargs["evaluation_strategy"] = strategy

    trainer_kwargs = {
        "model": model,
        "args": TrainingArguments(**training_kwargs),
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    if args.early_stopping_patience > 0:
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ]

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved BERT multiple-choice checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
