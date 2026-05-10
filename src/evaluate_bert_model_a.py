import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score


OPTION_COLUMNS = ["A", "B", "C", "D"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


class NoWordNet:
    """Keep METEOR offline by skipping WordNet synonym matching."""

    def synsets(self, *args, **kwargs):
        return []


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def lcs_length(left_tokens, right_tokens):
    previous = [0] * (len(right_tokens) + 1)
    for left_token in left_tokens:
        current = [0]
        for index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    overlap = lcs_length(reference_tokens, hypothesis_tokens)
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def compute_text_metrics(reference_texts, hypothesis_texts):
    references = [tokenize(text) for text in reference_texts]
    hypotheses = [tokenize(text) for text in hypothesis_texts]
    bleu = corpus_bleu(
        [[reference] for reference in references],
        hypotheses,
        smoothing_function=SmoothingFunction().method1,
    )
    rouge_l = sum(
        rouge_l_f1(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ) / max(len(references), 1)
    meteor = sum(
        meteor_score([reference], hypothesis, wordnet=NoWordNet())
        for reference, hypothesis in zip(references, hypotheses)
    ) / max(len(references), 1)
    return {"BLEU": bleu, "ROUGE-L": rouge_l, "METEOR": meteor}


def to_long(wide_df):
    id_vars = [
        column
        for column in [
            "qid",
            "example_id",
            "question_id",
            "split",
            "level",
            "article",
            "question",
            "answer",
            "correct_answer_text",
        ]
        if column in wide_df.columns
    ]
    long_df = wide_df.melt(
        id_vars=id_vars,
        value_vars=OPTION_COLUMNS,
        var_name="option_letter",
        value_name="option",
    )
    long_df["question_option"] = (
        long_df["question"].fillna("").astype(str)
        + " [SEP] "
        + long_df["option"].fillna("").astype(str)
    )
    return long_df


def choose_predictions(wide_df, long_df, option_scores):
    ranked = long_df[["qid", "option_letter"]].copy()
    ranked["score"] = option_scores
    ranked = ranked.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates("qid")
    wide_by_qid = wide_df.set_index("qid")

    rows = []
    for qid, predicted_letter, score in zip(
        ranked["qid"], ranked["option_letter"], ranked["score"]
    ):
        true_letter = wide_by_qid.at[qid, "answer"]
        rows.append(
            {
                "qid": qid,
                "question_id": wide_by_qid.at[qid, "question_id"],
                "level": wide_by_qid.at[qid, "level"],
                "true_letter": true_letter,
                "predicted_letter": predicted_letter,
                "score": float(score),
                "true_answer_text": wide_by_qid.at[qid, true_letter],
                "predicted_answer_text": wide_by_qid.at[qid, predicted_letter],
            }
        )
    return pd.DataFrame(rows)


def resolve_device(torch, requested_device):
    requested = str(requested_device or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return requested


def score_options(model, tokenizer, torch, device, long_df, batch_size, max_length):
    scores = []
    autocast_enabled = str(device).startswith("cuda")
    for start in range(0, len(long_df), batch_size):
        batch = long_df.iloc[start : start + batch_size]
        encoded = tokenizer(
            batch["article"].fillna("").astype(str).tolist(),
            batch["question_option"].fillna("").astype(str).tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad(), torch.autocast(
            device_type="cuda", enabled=autocast_enabled, dtype=torch.float16
        ):
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(probabilities.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


def evaluate_split(args, model, tokenizer, torch, device, split_name, csv_path):
    wide_df = pd.read_csv(csv_path).reset_index(drop=True)
    if args.max_questions is not None and len(wide_df) > args.max_questions:
        wide_df = wide_df.sample(n=args.max_questions, random_state=args.random_state).reset_index(drop=True)
    wide_df["qid"] = np.arange(len(wide_df))
    long_df = to_long(wide_df)

    print(f"Scoring {split_name}: {len(wide_df)} questions / {len(long_df)} option rows", flush=True)
    scores = score_options(model, tokenizer, torch, device, long_df, args.batch_size, args.max_length)
    predictions = choose_predictions(wide_df, long_df, scores)
    metrics = compute_text_metrics(
        predictions["true_answer_text"],
        predictions["predicted_answer_text"],
    )
    metrics["Exact Match Diagnostic"] = (
        predictions["true_letter"] == predictions["predicted_letter"]
    ).mean()

    prediction_path = f"{args.output_prefix}_predictions_{split_name}.csv"
    predictions.to_csv(prediction_path, index=False)
    return {
        "model": "BERT Model A",
        "checkpoint": args.checkpoint,
        "split": split_name,
        "questions": len(wide_df),
        "device": device,
        **metrics,
        "predictions_file": prediction_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT Model A with answer-text metrics.")
    parser.add_argument("--checkpoint", default="models/model_a/neural/bert")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default="bts/results/bert_model_a")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install BERT dependencies with pip install -r bts/requirements-bert.txt") from exc

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"BERT checkpoint not found: {checkpoint}")

    device = resolve_device(torch, args.device)
    print(f"Evaluation device: {device}", flush=True)
    if str(device).startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    split_paths = {"dev": args.dev_csv, "test": args.test_csv}
    results = [
        evaluate_split(args, model, tokenizer, torch, device, split_name, split_paths[split_name])
        for split_name in args.splits
    ]

    results_df = pd.DataFrame(results)
    results_path = f"{args.output_prefix}_eval_results.csv"
    results_df.to_csv(results_path, index=False)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(json.dumps({"results_file": results_path}, indent=2), flush=True)


if __name__ == "__main__":
    main()
