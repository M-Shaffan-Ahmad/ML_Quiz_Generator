import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score


OPTION_COLUMNS = ["A", "B", "C", "D"]
INDEX_TO_ANSWER = {index: letter for index, letter in enumerate(OPTION_COLUMNS)}
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


def resolve_device(torch, requested_device):
    requested = str(requested_device or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return requested


def encode_choice_batch(tokenizer, rows, max_length):
    first_sentences = []
    second_sentences = []
    for _, row in rows.iterrows():
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
        return_tensors="pt",
    )
    return {
        key: value.view(len(rows), len(OPTION_COLUMNS), -1)
        for key, value in encoded.items()
    }


def predict_split(model, tokenizer, torch, device, wide_df, batch_size, max_length):
    predicted_letters = []
    predicted_scores = []
    autocast_enabled = str(device).startswith("cuda")

    for start in range(0, len(wide_df), batch_size):
        batch = wide_df.iloc[start : start + batch_size]
        encoded = encode_choice_batch(tokenizer, batch, max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad(), torch.autocast(
            device_type="cuda", enabled=autocast_enabled, dtype=torch.float16
        ):
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
        winners = torch.argmax(probabilities, dim=-1).detach().cpu().numpy()
        scores = probabilities.max(dim=-1).values.detach().cpu().numpy()
        predicted_letters.extend(INDEX_TO_ANSWER[int(index)] for index in winners)
        predicted_scores.extend(float(score) for score in scores)

    return predicted_letters, predicted_scores


def evaluate_split(args, model, tokenizer, torch, device, split_name, csv_path):
    wide_df = pd.read_csv(csv_path).reset_index(drop=True)
    if args.max_questions is not None and len(wide_df) > args.max_questions:
        wide_df = wide_df.sample(n=args.max_questions, random_state=args.random_state).reset_index(drop=True)
    print(f"Scoring {split_name}: {len(wide_df)} questions", flush=True)

    predicted_letters, predicted_scores = predict_split(
        model, tokenizer, torch, device, wide_df, args.batch_size, args.max_length
    )
    rows = []
    for index, row in wide_df.iterrows():
        true_letter = row["answer"]
        predicted_letter = predicted_letters[index]
        rows.append(
            {
                "qid": index,
                "question_id": row["question_id"],
                "level": row["level"],
                "true_letter": true_letter,
                "predicted_letter": predicted_letter,
                "score": predicted_scores[index],
                "true_answer_text": row[true_letter],
                "predicted_answer_text": row[predicted_letter],
            }
        )

    predictions = pd.DataFrame(rows)
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
        "model": "BERT Multiple Choice Model A",
        "checkpoint": args.checkpoint,
        "split": split_name,
        "questions": len(wide_df),
        "device": device,
        **metrics,
        "predictions_file": prediction_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT multiple-choice Model A.")
    parser.add_argument("--checkpoint", default="models/model_a/neural/bert_mc")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--splits", nargs="+", choices=["dev", "test"], default=["dev", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default="bts/results/bert_mc_model_a")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForMultipleChoice, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install BERT dependencies with pip install -r bts/requirements-bert.txt") from exc

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"BERT multiple-choice checkpoint not found: {checkpoint}")

    device = resolve_device(torch, args.device)
    print(f"Evaluation device: {device}", flush=True)
    if str(device).startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMultipleChoice.from_pretrained(checkpoint)
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
