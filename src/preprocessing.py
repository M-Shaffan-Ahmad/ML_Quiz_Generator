import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OPTION_COLUMNS = ["A", "B", "C", "D"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def answer_text(row):
    if "correct_answer_text" in row and pd.notna(row["correct_answer_text"]):
        return str(row["correct_answer_text"])
    answer_letter = str(row.get("answer", "")).strip()
    if answer_letter in OPTION_COLUMNS:
        return str(row.get(answer_letter, ""))
    return ""


def build_processed_frame(frame):
    processed = frame.copy()
    processed["correct_answer_text"] = frame.apply(answer_text, axis=1)
    processed["article_token_count"] = processed["article"].fillna("").map(lambda value: len(tokenize(value)))
    processed["question_token_count"] = processed["question"].fillna("").map(lambda value: len(tokenize(value)))
    processed["answer_token_count"] = processed["correct_answer_text"].fillna("").map(lambda value: len(tokenize(value)))
    for option in OPTION_COLUMNS:
        if option in processed:
            processed[f"{option}_token_count"] = processed[option].fillna("").map(lambda value: len(tokenize(value)))
    return processed


def preprocess_split(split, max_rows=None):
    source_path = RAW_DIR / f"{split}.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing expected raw split: {source_path}")
    frame = pd.read_csv(source_path)
    if max_rows:
        frame = frame.head(max_rows)
    processed = build_processed_frame(frame)
    output_path = PROCESSED_DIR / f"{split}_processed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return {
        "split": split,
        "rows": len(processed),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "mean_article_tokens": round(float(processed["article_token_count"].mean()), 4),
        "mean_question_tokens": round(float(processed["question_token_count"].mean()), 4),
        "mean_answer_tokens": round(float(processed["answer_token_count"].mean()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess RACE CSV splits into feature-engineered CSV files.")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for quick smoke runs.")
    args = parser.parse_args()

    summaries = [preprocess_split(split, max(args.max_rows, 0) or None) for split in args.splits]
    summary_path = PROCESSED_DIR / "preprocessing_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nSaved summary to {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
