import argparse
from pathlib import Path

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

from model_a_generation import OPTION_COLUMNS, clean_answer, normalize_text, tokenize
from model_b import DEFAULT_MODEL_B_DIR, generate_model_b


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NoWordNet:
    """Keep METEOR offline by skipping WordNet synonym matching."""

    def synsets(self, *args, **kwargs):
        return []


def answer_text_from_row(row):
    if "correct_answer_text" in row and pd.notna(row["correct_answer_text"]):
        return clean_answer(row["correct_answer_text"])
    answer_letter = str(row.get("answer", "")).strip()
    if answer_letter in OPTION_COLUMNS:
        return clean_answer(row.get(answer_letter, ""))
    return ""


def option_texts(row):
    values = []
    for label in OPTION_COLUMNS:
        value = row.get(label, "")
        if pd.notna(value):
            text = clean_answer(value)
            if text:
                values.append(text)
    return values


def gold_distractors_from_row(row):
    answer_letter = str(row.get("answer", "")).strip()
    distractors = []
    for label in OPTION_COLUMNS:
        if label == answer_letter:
            continue
        value = row.get(label, "")
        if pd.notna(value):
            text = clean_answer(value)
            if text:
                distractors.append(text)
    return distractors


def rouge_l_f1(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    previous = [0] * (len(hypothesis_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            if reference_token == hypothesis_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    overlap = previous[-1]
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def compute_generation_metrics(reference_groups, hypotheses):
    if not reference_groups or not hypotheses:
        return {"BLEU": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0, "exact_text_match": 0.0}

    tokenized_reference_groups = [[tokenize(reference) for reference in refs] for refs in reference_groups]
    tokenized_hypotheses = [tokenize(hypothesis) for hypothesis in hypotheses]
    bleu = corpus_bleu(
        tokenized_reference_groups,
        tokenized_hypotheses,
        smoothing_function=SmoothingFunction().method1,
    )
    rouge_l = sum(
        max(rouge_l_f1(reference, hypothesis) for reference in references)
        for references, hypothesis in zip(tokenized_reference_groups, tokenized_hypotheses)
    ) / len(tokenized_hypotheses)
    meteor = sum(
        meteor_score(references, hypothesis, wordnet=NoWordNet())
        for references, hypothesis in zip(tokenized_reference_groups, tokenized_hypotheses)
    ) / len(tokenized_hypotheses)
    exact = sum(
        any(normalize_text(hypothesis) == normalize_text(reference) for reference in references)
        for references, hypothesis in zip(reference_groups, hypotheses)
    ) / len(hypotheses)
    return {"BLEU": bleu, "ROUGE-L": rouge_l, "METEOR": meteor, "exact_text_match": exact}


def evaluate_split(frame, split_name, mode, max_rows, model_dir):
    if max_rows:
        frame = frame.head(max_rows)

    reference_groups = []
    hypotheses = []
    evaluated_questions = 0
    full_quiz_questions = 0

    for _, row in frame.iterrows():
        article = str(row.get("article", ""))
        question = str(row.get("question", ""))
        correct_answer = answer_text_from_row(row)
        gold_distractors = gold_distractors_from_row(row)
        if not article.strip() or not question.strip() or not correct_answer or len(gold_distractors) < 1:
            continue

        extra_options = option_texts(row) if mode == "option_pool" else None
        output = generate_model_b(
            article,
            question,
            correct_answer,
            extra_options=extra_options,
            top_k=3,
            model_dir=model_dir,
        )
        generated = [distractor.text for distractor in output.distractors]
        if not generated:
            continue
        evaluated_questions += 1
        if len(generated) >= 3:
            full_quiz_questions += 1
        for generated_text in generated:
            reference_groups.append(gold_distractors)
            hypotheses.append(generated_text)

    metrics = compute_generation_metrics(reference_groups, hypotheses)
    return {
        "split": split_name,
        "mode": mode,
        "evaluated_questions": evaluated_questions,
        "full_three_distractor_questions": full_quiz_questions,
        "generated_distractors": len(hypotheses),
        "mean_generated_per_question": len(hypotheses) / max(evaluated_questions, 1),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model B distractor text quality with BLEU/ROUGE/METEOR.")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_B_DIR))
    parser.add_argument("--output", default=str(DEFAULT_MODEL_B_DIR / "model_b_generation_text_metrics.csv"))
    parser.add_argument("--max-eval-rows", type=int, default=500)
    parser.add_argument("--modes", nargs="+", default=["option_pool", "passage_grounded"])
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    rows = []
    for split_name, csv_path in (("dev", args.dev_csv), ("test", args.test_csv)):
        frame = pd.read_csv(csv_path)
        for mode in args.modes:
            rows.append(evaluate_split(frame, split_name, mode, args.max_eval_rows, model_dir))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output_path, index=False)
    print(result_frame.to_string(index=False))
    try:
        display_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output_path
    print(f"Saved Model B text metrics to {display_path}")


if __name__ == "__main__":
    main()
