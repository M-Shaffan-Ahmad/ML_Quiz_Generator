import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

from model_a_generation import (
    ARTIFACT_FILE,
    CLASSIFIER_NUMERIC_FEATURES,
    DEFAULT_GENERATOR_DIR,
    OPTION_COLUMNS,
    QUESTION_TYPES,
    RANKER_NUMERIC_FEATURES,
    build_candidate_frame,
    build_classifier_feature,
    build_ranker_feature,
    classifier_features_to_matrix,
    clean_answer,
    find_answer_span,
    find_source_sentence,
    infer_answer_type,
    infer_question_type,
    jaccard_overlap,
    make_typed_question,
    normalize_text,
    predict_question_type,
    rank_candidate,
    ranker_features_to_matrix,
    split_sentences,
)


def answer_text_from_row(row):
    if "correct_answer_text" in row and pd.notna(row["correct_answer_text"]):
        return clean_answer(row["correct_answer_text"])
    answer_letter = str(row.get("answer", "")).strip()
    if answer_letter in OPTION_COLUMNS:
        return clean_answer(row.get(answer_letter, ""))
    return ""


def sample_frame(frame, max_rows, random_state):
    if max_rows and len(frame) > max_rows:
        return frame.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return frame.reset_index(drop=True)


def gold_candidate_from_row(row):
    article = str(row["article"])
    question = str(row["question"])
    answer = answer_text_from_row(row)
    if not article.strip() or not answer:
        return None

    source_sentence, start, end, sentence_index = find_source_sentence(article, answer, question)
    if not source_sentence:
        return None

    sentence_count = max(len(split_sentences(article)), 1)
    if start < 0:
        start, end = find_answer_span(source_sentence, answer)
    answer_type = infer_answer_type(answer, source_sentence, max(start, 0))
    candidate = {
        "sentence": source_sentence,
        "answer": answer,
        "start": start,
        "end": end,
        "answer_type": answer_type,
        "sentence_index": sentence_index,
        "sentence_count": sentence_count,
    }
    candidate["heuristic_score"] = rank_candidate(candidate, sentence_index, sentence_count)
    return candidate


def build_classifier_dataset(frame, max_rows=None, random_state=42):
    rows = []
    labels = []
    skipped = 0
    for _, row in sample_frame(frame, max_rows, random_state).iterrows():
        candidate = gold_candidate_from_row(row)
        if candidate is None:
            skipped += 1
            continue
        label = infer_question_type(row["question"])
        rows.append(build_classifier_feature(candidate, str(row["article"])))
        labels.append(label)
    return rows, np.asarray(labels), skipped


def fit_classifier_matrix(rows):
    text_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    category_vectorizer = DictVectorizer(sparse=True)
    X_text = text_vectorizer.fit_transform([row["feature_text"] for row in rows])
    X_cat = category_vectorizer.fit_transform(
        [{"answer_type": row["answer_type"]} for row in rows]
    )
    X_num = csr_matrix(
        np.asarray(
            [[float(row.get(name, 0.0)) for name in CLASSIFIER_NUMERIC_FEATURES] for row in rows],
            dtype=np.float32,
        )
    )
    X = hstack([X_text, X_cat, X_num]).tocsr()
    artifacts = {
        "text_vectorizer": text_vectorizer,
        "category_vectorizer": category_vectorizer,
    }
    return X, artifacts


def evaluate_classifier(model, artifacts, rows, labels):
    X = classifier_features_to_matrix(rows, artifacts)
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def train_question_type_classifier(train_rows, train_labels, dev_rows, dev_labels, test_rows, test_labels):
    X_train, artifacts = fit_classifier_matrix(train_rows)
    models = {
        "Logistic Regression": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        ),
        "Linear SVM": SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-5,
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        ),
    }
    results = []
    trained_models = {}
    for name, model in models.items():
        print(f"Training question-type classifier: {name}", flush=True)
        model.fit(X_train, train_labels)
        trained_models[name] = model
        for split, rows, labels in (
            ("dev", dev_rows, dev_labels),
            ("test", test_rows, test_labels),
        ):
            metrics = evaluate_classifier(model, artifacts, rows, labels)
            results.append({"model": name, "split": split, **metrics})

    result_frame = pd.DataFrame(results)
    dev_scores = result_frame[result_frame["split"] == "dev"].sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )
    best_name = str(dev_scores.iloc[0]["model"])
    artifacts.update(
        {
            "question_type_classifier": trained_models[best_name],
            "question_type_classifier_name": best_name,
            "question_types": QUESTION_TYPES,
            "classifier_results": result_frame.to_dict(orient="records"),
        }
    )
    return artifacts, result_frame


def add_gold_candidate_if_needed(candidates, row):
    gold = gold_candidate_from_row(row)
    if gold is None:
        return candidates
    gold_key = normalize_text(gold["answer"])
    if not any(normalize_text(candidate["answer"]) == gold_key for candidate in candidates):
        candidates = [gold] + candidates
    return candidates


def candidate_label(candidate, gold_answer):
    return int(normalize_text(candidate["answer"]) == normalize_text(gold_answer))


def build_ranker_dataset(frame, classifier_artifacts, max_rows=None, negatives_per_question=4, random_state=42):
    feature_rows = []
    labels = []
    skipped = 0
    for _, row in sample_frame(frame, max_rows, random_state).iterrows():
        article = str(row["article"])
        gold_answer = answer_text_from_row(row)
        if not gold_answer:
            skipped += 1
            continue
        candidates = add_gold_candidate_if_needed(build_candidate_frame(article), row)
        positives = [candidate for candidate in candidates if candidate_label(candidate, gold_answer)]
        negatives = [candidate for candidate in candidates if not candidate_label(candidate, gold_answer)]
        if not positives:
            skipped += 1
            continue

        selected = positives[:1] + negatives[:negatives_per_question]
        for candidate in selected:
            question_type, confidence = predict_question_type(candidate, article, classifier_artifacts)
            question = make_typed_question(candidate["sentence"], candidate["answer"], question_type)
            feature_rows.append(build_ranker_feature(candidate, question, question_type, confidence))
            labels.append(candidate_label(candidate, gold_answer))

    return feature_rows, np.asarray(labels), skipped


def train_ranker(train_frame, classifier_artifacts, max_rows, negatives_per_question):
    print("Building ranker training rows...", flush=True)
    rows, labels, skipped = build_ranker_dataset(
        train_frame,
        classifier_artifacts,
        max_rows=max_rows,
        negatives_per_question=negatives_per_question,
        random_state=43,
    )
    print(f"Ranker rows: {len(rows)}; positives: {int(labels.sum())}; skipped: {skipped}", flush=True)
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(rows)
    ranker = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    ranker.fit(X, labels)
    ranker.n_jobs = 1
    classifier_artifacts["ranker_vectorizer"] = vectorizer
    classifier_artifacts["ranker"] = ranker
    classifier_artifacts["ranker_skipped_train_rows"] = skipped
    return classifier_artifacts


def ranker_score_rows(rows, artifacts):
    X = ranker_features_to_matrix(rows, artifacts)
    ranker = artifacts["ranker"]
    if hasattr(ranker, "predict_proba"):
        return ranker.predict_proba(X)[:, 1]
    return ranker.predict(X)


def evaluate_ranker(frame, artifacts, max_rows=None, negatives_per_question=8, random_state=44):
    top1 = []
    reciprocal_ranks = []
    skipped = 0
    for _, row in sample_frame(frame, max_rows, random_state).iterrows():
        article = str(row["article"])
        gold_answer = answer_text_from_row(row)
        if not gold_answer:
            skipped += 1
            continue
        candidates = add_gold_candidate_if_needed(build_candidate_frame(article), row)
        positives = [candidate for candidate in candidates if candidate_label(candidate, gold_answer)]
        negatives = [candidate for candidate in candidates if not candidate_label(candidate, gold_answer)]
        selected = positives[:1] + negatives[:negatives_per_question]
        if not positives or len(selected) < 2:
            skipped += 1
            continue

        feature_rows = []
        labels = []
        for candidate in selected:
            question_type, confidence = predict_question_type(candidate, article, artifacts)
            question = make_typed_question(candidate["sentence"], candidate["answer"], question_type)
            feature_rows.append(build_ranker_feature(candidate, question, question_type, confidence))
            labels.append(candidate_label(candidate, gold_answer))

        scores = ranker_score_rows(feature_rows, artifacts)
        order = np.argsort(-scores)
        ordered_labels = np.asarray(labels)[order]
        top1.append(int(ordered_labels[0] == 1))
        positive_positions = np.where(ordered_labels == 1)[0]
        if len(positive_positions):
            reciprocal_ranks.append(1.0 / float(positive_positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    return {
        "top1_accuracy": float(np.mean(top1)) if top1 else 0.0,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "evaluated_questions": int(len(top1)),
        "skipped_questions": int(skipped),
    }


def split_label_distribution(labels, split):
    counts = pd.Series(labels).value_counts().rename_axis("question_type").reset_index(name="count")
    counts["split"] = split
    return counts[["split", "question_type", "count"]]


def save_examples(test_frame, artifacts, output_path, max_examples=8):
    from model_a_generation import generate_questions

    rows = []
    for _, row in test_frame.head(max_examples).iterrows():
        generated = generate_questions(
            str(row["article"]),
            top_k=3,
            mode="both",
            model_dir=output_path.parent,
        )
        for rank, item in enumerate(generated, start=1):
            rows.append(
                {
                    "question_id": row.get("question_id", ""),
                    "rank": rank,
                    "question_type": item.question_type,
                    "answer": item.answer,
                    "generated_question": item.question,
                    "source_sentence": item.source_sentence,
                    "classifier_confidence": item.classifier_confidence,
                    "ranker_score": item.ranker_score,
                    "selection_score": item.score,
                    "cluster_id": item.cluster_id,
                    "cluster_question_type": item.cluster_question_type,
                    "cluster_purity": item.cluster_purity,
                    "cluster_distance": item.cluster_distance,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train supervised Model A question generator.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_GENERATOR_DIR))
    parser.add_argument("--max-classifier-train-rows", type=int, default=60000)
    parser.add_argument("--max-ranker-train-rows", type=int, default=25000)
    parser.add_argument("--max-ranker-eval-rows", type=int, default=1200)
    parser.add_argument("--negatives-per-question", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(args.train_csv)
    dev_frame = pd.read_csv(args.dev_csv)
    test_frame = pd.read_csv(args.test_csv)

    print("Building question-type classifier datasets...", flush=True)
    train_rows, train_labels, train_skipped = build_classifier_dataset(
        train_frame,
        max_rows=args.max_classifier_train_rows,
        random_state=args.random_state,
    )
    dev_rows, dev_labels, dev_skipped = build_classifier_dataset(dev_frame)
    test_rows, test_labels, test_skipped = build_classifier_dataset(test_frame)
    print(
        f"Classifier rows: train={len(train_rows)}, dev={len(dev_rows)}, test={len(test_rows)}; "
        f"skipped train/dev/test={train_skipped}/{dev_skipped}/{test_skipped}",
        flush=True,
    )

    label_distribution = pd.concat(
        [
            split_label_distribution(train_labels, "train"),
            split_label_distribution(dev_labels, "dev"),
            split_label_distribution(test_labels, "test"),
        ],
        ignore_index=True,
    )
    label_distribution.to_csv(output_dir / "question_type_label_distribution.csv", index=False)

    artifacts, classifier_results = train_question_type_classifier(
        train_rows,
        train_labels,
        dev_rows,
        dev_labels,
        test_rows,
        test_labels,
    )
    classifier_results.to_csv(output_dir / "question_type_classifier_results.csv", index=False)
    print(classifier_results.to_string(index=False), flush=True)

    artifacts = train_ranker(
        train_frame,
        artifacts,
        max_rows=args.max_ranker_train_rows,
        negatives_per_question=args.negatives_per_question,
    )

    ranker_results = []
    for split, frame in (("dev", dev_frame), ("test", test_frame)):
        metrics = evaluate_ranker(
            frame,
            artifacts,
            max_rows=args.max_ranker_eval_rows,
            negatives_per_question=max(args.negatives_per_question, 8),
        )
        ranker_results.append({"split": split, **metrics})
    ranker_results_frame = pd.DataFrame(ranker_results)
    ranker_results_frame.to_csv(output_dir / "generation_ranker_results.csv", index=False)
    print(ranker_results_frame.to_string(index=False), flush=True)

    artifacts["training_config"] = vars(args)
    artifacts["label_distribution"] = label_distribution.to_dict(orient="records")
    artifacts["ranker_results"] = ranker_results_frame.to_dict(orient="records")
    joblib.dump(artifacts, output_dir / ARTIFACT_FILE)

    save_examples(test_frame, artifacts, output_dir / "generation_examples.csv")
    with open(output_dir / "generator_training_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "classifier_results": classifier_results.to_dict(orient="records"),
                "ranker_results": ranker_results_frame.to_dict(orient="records"),
                "best_classifier": artifacts["question_type_classifier_name"],
                "artifact_file": str(output_dir / ARTIFACT_FILE),
            },
            file,
            indent=2,
        )
    print(f"Saved generator artifacts to {output_dir / ARTIFACT_FILE}", flush=True)


if __name__ == "__main__":
    main()
