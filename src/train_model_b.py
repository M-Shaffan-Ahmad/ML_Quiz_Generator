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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from model_a_generation import (
    OPTION_COLUMNS,
    clean_answer,
    find_source_sentence,
    infer_answer_type,
    infer_question_type,
    normalize_text,
    split_sentences,
)
from model_b import (
    DEFAULT_MODEL_B_DIR,
    DISTRACTOR_NUMERIC_FEATURES,
    HINT_NUMERIC_FEATURES,
    MODEL_B_ARTIFACT_FILE,
    build_distractor_feature,
    build_hint_feature,
    extract_distractor_candidates,
    generate_model_b,
    hint_features_to_matrix,
    load_model_b_artifacts,
)


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


def sample_frame(frame, max_rows, random_state):
    if max_rows and len(frame) > max_rows:
        return frame.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return frame.reset_index(drop=True)


def normalized_set(values):
    return {normalize_text(value) for value in values if normalize_text(value)}


def option_candidate(article, question, text):
    source_sentence, start, _end, sentence_index = find_source_sentence(article, text, question)
    sentence_count = max(len(split_sentences(article)), 1)
    return {
        "answer": clean_answer(text),
        "answer_type": infer_answer_type(text, source_sentence or text, max(start, 0)),
        "sentence": source_sentence,
        "sentence_index": sentence_index,
        "sentence_count": sentence_count,
        "source": "option",
    }


def add_correct_option_negative(candidates, article, question, correct_answer):
    correct_key = normalize_text(correct_answer)
    if not correct_key:
        return candidates
    if any(normalize_text(candidate["answer"]) == correct_key for candidate in candidates):
        return candidates
    return candidates + [option_candidate(article, question, correct_answer)]


def build_distractor_dataset(frame, max_rows=None, negatives_per_question=8, random_state=42):
    feature_rows = []
    labels = []
    groups = []
    candidates_by_group = {}
    gold_by_group = {}
    skipped = 0
    rng = np.random.default_rng(random_state)

    for row_number, row in sample_frame(frame, max_rows, random_state).iterrows():
        article = str(row.get("article", ""))
        question = str(row.get("question", ""))
        correct_answer = answer_text_from_row(row)
        gold_distractors = gold_distractors_from_row(row)
        gold_keys = normalized_set(gold_distractors)
        if not article.strip() or not question.strip() or not correct_answer or len(gold_keys) < 1:
            skipped += 1
            continue

        candidates = extract_distractor_candidates(
            article,
            question,
            correct_answer,
            extra_options=option_texts(row),
            max_candidates=100,
        )
        candidates = add_correct_option_negative(candidates, article, question, correct_answer)
        positives = [candidate for candidate in candidates if normalize_text(candidate["answer"]) in gold_keys]
        negatives = [candidate for candidate in candidates if normalize_text(candidate["answer"]) not in gold_keys]
        if not positives or not negatives:
            skipped += 1
            continue

        if len(negatives) > negatives_per_question:
            selected_negative_indices = rng.choice(
                len(negatives),
                size=negatives_per_question,
                replace=False,
            )
            negatives = [negatives[int(index)] for index in selected_negative_indices]

        group_id = str(row.get("question_id", f"row_{row_number}"))
        selected_candidates = positives + negatives
        candidates_by_group[group_id] = selected_candidates
        gold_by_group[group_id] = gold_keys

        for candidate in selected_candidates:
            question_type = str(row.get("question_type", "")) or None
            if not question_type:
                question_type = infer_question_type(question)
            feature_rows.append(
                build_distractor_feature(candidate, article, question, correct_answer, question_type)
            )
            labels.append(int(normalize_text(candidate["answer"]) in gold_keys))
            groups.append(group_id)

    return feature_rows, np.asarray(labels, dtype=np.int8), groups, candidates_by_group, gold_by_group, skipped


def fit_distractor_matrices(rows):
    text_vectorizer = TfidfVectorizer(
        max_features=35000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    category_vectorizer = DictVectorizer(sparse=True)
    X_text = text_vectorizer.fit_transform([row["feature_text"] for row in rows])
    X_cat = category_vectorizer.fit_transform(
        [
            {
                "question_type": row["question_type"],
                "candidate_type": row["candidate_type"],
                "correct_type": row["correct_type"],
                "candidate_source": row["candidate_source"],
            }
            for row in rows
        ]
    )
    X_num = csr_matrix(
        np.asarray(
            [[float(row.get(name, 0.0)) for name in DISTRACTOR_NUMERIC_FEATURES] for row in rows],
            dtype=np.float32,
        )
    )
    artifacts = {
        "distractor_text_vectorizer": text_vectorizer,
        "distractor_category_vectorizer": category_vectorizer,
        "distractor_numeric_features": list(DISTRACTOR_NUMERIC_FEATURES),
    }
    return hstack([X_text, X_cat, X_num]).tocsr(), hstack([X_cat, X_num]).tocsr(), artifacts


def distractor_matrix(rows, artifacts, feature_mode):
    old_mode = artifacts.get("distractor_feature_mode")
    artifacts["distractor_feature_mode"] = feature_mode
    from model_b import distractor_features_to_matrix

    matrix = distractor_features_to_matrix(rows, artifacts)
    if old_mode is None:
        artifacts.pop("distractor_feature_mode", None)
    else:
        artifacts["distractor_feature_mode"] = old_mode
    return matrix


def model_scores(model, matrix):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1]
    decision = model.decision_function(matrix) if hasattr(model, "decision_function") else model.predict(matrix)
    decision = np.asarray(decision, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-decision))


def evaluate_distractor_ranker(model, rows, labels, groups, artifacts, feature_mode):
    matrix = distractor_matrix(rows, artifacts, feature_mode)
    scores = model_scores(model, matrix)
    frame = pd.DataFrame(
        {
            "group": groups,
            "label": labels.astype(int),
            "score": scores,
        }
    )
    predicted = np.zeros(len(frame), dtype=np.int8)
    ranking_rows = []

    for group_id, group_frame in frame.groupby("group", sort=False):
        ordered = group_frame.sort_values("score", ascending=False)
        selected = ordered.head(3)
        predicted[selected.index.to_numpy()] = 1
        hits = int(selected["label"].sum())
        positive_count = max(int(group_frame["label"].sum()), 1)
        precision_at_3 = hits / max(len(selected), 1)
        recall_at_3 = hits / positive_count
        f1_at_3 = 0.0
        if precision_at_3 + recall_at_3:
            f1_at_3 = 2 * precision_at_3 * recall_at_3 / (precision_at_3 + recall_at_3)
        positive_positions = np.where(ordered["label"].to_numpy() == 1)[0]
        ranking_rows.append(
            {
                "group": group_id,
                "precision_at_3": precision_at_3,
                "recall_at_3": recall_at_3,
                "f1_at_3": f1_at_3,
                "exact_set_match": float(hits == positive_count and positive_count <= 3),
                "mrr": 1.0 / float(positive_positions[0] + 1) if len(positive_positions) else 0.0,
            }
        )

    confusion = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "candidate_accuracy": float(accuracy_score(labels, predicted)),
        "candidate_precision": float(precision_score(labels, predicted, zero_division=0)),
        "candidate_recall": float(recall_score(labels, predicted, zero_division=0)),
        "candidate_f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision_at_3": float(np.mean([row["precision_at_3"] for row in ranking_rows])) if ranking_rows else 0.0,
        "recall_at_3": float(np.mean([row["recall_at_3"] for row in ranking_rows])) if ranking_rows else 0.0,
        "f1_at_3": float(np.mean([row["f1_at_3"] for row in ranking_rows])) if ranking_rows else 0.0,
        "exact_set_match": float(np.mean([row["exact_set_match"] for row in ranking_rows])) if ranking_rows else 0.0,
        "mean_reciprocal_rank": float(np.mean([row["mrr"] for row in ranking_rows])) if ranking_rows else 0.0,
        "evaluated_questions": int(len(ranking_rows)),
        "confusion_matrix": confusion.tolist(),
    }


def train_distractor_ranker(train_rows, train_labels, dev_pack, test_pack, rf_max_train_rows, random_state):
    X_full, X_structured, artifacts = fit_distractor_matrices(train_rows)
    models = []

    logistic_ranker = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    print("Training Model B distractor ranker: Logistic Regression", flush=True)
    logistic_ranker.fit(X_full, train_labels)
    models.append(("Logistic Regression", logistic_ranker, "full"))

    rf_indices = np.arange(len(train_labels))
    if rf_max_train_rows and len(rf_indices) > rf_max_train_rows:
        rng = np.random.default_rng(random_state)
        rf_indices = rng.choice(rf_indices, size=rf_max_train_rows, replace=False)
    random_forest = RandomForestClassifier(
        n_estimators=220,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    print("Training Model B distractor ranker: Random Forest", flush=True)
    random_forest.fit(X_structured[rf_indices], train_labels[rf_indices])
    random_forest.n_jobs = 1
    models.append(("Random Forest", random_forest, "structured"))

    result_rows = []
    trained = {}
    for name, model, feature_mode in models:
        trained[name] = (model, feature_mode)
        for split, pack in (("dev", dev_pack), ("test", test_pack)):
            rows, labels, groups = pack
            metrics = evaluate_distractor_ranker(model, rows, labels, groups, artifacts, feature_mode)
            result_rows.append(
                {
                    "model": name,
                    "split": split,
                    "feature_mode": feature_mode,
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                }
            )

    result_frame = pd.DataFrame(result_rows)
    dev_scores = result_frame[result_frame["split"] == "dev"].sort_values(
        ["f1_at_3", "mean_reciprocal_rank", "candidate_f1"],
        ascending=False,
    )
    best_name = str(dev_scores.iloc[0]["model"])
    best_model, best_mode = trained[best_name]
    if hasattr(best_model, "n_jobs"):
        best_model.n_jobs = 1
    artifacts.update(
        {
            "distractor_ranker": best_model,
            "distractor_ranker_name": best_name,
            "distractor_feature_mode": best_mode,
            "distractor_ranker_results": result_frame.to_dict(orient="records"),
        }
    )
    return artifacts, result_frame


def gold_hint_index(article, question, correct_answer):
    sentences = split_sentences(article)
    if not sentences:
        return None
    source_sentence, _start, _end, index = find_source_sentence(article, correct_answer, question)
    if source_sentence and 0 <= index < len(sentences):
        return index
    scores = [
        0.7 * float(normalize_text(correct_answer) in normalize_text(sentence))
        + 0.2 * len(set(normalize_text(question).split()) & set(normalize_text(sentence).split()))
        + 0.1 * len(set(normalize_text(correct_answer).split()) & set(normalize_text(sentence).split()))
        for sentence in sentences
    ]
    return int(np.argmax(scores))


def build_hint_dataset(frame, max_rows=None, random_state=42):
    feature_rows = []
    labels = []
    groups = []
    skipped = 0

    for row_number, row in sample_frame(frame, max_rows, random_state).iterrows():
        article = str(row.get("article", ""))
        question = str(row.get("question", ""))
        correct_answer = answer_text_from_row(row)
        sentences = split_sentences(article)
        index = gold_hint_index(article, question, correct_answer)
        if not sentences or index is None or not correct_answer:
            skipped += 1
            continue
        group_id = str(row.get("question_id", f"row_{row_number}"))
        for sentence_index, sentence in enumerate(sentences):
            feature_rows.append(
                build_hint_feature(sentence, sentence_index, len(sentences), article, question, correct_answer)
            )
            labels.append(int(sentence_index == index))
            groups.append(group_id)
    return feature_rows, np.asarray(labels, dtype=np.int8), groups, skipped


def fit_hint_matrix(rows):
    text_vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_text = text_vectorizer.fit_transform([row["feature_text"] for row in rows])
    X_num = csr_matrix(
        np.asarray(
            [[float(row.get(name, 0.0)) for name in HINT_NUMERIC_FEATURES] for row in rows],
            dtype=np.float32,
        )
    )
    artifacts = {"hint_text_vectorizer": text_vectorizer}
    return hstack([X_text, X_num]).tocsr(), artifacts


def evaluate_hint_ranker(model, rows, labels, groups, artifacts):
    matrix = hint_features_to_matrix(rows, artifacts)
    scores = model_scores(model, matrix)
    frame = pd.DataFrame({"group": groups, "label": labels.astype(int), "score": scores})
    top1 = []
    reciprocal_ranks = []
    for _group_id, group_frame in frame.groupby("group", sort=False):
        ordered = group_frame.sort_values("score", ascending=False)
        ordered_labels = ordered["label"].to_numpy()
        top1.append(int(ordered_labels[0] == 1))
        positive_positions = np.where(ordered_labels == 1)[0]
        reciprocal_ranks.append(1.0 / float(positive_positions[0] + 1) if len(positive_positions) else 0.0)
    return {
        "top1_accuracy": float(np.mean(top1)) if top1 else 0.0,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "evaluated_questions": int(len(top1)),
    }


def train_hint_ranker(train_rows, train_labels, dev_pack, test_pack, random_state):
    X_train, artifacts = fit_hint_matrix(train_rows)
    hint_ranker = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    print("Training Model B hint sentence ranker: Logistic Regression", flush=True)
    hint_ranker.fit(X_train, train_labels)

    result_rows = []
    for split, pack in (("dev", dev_pack), ("test", test_pack)):
        rows, labels, groups = pack
        metrics = evaluate_hint_ranker(hint_ranker, rows, labels, groups, artifacts)
        result_rows.append({"model": "Logistic Regression", "split": split, **metrics})
    artifacts.update(
        {
            "hint_ranker": hint_ranker,
            "hint_ranker_name": "Logistic Regression",
            "hint_ranker_results": result_rows,
        }
    )
    return artifacts, pd.DataFrame(result_rows)


def save_examples(test_frame, output_path, max_examples=10):
    load_model_b_artifacts.cache_clear()
    rows = []
    for _, row in test_frame.head(max_examples).iterrows():
        article = str(row.get("article", ""))
        question = str(row.get("question", ""))
        correct_answer = answer_text_from_row(row)
        if not article.strip() or not question.strip() or not correct_answer:
            continue
        output = generate_model_b(
            article,
            question,
            correct_answer,
            extra_options=None,
            top_k=3,
            model_dir=output_path.parent,
        )
        for rank, distractor in enumerate(output.distractors, start=1):
            rows.append(
                {
                    "question_id": row.get("question_id", ""),
                    "component": "distractor",
                    "rank_or_level": rank,
                    "question_type": output.question_type,
                    "correct_answer": correct_answer,
                    "text": distractor.text,
                    "score": distractor.score,
                    "answer_type": distractor.answer_type,
                    "source_sentence": distractor.source_sentence,
                    "method": distractor.method,
                    "question": question,
                }
            )
        for hint in output.hints:
            rows.append(
                {
                    "question_id": row.get("question_id", ""),
                    "component": "hint",
                    "rank_or_level": hint.level,
                    "question_type": output.question_type,
                    "correct_answer": correct_answer,
                    "text": hint.text,
                    "score": hint.score,
                    "answer_type": "",
                    "source_sentence": hint.source_sentence,
                    "method": hint.method,
                    "question": question,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train Model B distractor and hint generator.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_B_DIR))
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument("--max-eval-rows", type=int, default=2500)
    parser.add_argument("--negatives-per-question", type=int, default=8)
    parser.add_argument("--rf-max-train-rows", type=int, default=90000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(args.train_csv)
    dev_frame = pd.read_csv(args.dev_csv)
    test_frame = pd.read_csv(args.test_csv)

    print("Building Model B distractor datasets...", flush=True)
    train_rows, train_labels, train_groups, _train_candidates, _train_gold, train_skipped = build_distractor_dataset(
        train_frame,
        max_rows=args.max_train_rows,
        negatives_per_question=args.negatives_per_question,
        random_state=args.random_state,
    )
    dev_rows, dev_labels, dev_groups, _dev_candidates, _dev_gold, dev_skipped = build_distractor_dataset(
        dev_frame,
        max_rows=args.max_eval_rows,
        negatives_per_question=max(args.negatives_per_question, 10),
        random_state=args.random_state + 1,
    )
    test_rows, test_labels, test_groups, _test_candidates, _test_gold, test_skipped = build_distractor_dataset(
        test_frame,
        max_rows=args.max_eval_rows,
        negatives_per_question=max(args.negatives_per_question, 10),
        random_state=args.random_state + 2,
    )
    print(
        "Distractor rows: "
        f"train={len(train_rows)} positives={int(train_labels.sum())}; "
        f"dev={len(dev_rows)} positives={int(dev_labels.sum())}; "
        f"test={len(test_rows)} positives={int(test_labels.sum())}; "
        f"skipped train/dev/test={train_skipped}/{dev_skipped}/{test_skipped}",
        flush=True,
    )
    distractor_artifacts, distractor_results = train_distractor_ranker(
        train_rows,
        train_labels,
        (dev_rows, dev_labels, dev_groups),
        (test_rows, test_labels, test_groups),
        rf_max_train_rows=args.rf_max_train_rows,
        random_state=args.random_state,
    )
    distractor_results.to_csv(output_dir / "model_b_distractor_results.csv", index=False)
    print(distractor_results.to_string(index=False), flush=True)

    print("Building Model B hint datasets...", flush=True)
    hint_train_rows, hint_train_labels, hint_train_groups, hint_train_skipped = build_hint_dataset(
        train_frame,
        max_rows=args.max_train_rows,
        random_state=args.random_state,
    )
    hint_dev_rows, hint_dev_labels, hint_dev_groups, hint_dev_skipped = build_hint_dataset(
        dev_frame,
        max_rows=args.max_eval_rows,
        random_state=args.random_state + 3,
    )
    hint_test_rows, hint_test_labels, hint_test_groups, hint_test_skipped = build_hint_dataset(
        test_frame,
        max_rows=args.max_eval_rows,
        random_state=args.random_state + 4,
    )
    print(
        "Hint rows: "
        f"train={len(hint_train_rows)} positives={int(hint_train_labels.sum())}; "
        f"dev={len(hint_dev_rows)} positives={int(hint_dev_labels.sum())}; "
        f"test={len(hint_test_rows)} positives={int(hint_test_labels.sum())}; "
        f"skipped train/dev/test={hint_train_skipped}/{hint_dev_skipped}/{hint_test_skipped}",
        flush=True,
    )
    hint_artifacts, hint_results = train_hint_ranker(
        hint_train_rows,
        hint_train_labels,
        (hint_dev_rows, hint_dev_labels, hint_dev_groups),
        (hint_test_rows, hint_test_labels, hint_test_groups),
        random_state=args.random_state,
    )
    hint_results.to_csv(output_dir / "model_b_hint_results.csv", index=False)
    print(hint_results.to_string(index=False), flush=True)

    artifacts = {
        **distractor_artifacts,
        **hint_artifacts,
        "training_config": vars(args),
        "dataset_summary": {
            "distractor_train_rows": int(len(train_rows)),
            "distractor_train_positives": int(train_labels.sum()),
            "distractor_skipped_train_questions": int(train_skipped),
            "hint_train_rows": int(len(hint_train_rows)),
            "hint_train_positives": int(hint_train_labels.sum()),
            "hint_skipped_train_questions": int(hint_train_skipped),
        },
    }
    joblib.dump(artifacts, output_dir / MODEL_B_ARTIFACT_FILE)

    save_examples(test_frame, output_dir / "model_b_examples.csv")
    with open(output_dir / "model_b_training_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "artifact_file": str(output_dir / MODEL_B_ARTIFACT_FILE),
                "selected_distractor_ranker": artifacts["distractor_ranker_name"],
                "selected_distractor_feature_mode": artifacts["distractor_feature_mode"],
                "selected_hint_ranker": artifacts["hint_ranker_name"],
                "distractor_results": distractor_results.to_dict(orient="records"),
                "hint_results": hint_results.to_dict(orient="records"),
                "dataset_summary": artifacts["dataset_summary"],
            },
            file,
            indent=2,
        )
    print(f"Saved Model B artifacts to {output_dir / MODEL_B_ARTIFACT_FILE}", flush=True)


if __name__ == "__main__":
    main()
