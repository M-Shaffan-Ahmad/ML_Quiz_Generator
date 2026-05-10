import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_a_generation import (
    OPTION_COLUMNS,
    clean_answer,
    find_answer_span,
    find_source_sentence,
    infer_answer_type,
    infer_question_type,
    rank_candidate,
    split_sentences,
)
from model_a_unsupervised import (
    DEFAULT_UNSUPERVISED_DIR,
    UNSUPERVISED_ARTIFACT_FILE,
    build_option_cluster_matrix,
    option_cluster_texts,
    option_dense_features,
    question_cluster_text,
)


QUESTION_CUE_WORDS = {"who", "what", "where", "when", "why", "how", "which", "many", "much"}
QUESTION_CLUSTER_STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) - QUESTION_CUE_WORDS)


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
    candidate = {
        "sentence": source_sentence,
        "answer": answer,
        "start": start,
        "end": end,
        "answer_type": infer_answer_type(answer, source_sentence, max(start, 0)),
        "sentence_index": sentence_index,
        "sentence_count": sentence_count,
    }
    candidate["heuristic_score"] = rank_candidate(candidate, sentence_index, sentence_count)
    return candidate


def build_question_cluster_frame(frame, max_rows, random_state):
    rows = []
    skipped = 0
    for _, row in sample_frame(frame, max_rows, random_state).iterrows():
        candidate = gold_candidate_from_row(row)
        if candidate is None:
            skipped += 1
            continue
        question_type = infer_question_type(row["question"])
        rows.append(
            {
                "question_id": row.get("question_id", row.get("id", "")),
                "article": str(row["article"]),
                "question": str(row["question"]),
                "answer": candidate["answer"],
                "answer_type": candidate["answer_type"],
                "source_sentence": candidate["sentence"],
                "question_type": question_type,
                "cluster_text": question_cluster_text(
                    row["question"],
                    candidate["answer"],
                    candidate["answer_type"],
                    candidate["sentence"],
                    row["article"],
                ),
            }
        )
    return pd.DataFrame(rows), skipped


def balance_question_types(frame, max_per_type, random_state):
    if frame.empty or not max_per_type:
        return frame
    balanced = []
    for _label, group in frame.groupby("question_type"):
        if len(group) > max_per_type:
            balanced.append(group.sample(n=max_per_type, random_state=random_state))
        else:
            balanced.append(group)
    return pd.concat(balanced, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def to_option_long(frame, max_questions, random_state):
    sampled = sample_frame(frame, max_questions, random_state).copy()
    sampled["qid"] = np.arange(len(sampled))
    id_vars = [
        column
        for column in [
            "qid",
            "question_id",
            "example_id",
            "level",
            "article",
            "question",
            "answer",
            "correct_answer_text",
        ]
        if column in sampled.columns
    ]
    long_df = sampled.melt(
        id_vars=id_vars,
        value_vars=OPTION_COLUMNS,
        var_name="option_letter",
        value_name="option",
    )
    long_df["label"] = (long_df["answer"] == long_df["option_letter"]).astype(int)
    return long_df


def choose_k(matrix, k_values, sample_size, random_state):
    rng = np.random.default_rng(random_state)
    sample_count = min(sample_size, len(matrix))
    sample_index = rng.choice(len(matrix), size=sample_count, replace=False)
    sample_matrix = matrix[sample_index]

    rows = []
    best = None
    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=2048,
            n_init=5,
            max_iter=200,
        )
        model.fit(matrix)
        sample_labels = model.predict(sample_matrix)
        if len(set(sample_labels)) > 1:
            silhouette = float(silhouette_score(sample_matrix, sample_labels))
        else:
            silhouette = 0.0
        row = {"k": int(k), "silhouette": silhouette, "inertia": float(model.inertia_)}
        rows.append(row)
        if best is None or silhouette > best["silhouette"]:
            best = {**row, "model": model}
    return best["model"], pd.DataFrame(rows)


def cluster_purity(clusters, labels):
    frame = pd.DataFrame({"cluster": clusters, "label": labels})
    total = len(frame)
    if total == 0:
        return 0.0
    majority = 0
    for _, group in frame.groupby("cluster"):
        majority += int(group["label"].value_counts().iloc[0])
    return majority / total


def sampled_silhouette(matrix, clusters, sample_size, random_state):
    if len(set(clusters)) < 2:
        return 0.0
    if len(matrix) <= sample_size:
        return float(silhouette_score(matrix, clusters))
    rng = np.random.default_rng(random_state)
    sample_index = rng.choice(len(matrix), size=sample_size, replace=False)
    sample_labels = np.asarray(clusters)[sample_index]
    if len(set(sample_labels)) < 2:
        return 0.0
    return float(silhouette_score(matrix[sample_index], sample_labels))


def top_terms_for_clusters(kmeans, svd, feature_names, text_feature_count, top_n=10):
    reconstructed = np.asarray(kmeans.cluster_centers_ @ svd.components_)
    reconstructed = reconstructed[:, :text_feature_count]
    terms = np.asarray(feature_names)
    result = {}
    for cluster_id, center in enumerate(reconstructed):
        if len(center) == 0:
            result[str(cluster_id)] = []
            continue
        top_indices = np.argsort(center)[-top_n:][::-1]
        result[str(cluster_id)] = [str(terms[index]) for index in top_indices if center[index] > 0]
    return result


def question_profiles(frame, clusters, top_terms):
    rows = []
    profiles = {}
    frame = frame.copy()
    frame["cluster"] = clusters
    for cluster_id, group in frame.groupby("cluster"):
        type_counts = group["question_type"].value_counts()
        answer_type_counts = group["answer_type"].value_counts()
        dominant = str(type_counts.index[0])
        purity = float(type_counts.iloc[0] / len(group))
        profile = {
            "cluster_id": int(cluster_id),
            "count": int(len(group)),
            "dominant_question_type": dominant,
            "question_type_purity": purity,
            "question_type_distribution": type_counts.to_dict(),
            "answer_type_distribution": answer_type_counts.to_dict(),
            "top_terms": top_terms.get(str(cluster_id), []),
        }
        profiles[str(cluster_id)] = profile
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(group)),
                "dominant_question_type": dominant,
                "question_type_purity": round(purity, 4),
                "top_terms": ", ".join(profile["top_terms"]),
                "question_type_distribution": json.dumps(profile["question_type_distribution"]),
                "answer_type_distribution": json.dumps(profile["answer_type_distribution"]),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id"), profiles


def answer_profiles(frame, clusters, top_terms):
    rows = []
    profiles = {}
    frame = frame.copy()
    frame["cluster"] = clusters
    for cluster_id, group in frame.groupby("cluster"):
        label_counts = group["label"].value_counts()
        correct_rate = float(group["label"].mean())
        dominant_label = int(label_counts.index[0])
        purity = float(label_counts.iloc[0] / len(group))
        profile = {
            "cluster_id": int(cluster_id),
            "count": int(len(group)),
            "correct_rate": correct_rate,
            "dominant_label": dominant_label,
            "correctness_purity": purity,
            "top_terms": top_terms.get(str(cluster_id), []),
        }
        profiles[str(cluster_id)] = profile
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(group)),
                "correct_rate": round(correct_rate, 4),
                "dominant_label": dominant_label,
                "correctness_purity": round(purity, 4),
                "top_terms": ", ".join(profile["top_terms"]),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id"), profiles


def save_dendrogram(matrix, output_path, sample_size, random_state, title):
    sample_count = min(sample_size, len(matrix))
    rng = np.random.default_rng(random_state)
    sample_index = rng.choice(len(matrix), size=sample_count, replace=False)
    linked = linkage(matrix[sample_index], method="ward")
    plt.figure(figsize=(14, 7))
    dendrogram(linked, no_labels=True, color_threshold=None)
    plt.title(title)
    plt.xlabel("Sampled rows")
    plt.ylabel("Ward distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def fit_question_cluster(train_frame, args, output_dir):
    print("Building question/candidate clustering frame...", flush=True)
    train_questions, skipped = build_question_cluster_frame(
        train_frame,
        args.max_question_train_rows,
        args.random_state,
    )
    raw_question_rows = len(train_questions)
    if args.balance_question_types:
        train_questions = balance_question_types(
            train_questions,
            args.max_question_type_rows,
            args.random_state,
        )
    print(f"Question cluster rows: {len(train_questions)} | skipped: {skipped}", flush=True)

    vectorizer = TfidfVectorizer(
        max_features=args.question_max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=QUESTION_CLUSTER_STOP_WORDS,
        sublinear_tf=True,
    )
    text_matrix = vectorizer.fit_transform(train_questions["cluster_text"])
    components = min(args.svd_components, max(2, text_matrix.shape[1] - 1), max(2, len(train_questions) - 1))
    svd = TruncatedSVD(n_components=components, random_state=args.random_state)
    reduced = svd.fit_transform(text_matrix)

    k_values = list(range(args.min_k, args.max_k + 1))
    kmeans, k_table = choose_k(reduced, k_values, args.k_eval_sample_size, args.random_state)
    k_table.to_csv(output_dir / "question_cluster_k_selection.csv", index=False)

    clusters = kmeans.predict(reduced)
    top_terms = top_terms_for_clusters(
        kmeans,
        svd,
        vectorizer.get_feature_names_out(),
        text_matrix.shape[1],
    )
    profiles_table, profiles = question_profiles(train_questions, clusters, top_terms)
    profiles_table.to_csv(output_dir / "question_cluster_profiles.csv", index=False)
    save_dendrogram(
        reduced,
        output_dir / "question_cluster_dendrogram.png",
        args.dendrogram_sample_size,
        args.random_state,
        "Question/Candidate Hierarchical Dendrogram",
    )

    metrics = {
        "train_rows": int(len(train_questions)),
        "raw_train_rows_before_balance": int(raw_question_rows),
        "skipped_rows": int(skipped),
        "balanced_question_types": bool(args.balance_question_types),
        "best_k": int(kmeans.n_clusters),
        "silhouette": sampled_silhouette(reduced, clusters, args.k_eval_sample_size, args.random_state),
        "question_type_purity": cluster_purity(clusters, train_questions["question_type"]),
        "explained_variance_ratio": float(np.sum(svd.explained_variance_ratio_)),
    }
    artifact = {
        "vectorizer": vectorizer,
        "svd": svd,
        "kmeans": kmeans,
        "profiles": profiles,
        "metrics": metrics,
    }
    return artifact, train_questions, metrics


def fit_answer_cluster(train_frame, args, output_dir):
    print("Building answer-option clustering frame...", flush=True)
    train_options = to_option_long(train_frame, args.max_answer_train_questions, args.random_state)
    print(f"Answer cluster option rows: {len(train_options)}", flush=True)

    vectorizer = TfidfVectorizer(
        max_features=args.answer_max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )
    text_matrix = vectorizer.fit_transform(option_cluster_texts(train_options))
    dense = option_dense_features(train_options, vectorizer)
    feature_matrix = hstack([text_matrix, dense]).tocsr()
    components = min(args.svd_components, max(2, feature_matrix.shape[1] - 1), max(2, len(train_options) - 1))
    svd = TruncatedSVD(n_components=components, random_state=args.random_state)
    reduced = svd.fit_transform(feature_matrix)

    k_values = list(range(args.min_k, args.max_k + 1))
    kmeans, k_table = choose_k(reduced, k_values, args.k_eval_sample_size, args.random_state)
    k_table.to_csv(output_dir / "answer_cluster_k_selection.csv", index=False)

    clusters = kmeans.predict(reduced)
    top_terms = top_terms_for_clusters(
        kmeans,
        svd,
        vectorizer.get_feature_names_out(),
        text_matrix.shape[1],
    )
    profiles_table, profiles = answer_profiles(train_options, clusters, top_terms)
    profiles_table.to_csv(output_dir / "answer_cluster_profiles.csv", index=False)
    save_dendrogram(
        reduced,
        output_dir / "answer_cluster_dendrogram.png",
        args.dendrogram_sample_size,
        args.random_state,
        "Answer-Option Hierarchical Dendrogram",
    )

    metrics = {
        "train_questions": int(train_options["qid"].nunique()),
        "train_option_rows": int(len(train_options)),
        "best_k": int(kmeans.n_clusters),
        "silhouette": sampled_silhouette(reduced, clusters, args.k_eval_sample_size, args.random_state),
        "correctness_purity": cluster_purity(clusters, train_options["label"]),
        "explained_variance_ratio": float(np.sum(svd.explained_variance_ratio_)),
    }
    artifact = {
        "vectorizer": vectorizer,
        "svd": svd,
        "kmeans": kmeans,
        "profiles": profiles,
        "metrics": metrics,
    }
    return artifact, train_options, metrics


def evaluate_question_split(split_name, frame, artifact, args):
    eval_frame, skipped = build_question_cluster_frame(
        frame,
        args.max_eval_questions,
        args.random_state + 17,
    )
    if eval_frame.empty:
        return {
            "split": split_name,
            "rows": 0,
            "skipped_rows": int(skipped),
            "silhouette": 0.0,
            "question_type_purity": 0.0,
        }
    matrix = artifact["svd"].transform(artifact["vectorizer"].transform(eval_frame["cluster_text"]))
    clusters = artifact["kmeans"].predict(matrix)
    return {
        "split": split_name,
        "rows": int(len(eval_frame)),
        "skipped_rows": int(skipped),
        "silhouette": sampled_silhouette(matrix, clusters, args.k_eval_sample_size, args.random_state),
        "question_type_purity": cluster_purity(clusters, eval_frame["question_type"]),
    }


def evaluate_answer_split(split_name, frame, artifact, args):
    eval_options = to_option_long(frame, args.max_eval_questions, args.random_state + 23)
    if eval_options.empty:
        return {
            "split": split_name,
            "option_rows": 0,
            "silhouette": 0.0,
            "correctness_purity": 0.0,
            "cluster_prior_answer_accuracy": 0.0,
        }
    matrix = build_option_cluster_matrix(eval_options, artifact)
    clusters = artifact["kmeans"].predict(matrix)
    distances = artifact["kmeans"].transform(matrix)
    profiles = artifact["profiles"]
    prior_scores = []
    for row_index, cluster_id in enumerate(clusters):
        profile = profiles.get(str(int(cluster_id)), {})
        correct_rate = float(profile.get("correct_rate", 0.25))
        prior_scores.append(correct_rate - 0.01 * float(distances[row_index, int(cluster_id)]))

    scored = eval_options[["qid", "option_letter", "answer"]].copy()
    scored["cluster_prior_score"] = prior_scores
    best = scored.sort_values(["qid", "cluster_prior_score"], ascending=[True, False]).drop_duplicates("qid")
    accuracy = float((best["option_letter"] == best["answer"]).mean())
    return {
        "split": split_name,
        "questions": int(eval_options["qid"].nunique()),
        "option_rows": int(len(eval_options)),
        "silhouette": sampled_silhouette(matrix, clusters, args.k_eval_sample_size, args.random_state),
        "correctness_purity": cluster_purity(clusters, eval_options["label"]),
        "cluster_prior_answer_accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Train unsupervised/semi-supervised Model A clustering artifacts.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_UNSUPERVISED_DIR))
    parser.add_argument("--max-question-train-rows", type=int, default=50000)
    parser.add_argument("--balance-question-types", action="store_true")
    parser.add_argument("--max-question-type-rows", type=int, default=2500)
    parser.add_argument("--max-answer-train-questions", type=int, default=20000)
    parser.add_argument("--max-eval-questions", type=int, default=2500)
    parser.add_argument("--question-max-features", type=int, default=30000)
    parser.add_argument("--answer-max-features", type=int, default=30000)
    parser.add_argument("--svd-components", type=int, default=80)
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--k-eval-sample-size", type=int, default=5000)
    parser.add_argument("--dendrogram-sample-size", type=int, default=450)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RACE CSV files...", flush=True)
    train_frame = pd.read_csv(args.train_csv)
    dev_frame = pd.read_csv(args.dev_csv)
    test_frame = pd.read_csv(args.test_csv)

    question_artifact, _question_train, question_train_metrics = fit_question_cluster(
        train_frame,
        args,
        output_dir,
    )
    answer_artifact, _answer_train, answer_train_metrics = fit_answer_cluster(
        train_frame,
        args,
        output_dir,
    )

    question_eval = [
        {"split": "train", **question_train_metrics},
        evaluate_question_split("dev", dev_frame, question_artifact, args),
        evaluate_question_split("test", test_frame, question_artifact, args),
    ]
    answer_eval = [
        {"split": "train", **answer_train_metrics},
        evaluate_answer_split("dev", dev_frame, answer_artifact, args),
        evaluate_answer_split("test", test_frame, answer_artifact, args),
    ]

    pd.DataFrame(question_eval).to_csv(output_dir / "question_cluster_eval.csv", index=False)
    pd.DataFrame(answer_eval).to_csv(output_dir / "answer_cluster_eval.csv", index=False)

    artifacts = {
        "question_cluster": question_artifact,
        "answer_cluster": answer_artifact,
        "summary": {
            "question_cluster_eval": question_eval,
            "answer_cluster_eval": answer_eval,
            "artifact_file": str(output_dir / UNSUPERVISED_ARTIFACT_FILE),
        },
    }
    joblib.dump(artifacts, output_dir / UNSUPERVISED_ARTIFACT_FILE)

    with open(output_dir / "unsupervised_summary.json", "w", encoding="utf-8") as file:
        json.dump(artifacts["summary"], file, indent=2)

    print(json.dumps(artifacts["summary"], indent=2), flush=True)
    print(f"Saved unsupervised artifacts to {output_dir / UNSUPERVISED_ARTIFACT_FILE}", flush=True)


if __name__ == "__main__":
    main()
