import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import paired_cosine_distances


OPTION_COLUMNS = ["A", "B", "C", "D"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNSUPERVISED_DIR = PROJECT_ROOT / "models" / "model_a" / "traditional" / "unsupervised"
UNSUPERVISED_ARTIFACT_FILE = "model_a_unsupervised.joblib"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def jaccard_overlap(left_text, right_text):
    left_tokens = set(tokenize(left_text))
    right_tokens = set(tokenize(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def safe_cosine(left_matrix, right_matrix):
    values = 1 - paired_cosine_distances(left_matrix, right_matrix)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def option_cluster_texts(df_long):
    return (
        df_long["article"].fillna("").astype(str).str.slice(0, 1400)
        + " [QUESTION] "
        + df_long["question"].fillna("").astype(str)
        + " [OPTION] "
        + df_long["option"].fillna("").astype(str)
        + " [LETTER] "
        + df_long["option_letter"].fillna("").astype(str)
    ).tolist()


def question_cluster_text(question, answer, answer_type, source_sentence, article):
    return (
        f"{answer_type} [QUESTION] {question} "
        f"[ANSWER] {answer}"
    )


def candidate_cluster_text(candidate, article, question=""):
    return question_cluster_text(
        question,
        candidate.get("answer", ""),
        candidate.get("answer_type", "fact"),
        candidate.get("sentence", ""),
        article,
    )


def option_dense_features(df_long, vectorizer):
    question = vectorizer.transform(df_long["question"].fillna("").astype(str))
    option = vectorizer.transform(df_long["option"].fillna("").astype(str))
    article = vectorizer.transform(df_long["article"].fillna("").astype(str))

    qo_sim = safe_cosine(question, option)
    ao_sim = safe_cosine(article, option)
    qa_sim = safe_cosine(question, article)

    option_words = df_long["option"].fillna("").map(lambda text: len(tokenize(text))).to_numpy()
    question_words = df_long["question"].fillna("").map(lambda text: len(tokenize(text))).to_numpy()
    article_words = df_long["article"].fillna("").map(lambda text: len(tokenize(text))).to_numpy()

    article_option_overlap = np.fromiter(
        (
            jaccard_overlap(article_text, option_text)
            for article_text, option_text in zip(df_long["article"], df_long["option"])
        ),
        dtype=np.float32,
        count=len(df_long),
    )
    question_option_overlap = np.fromiter(
        (
            jaccard_overlap(question_text, option_text)
            for question_text, option_text in zip(df_long["question"], df_long["option"])
        ),
        dtype=np.float32,
        count=len(df_long),
    )
    option_position = pd.get_dummies(df_long["option_letter"]).reindex(columns=OPTION_COLUMNS, fill_value=0)

    dense = np.column_stack(
        [
            qo_sim,
            ao_sim,
            qa_sim,
            article_option_overlap,
            question_option_overlap,
            np.log1p(option_words),
            np.log1p(question_words),
            np.log1p(article_words),
            option_words / np.maximum(question_words, 1),
            option_words / np.maximum(article_words, 1),
            option_position.to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)
    return csr_matrix(dense)


def build_option_cluster_matrix(df_long, artifact):
    vectorizer = artifact["vectorizer"]
    text = vectorizer.transform(option_cluster_texts(df_long))
    dense = option_dense_features(df_long, vectorizer)
    features = hstack([text, dense]).tocsr()
    return artifact["svd"].transform(features)


def build_candidate_cluster_matrix(candidate, article, artifact, question=""):
    vectorizer = artifact["vectorizer"]
    text = vectorizer.transform([candidate_cluster_text(candidate, article, question)])
    return artifact["svd"].transform(text)


@lru_cache(maxsize=2)
def load_unsupervised_artifacts(model_dir=None):
    model_path = Path(model_dir) if model_dir else DEFAULT_UNSUPERVISED_DIR
    artifact_path = model_path / UNSUPERVISED_ARTIFACT_FILE
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)


def predict_question_cluster(candidate, article, question="", artifacts=None):
    artifacts = artifacts or load_unsupervised_artifacts()
    if not artifacts or "question_cluster" not in artifacts:
        return None
    artifact = artifacts["question_cluster"]
    matrix = build_candidate_cluster_matrix(candidate, article, artifact, question)
    kmeans = artifact["kmeans"]
    cluster_id = int(kmeans.predict(matrix)[0])
    distances = kmeans.transform(matrix)[0]
    profile = artifact.get("profiles", {}).get(str(cluster_id), {})
    return {
        "cluster_id": cluster_id,
        "distance": float(distances[cluster_id]),
        "dominant_question_type": profile.get("dominant_question_type", ""),
        "question_type_purity": float(profile.get("question_type_purity", 0.0)),
        "top_terms": profile.get("top_terms", []),
    }


def predict_answer_clusters(df_long, artifacts=None):
    artifacts = artifacts or load_unsupervised_artifacts()
    if not artifacts or "answer_cluster" not in artifacts:
        return []
    artifact = artifacts["answer_cluster"]
    matrix = build_option_cluster_matrix(df_long, artifact)
    kmeans = artifact["kmeans"]
    clusters = kmeans.predict(matrix)
    distances = kmeans.transform(matrix)
    rows = []
    profiles = artifact.get("profiles", {})
    for row_index, cluster_id in enumerate(clusters):
        cluster_id = int(cluster_id)
        profile = profiles.get(str(cluster_id), {})
        rows.append(
            {
                "cluster_id": cluster_id,
                "distance": float(distances[row_index, cluster_id]),
                "cluster_correct_rate": float(profile.get("correct_rate", 0.25)),
                "cluster_purity": float(profile.get("correctness_purity", 0.0)),
                "dominant_label": int(profile.get("dominant_label", 0)),
                "top_terms": profile.get("top_terms", []),
            }
        )
    return rows
