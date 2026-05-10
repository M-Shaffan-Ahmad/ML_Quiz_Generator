import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import paired_cosine_distances


OPTION_COLUMNS = ["A", "B", "C", "D"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


class NoWordNet:
    """Keep METEOR offline by skipping WordNet synonym matching."""

    def synsets(self, *args, **kwargs):
        return []


def tokenize_for_metrics(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def jaccard_overlap(left_text, right_text):
    left_tokens = set(tokenize_for_metrics(left_text))
    right_tokens = set(tokenize_for_metrics(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


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
    references = [tokenize_for_metrics(text) for text in reference_texts]
    hypotheses = [tokenize_for_metrics(text) for text in hypothesis_texts]

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
    long_df["label"] = (long_df["answer"] == long_df["option_letter"]).astype(int)
    long_df["combined_text"] = (
        long_df["article"].fillna("").astype(str)
        + " "
        + long_df["article"].fillna("").astype(str)
        + " "
        + long_df["question"].fillna("").astype(str)
        + " "
        + long_df["option"].fillna("").astype(str)
    )
    return long_df


def sparse_parts(df_long, vectorizer):
    question = vectorizer.transform(df_long["question"].fillna("").astype(str))
    option = vectorizer.transform(df_long["option"].fillna("").astype(str))
    article = vectorizer.transform(df_long["article"].fillna("").astype(str))
    return question, option, article


def question_relative_features(df_long, metric_map):
    if "qid" in df_long.columns:
        group_key = df_long["qid"].to_numpy()
    else:
        group_key = np.repeat(np.arange(int(np.ceil(len(df_long) / len(OPTION_COLUMNS)))), len(OPTION_COLUMNS))[: len(df_long)]

    relative_parts = []
    for values in metric_map.values():
        values = np.asarray(values, dtype=np.float32)
        series = pd.Series(values)
        grouped = series.groupby(group_key, sort=False)
        group_count = grouped.transform("count").to_numpy(dtype=np.float32)
        group_mean = grouped.transform("mean").to_numpy(dtype=np.float32)
        group_sum = grouped.transform("sum").to_numpy(dtype=np.float32)
        group_max = grouped.transform("max").to_numpy(dtype=np.float32)
        group_min = grouped.transform("min").to_numpy(dtype=np.float32)
        group_std = grouped.transform("std").fillna(0.0).to_numpy(dtype=np.float32)
        mean_other = (group_sum - values) / np.maximum(group_count - 1.0, 1.0)
        rank_desc = grouped.rank(method="min", ascending=False).to_numpy(dtype=np.float32)
        normalized_rank = 1.0 - ((rank_desc - 1.0) / np.maximum(group_count - 1.0, 1.0))
        relative_parts.extend(
            [
                values - mean_other,
                values - group_mean,
                values - group_max,
                values - group_min,
                normalized_rank,
                (values == group_max).astype(np.float32),
                (values - group_mean) / np.maximum(group_std, 1e-6),
            ]
        )
    return np.column_stack(relative_parts).astype(np.float32)


def dense_handcrafted_features(df_long, question, option, article):
    qo_sim = 1 - paired_cosine_distances(question, option)
    ao_sim = 1 - paired_cosine_distances(article, option)
    qa_sim = 1 - paired_cosine_distances(question, article)

    option_words = df_long["option"].fillna("").map(lambda text: len(tokenize_for_metrics(text))).to_numpy()
    question_words = df_long["question"].fillna("").map(lambda text: len(tokenize_for_metrics(text))).to_numpy()
    article_words = df_long["article"].fillna("").map(lambda text: len(tokenize_for_metrics(text))).to_numpy()
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
    relative = question_relative_features(
        df_long,
        {
            "question_option_cosine": qo_sim,
            "article_option_cosine": ao_sim,
            "article_option_overlap": article_option_overlap,
            "question_option_overlap": question_option_overlap,
            "option_words": option_words,
        },
    )

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
            relative,
        ]
    ).astype(np.float32)
    return csr_matrix(dense)


def choose_predictions(wide_df, long_df, option_scores):
    ranked = long_df[["qid", "option_letter"]].copy()
    ranked["score"] = option_scores
    ranked = ranked.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates("qid")

    wide_by_qid = wide_df.set_index("qid")
    rows = []
    for qid, predicted_letter, score in zip(ranked["qid"], ranked["option_letter"], ranked["score"]):
        true_letter = wide_by_qid.at[qid, "answer"]
        rows.append(
            {
                "qid": qid,
                "question_id": wide_by_qid.at[qid, "question_id"],
                "level": wide_by_qid.at[qid, "level"],
                "true_letter": true_letter,
                "predicted_letter": predicted_letter,
                "score": score,
                "true_answer_text": wide_by_qid.at[qid, true_letter],
                "predicted_answer_text": wide_by_qid.at[qid, predicted_letter],
            }
        )
    return pd.DataFrame(rows)


def main():
    print("--- Model A: Logistic Regression + XGBoost Blend ---")
    print("Loading preprocessed CSVs...")
    train_df = pd.read_csv("data/raw/train.csv").reset_index(drop=True)
    dev_df = pd.read_csv("data/raw/dev.csv").reset_index(drop=True)
    test_df = pd.read_csv("data/raw/test.csv").reset_index(drop=True)

    train_df["qid"] = np.arange(len(train_df))
    dev_df["qid"] = np.arange(len(dev_df))
    test_df["qid"] = np.arange(len(test_df))

    train_long = to_long(train_df)
    dev_long = to_long(dev_df)
    test_long = to_long(test_df)

    print(f"Train questions: {len(train_df)} | option rows: {len(train_long)}")
    print(f"Dev questions:   {len(dev_df)} | option rows: {len(dev_long)}")
    print(f"Test questions:  {len(test_df)} | option rows: {len(test_long)}")

    print("Building CountVectorizer features...")
    ohe_vectorizer = CountVectorizer(
        binary=True,
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    train_ohe = ohe_vectorizer.fit_transform(train_long["combined_text"])
    dev_ohe = ohe_vectorizer.transform(dev_long["combined_text"])
    test_ohe = ohe_vectorizer.transform(test_long["combined_text"])

    print("Building TF-IDF, cosine, lexical-overlap, length, and position features...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    tfidf_vectorizer.fit(train_long["combined_text"])

    q_train, o_train, a_train = sparse_parts(train_long, tfidf_vectorizer)
    q_dev, o_dev, a_dev = sparse_parts(dev_long, tfidf_vectorizer)
    q_test, o_test, a_test = sparse_parts(test_long, tfidf_vectorizer)

    train_dense = dense_handcrafted_features(train_long, q_train, o_train, a_train)
    dev_dense = dense_handcrafted_features(dev_long, q_dev, o_dev, a_dev)
    test_dense = dense_handcrafted_features(test_long, q_test, o_test, a_test)

    x_train = hstack([train_ohe, train_dense]).tocsr()
    x_dev = hstack([dev_ohe, dev_dense]).tocsr()
    x_test = hstack([test_ohe, test_dense]).tocsr()
    y_train = train_long["label"].values

    print(f"Final train matrix: {x_train.shape}")

    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1500,
        random_state=42,
        class_weight="balanced",
    )
    lr_model.fit(x_train, y_train)

    print("Training XGBoost...")
    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=neg_count / max(pos_count, 1),
        n_jobs=-1,
    )
    xgb_model.fit(x_train, y_train)

    blend_w_xgb = 0.8
    blend_w_lr = 0.2
    results = []

    for split_name, wide_df, long_df, features in [
        ("dev", dev_df, dev_long, x_dev),
        ("test", test_df, test_long, x_test),
    ]:
        print(f"Scoring {split_name}...")
        lr_scores = lr_model.predict_proba(features)[:, 1]
        xgb_scores = xgb_model.predict_proba(features)[:, 1]
        blend_scores = blend_w_xgb * xgb_scores + blend_w_lr * lr_scores

        predictions = choose_predictions(wide_df, long_df, blend_scores)
        metrics = compute_text_metrics(
            predictions["true_answer_text"],
            predictions["predicted_answer_text"],
        )
        metrics["Exact Match Diagnostic"] = (
            predictions["true_letter"] == predictions["predicted_letter"]
        ).mean()
        results.append({"split": split_name, **metrics})
        predictions.to_csv(f"bts/results/model_a_predictions_{split_name}.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv("bts/results/model_a_eval_results.csv", index=False)

    print("\nModel A evaluation results")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nSaving trained artifacts...")
    joblib.dump(ohe_vectorizer, "models/model_a/traditional/ohe_vectorizer_final.joblib")
    joblib.dump(tfidf_vectorizer, "models/model_a/traditional/tfidf_vectorizer_final.joblib")
    joblib.dump(lr_model, "models/model_a/traditional/lr_model_final.joblib")
    joblib.dump(xgb_model, "models/model_a/traditional/xgb_model_final.joblib")
    joblib.dump(
        {"blend_w_xgb": blend_w_xgb, "blend_w_lr": blend_w_lr},
        "models/model_a/traditional/blend_weights_final.joblib",
    )
    print("Complete.")


if __name__ == "__main__":
    main()
