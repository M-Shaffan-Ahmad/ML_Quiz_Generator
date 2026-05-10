import re

import joblib
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.svm import LinearSVC


OPTION_COLUMNS = ["A", "B", "C", "D"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


class NoWordNet:
    def synsets(self, *args, **kwargs):
        return []


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def jaccard_overlap(left_text, right_text):
    left_tokens = set(tokenize(left_text))
    right_tokens = set(tokenize(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def lcs_length(left_tokens, right_tokens):
    previous = [0] * (len(right_tokens) + 1)
    for left_token in left_tokens:
        current = [0]
        for idx, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
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
        col
        for col in [
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
        if col in wide_df.columns
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


def get_sparse_parts(df_long, vectorizer):
    q = vectorizer.transform(df_long["question"].fillna("").astype(str))
    o = vectorizer.transform(df_long["option"].fillna("").astype(str))
    a = vectorizer.transform(df_long["article"].fillna("").astype(str))
    return q, o, a


def build_dense_handcrafted_features(df_long, q, o, a):
    qo_sim = 1 - paired_cosine_distances(q, o)
    ao_sim = 1 - paired_cosine_distances(a, o)
    qa_sim = 1 - paired_cosine_distances(q, a)

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


def build_features(df_long, ohe_vectorizer, tfidf_vectorizer):
    ohe = ohe_vectorizer.transform(df_long["combined_text"])
    question, option, article = get_sparse_parts(df_long, tfidf_vectorizer)
    dense = build_dense_handcrafted_features(df_long, question, option, article)
    return hstack([ohe, dense]).tocsr()


def choose_predictions(wide_df, long_df, scores):
    ranked = long_df[["qid", "option_letter"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates("qid")
    wide_by_qid = wide_df.set_index("qid")

    rows = []
    for qid, predicted_letter in zip(ranked["qid"], ranked["option_letter"]):
        true_letter = wide_by_qid.at[qid, "answer"]
        rows.append(
            {
                "qid": qid,
                "true_letter": true_letter,
                "predicted_letter": predicted_letter,
                "true_answer_text": wide_by_qid.at[qid, true_letter],
                "predicted_answer_text": wide_by_qid.at[qid, predicted_letter],
            }
        )
    return pd.DataFrame(rows)


def evaluate_scores(model_name, split_name, wide_df, long_df, scores):
    predictions = choose_predictions(wide_df, long_df, scores)
    metrics = compute_text_metrics(
        predictions["true_answer_text"],
        predictions["predicted_answer_text"],
    )
    metrics["Exact Match Diagnostic"] = (
        predictions["true_letter"] == predictions["predicted_letter"]
    ).mean()
    return {"model": model_name, "split": split_name, **metrics}


def main():
    print("Loading data and saved vectorizers/models...", flush=True)
    train_df = pd.read_csv("data/raw/train.csv").reset_index(drop=True)
    dev_df = pd.read_csv("data/raw/dev.csv").reset_index(drop=True)
    test_df = pd.read_csv("data/raw/test.csv").reset_index(drop=True)

    train_df["qid"] = np.arange(len(train_df))
    dev_df["qid"] = np.arange(len(dev_df))
    test_df["qid"] = np.arange(len(test_df))

    train_long = to_long(train_df)
    dev_long = to_long(dev_df)
    test_long = to_long(test_df)

    ohe_vectorizer = joblib.load("models/model_a/traditional/ohe_vectorizer_final.joblib")
    tfidf_vectorizer = joblib.load("models/model_a/traditional/tfidf_vectorizer_final.joblib")
    lr_model = joblib.load("models/model_a/traditional/lr_model_final.joblib")
    xgb_model = joblib.load("models/model_a/traditional/xgb_model_final.joblib")
    blend_weights = joblib.load("models/model_a/traditional/blend_weights_final.joblib")

    print("Building matrices...", flush=True)
    x_train = build_features(train_long, ohe_vectorizer, tfidf_vectorizer)
    x_dev = build_features(dev_long, ohe_vectorizer, tfidf_vectorizer)
    x_test = build_features(test_long, ohe_vectorizer, tfidf_vectorizer)
    y_train = train_long["label"].values

    print("Training Linear SVM for comparison...", flush=True)
    svm_model = LinearSVC(
        C=0.5,
        class_weight="balanced",
        random_state=42,
        max_iter=3000,
        dual="auto",
    )
    svm_model.fit(x_train, y_train)
    joblib.dump(svm_model, "models/model_a/traditional/svm_model_a_final.joblib")

    results = []
    for split_name, wide_df, long_df, x_split in [
        ("dev", dev_df, dev_long, x_dev),
        ("test", test_df, test_long, x_test),
    ]:
        print(f"Evaluating {split_name}...", flush=True)
        lr_scores = lr_model.predict_proba(x_split)[:, 1]
        xgb_scores = xgb_model.predict_proba(x_split)[:, 1]
        svm_scores = svm_model.decision_function(x_split)
        blend_scores = (
            blend_weights["blend_w_xgb"] * xgb_scores
            + blend_weights["blend_w_lr"] * lr_scores
        )

        results.append(evaluate_scores("Logistic Regression", split_name, wide_df, long_df, lr_scores))
        results.append(evaluate_scores("Linear SVM", split_name, wide_df, long_df, svm_scores))
        results.append(evaluate_scores("XGBoost", split_name, wide_df, long_df, xgb_scores))
        results.append(evaluate_scores("LR + XGBoost Ensemble", split_name, wide_df, long_df, blend_scores))

    results_df = pd.DataFrame(results)
    results_df.to_csv("bts/results/model_a_supervised_comparison.csv", index=False)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
