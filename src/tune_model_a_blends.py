import itertools
import re

import joblib
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from scipy.special import expit
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import paired_cosine_distances


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


def normalize_by_question(scores, qids):
    normalized = np.zeros_like(scores, dtype=np.float32)
    qid_array = np.asarray(qids)
    for qid in np.unique(qid_array):
        mask = qid_array == qid
        group = scores[mask].astype(np.float32)
        min_score = group.min()
        max_score = group.max()
        if max_score > min_score:
            normalized[mask] = (group - min_score) / (max_score - min_score)
        else:
            normalized[mask] = 0.0
    return normalized


def predictions_from_scores(wide_df, long_df, scores):
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


def exact_match_for_scores(wide_df, long_df, scores):
    predictions = predictions_from_scores(wide_df, long_df, scores)
    return (predictions["true_letter"] == predictions["predicted_letter"]).mean()


def full_metrics_for_scores(wide_df, long_df, scores):
    predictions = predictions_from_scores(wide_df, long_df, scores)
    metrics = compute_text_metrics(
        predictions["true_answer_text"],
        predictions["predicted_answer_text"],
    )
    metrics["Exact Match Diagnostic"] = (
        predictions["true_letter"] == predictions["predicted_letter"]
    ).mean()
    return metrics, predictions


def weight_grid(model_names, step=0.05):
    ticks = np.arange(0, 1 + step / 2, step)
    if len(model_names) == 2:
        for first in ticks:
            second = 1 - first
            yield dict(zip(model_names, [round(float(first), 4), round(float(second), 4)]))
    else:
        for values in itertools.product(ticks, repeat=len(model_names)):
            if abs(sum(values) - 1.0) < 1e-9:
                yield dict(zip(model_names, [round(float(value), 4) for value in values]))


def blended_scores(scores_by_model, weights):
    first_scores = next(iter(scores_by_model.values()))
    blended = np.zeros_like(first_scores, dtype=np.float32)
    for name, weight in weights.items():
        blended += weight * scores_by_model[name]
    return blended


def tune_weights(wide_df, long_df, scores_by_model, step=0.05):
    records = []
    for weights in weight_grid(list(scores_by_model), step=step):
        scores = blended_scores(scores_by_model, weights)
        exact = exact_match_for_scores(wide_df, long_df, scores)
        records.append({**weights, "dev_exact_match": exact})
    results = pd.DataFrame(records).sort_values("dev_exact_match", ascending=False).reset_index(drop=True)
    best_weights = {
        name: float(results.loc[0, name])
        for name in scores_by_model
    }
    return best_weights, results


def main():
    print("Loading optimized Model A artifacts...", flush=True)
    ohe_vectorizer = joblib.load("models/model_a/traditional/ohe_vectorizer_final.joblib")
    tfidf_vectorizer = joblib.load("models/model_a/traditional/tfidf_vectorizer_final.joblib")
    lr_model = joblib.load("models/model_a/traditional/lr_model_final.joblib")
    xgb_model = joblib.load("models/model_a/traditional/xgb_model_final.joblib")
    svm_model = joblib.load("models/model_a/traditional/svm_model_a_final.joblib")

    dev_df = pd.read_csv("data/raw/dev.csv").reset_index(drop=True)
    test_df = pd.read_csv("data/raw/test.csv").reset_index(drop=True)
    dev_df["qid"] = np.arange(len(dev_df))
    test_df["qid"] = np.arange(len(test_df))
    dev_long = to_long(dev_df)
    test_long = to_long(test_df)

    print("Building dev/test matrices...", flush=True)
    x_dev = build_features(dev_long, ohe_vectorizer, tfidf_vectorizer)
    x_test = build_features(test_long, ohe_vectorizer, tfidf_vectorizer)

    print("Scoring base models...", flush=True)
    dev_raw = {
        "lr": lr_model.predict_proba(x_dev)[:, 1],
        "xgb": xgb_model.predict_proba(x_dev)[:, 1],
        "svm": expit(svm_model.decision_function(x_dev)),
    }
    test_raw = {
        "lr": lr_model.predict_proba(x_test)[:, 1],
        "xgb": xgb_model.predict_proba(x_test)[:, 1],
        "svm": expit(svm_model.decision_function(x_test)),
    }

    dev_norm = {
        name: normalize_by_question(scores, dev_long["qid"])
        for name, scores in dev_raw.items()
    }
    test_norm = {
        name: normalize_by_question(scores, test_long["qid"])
        for name, scores in test_raw.items()
    }

    print("Tuning LR/XGB raw weights on dev exact match...", flush=True)
    best_lx_raw, grid_lx_raw = tune_weights(dev_df, dev_long, {"lr": dev_raw["lr"], "xgb": dev_raw["xgb"]})
    print("Tuning LR/XGB/SVM raw weights on dev exact match...", flush=True)
    best_lxs_raw, grid_lxs_raw = tune_weights(dev_df, dev_long, dev_raw)
    print("Tuning LR/XGB/SVM question-normalized weights on dev exact match...", flush=True)
    best_lxs_norm, grid_lxs_norm = tune_weights(dev_df, dev_long, dev_norm)

    candidates = [
        ("LR+XGB raw", best_lx_raw, {"dev": {"lr": dev_raw["lr"], "xgb": dev_raw["xgb"]}, "test": {"lr": test_raw["lr"], "xgb": test_raw["xgb"]}}),
        ("LR+XGB+SVM raw", best_lxs_raw, {"dev": dev_raw, "test": test_raw}),
        ("LR+XGB+SVM question-normalized", best_lxs_norm, {"dev": dev_norm, "test": test_norm}),
    ]

    rows = []
    for candidate_name, weights, score_sets in candidates:
        for split_name, wide_df, long_df in [("dev", dev_df, dev_long), ("test", test_df, test_long)]:
            scores = blended_scores(score_sets[split_name], weights)
            metrics, predictions = full_metrics_for_scores(wide_df, long_df, scores)
            rows.append({"candidate": candidate_name, "split": split_name, **weights, **metrics})
            if split_name == "test":
                predictions.to_csv(
                    f"model_a_predictions_tuned_{candidate_name.lower().replace('+', '_').replace(' ', '_')}.csv",
                    index=False,
                )

    summary = pd.DataFrame(rows)
    summary.to_csv("bts/results/model_a_tuned_blend_results.csv", index=False)
    grid_lx_raw.to_csv("bts/results/model_a_blend_grid_lr_xgb_raw.csv", index=False)
    grid_lxs_raw.to_csv("bts/results/model_a_blend_grid_lr_xgb_svm_raw.csv", index=False)
    grid_lxs_norm.to_csv("bts/results/model_a_blend_grid_lr_xgb_svm_normalized.csv", index=False)

    # Pick the final model by dev exact match, then report its test metrics.
    dev_rows = summary[summary["split"] == "dev"].copy()
    dev_rows["dev_selection_score"] = dev_rows["Exact Match Diagnostic"]
    final_candidate = dev_rows.sort_values("dev_selection_score", ascending=False).iloc[0]["candidate"]
    final_rows = summary[summary["candidate"] == final_candidate]
    final_rows.to_csv("bts/results/model_a_final_tuned_blend.csv", index=False)

    print("\nBlend tuning summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by dev exact match: {final_candidate}")
    print(final_rows.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
