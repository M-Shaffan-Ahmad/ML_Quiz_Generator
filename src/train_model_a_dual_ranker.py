import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

from run_model_a_evals import (
    compute_text_metrics,
    choose_predictions,
    dense_handcrafted_features,
    sparse_parts,
    to_long,
)


OPTION_COLUMNS = ["A", "B", "C", "D"]


def prepare_split(frame, ohe_vectorizer, tfidf_vectorizer):
    frame = frame.reset_index(drop=True).copy()
    frame["qid"] = np.arange(len(frame))
    long_df = to_long(frame)
    ohe = ohe_vectorizer.transform(long_df["combined_text"])
    question, option, article = sparse_parts(long_df, tfidf_vectorizer)
    dense = dense_handcrafted_features(long_df, question, option, article)
    features = hstack([ohe, dense]).tocsr()
    return frame, long_df, features


def minmax_by_question(values, qids):
    values = np.asarray(values, dtype=np.float32)
    series = pd.Series(values)
    grouped = series.groupby(qids, sort=False)
    group_min = grouped.transform("min").to_numpy(dtype=np.float32)
    group_max = grouped.transform("max").to_numpy(dtype=np.float32)
    return (values - group_min) / np.maximum(group_max - group_min, 1e-6)


def ranks_by_question(values, qids, ascending):
    series = pd.Series(np.asarray(values, dtype=np.float32))
    grouped = series.groupby(qids, sort=False)
    ranks = grouped.rank(method="min", ascending=ascending).to_numpy(dtype=np.float32)
    counts = grouped.transform("count").to_numpy(dtype=np.float32)
    normalized = 1.0 - ((ranks - 1.0) / np.maximum(counts - 1.0, 1.0))
    return ranks, normalized


def dual_resolver_scores(
    long_df,
    correct_scores,
    wrong_scores,
    wrong_penalty,
    rank_weight,
    agreement_bonus,
    conflict_penalty,
    candidate_k,
):
    qids = long_df["qid"].to_numpy()
    correct_norm = minmax_by_question(correct_scores, qids)
    wrong_norm = minmax_by_question(wrong_scores, qids)

    correct_rank, correct_rank_score = ranks_by_question(correct_scores, qids, ascending=False)
    low_wrong_rank, low_wrong_rank_score = ranks_by_question(wrong_scores, qids, ascending=True)
    high_wrong_rank, _high_wrong_rank_score = ranks_by_question(wrong_scores, qids, ascending=False)

    agrees = ((correct_rank <= candidate_k) & (low_wrong_rank <= candidate_k)).astype(np.float32)
    conflicts = ((correct_rank <= candidate_k) & (high_wrong_rank <= candidate_k)).astype(np.float32)

    return (
        correct_norm
        - wrong_penalty * wrong_norm
        + rank_weight * (correct_rank_score + low_wrong_rank_score)
        + agreement_bonus * agrees
        - conflict_penalty * conflicts
    )


def evaluate_scores(split_name, wide_df, long_df, scores, output_path=None):
    predictions = choose_predictions(wide_df, long_df, scores)
    metrics = compute_text_metrics(
        predictions["true_answer_text"],
        predictions["predicted_answer_text"],
    )
    metrics["Exact Match Diagnostic"] = (
        predictions["true_letter"] == predictions["predicted_letter"]
    ).mean()
    if output_path:
        predictions.to_csv(output_path, index=False)
    return {"split": split_name, **metrics}


def exact_match_for_scores(wide_df, long_df, scores):
    ranked = long_df[["qid", "option_letter"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(["qid", "score"], ascending=[True, False]).drop_duplicates("qid")
    answer_by_qid = wide_df.set_index("qid")["answer"]
    truth = ranked["qid"].map(answer_by_qid)
    return float((ranked["option_letter"] == truth).mean())


def train_wrong_models(features, labels, random_state):
    wrong_labels = 1 - labels
    print("Training wrong-answer Logistic Regression...", flush=True)
    lr_wrong = LogisticRegression(
        max_iter=1500,
        random_state=random_state,
        class_weight="balanced",
    )
    lr_wrong.fit(features, wrong_labels)

    print("Training wrong-answer XGBoost...", flush=True)
    pos_count = int((wrong_labels == 1).sum())
    neg_count = int((wrong_labels == 0).sum())
    xgb_wrong = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=random_state,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=neg_count / max(pos_count, 1),
        n_jobs=-1,
    )
    xgb_wrong.fit(features, wrong_labels)
    return lr_wrong, xgb_wrong


def tune_resolver(long_df, wide_df, correct_scores, wrong_scores):
    rows = []
    for wrong_penalty in [0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25]:
        for rank_weight in [0.0, 0.03, 0.06, 0.1, 0.15]:
            for agreement_bonus in [0.0, 0.03, 0.06, 0.1]:
                for conflict_penalty in [0.0, 0.03, 0.06, 0.1]:
                    for candidate_k in [1, 2, 3]:
                        scores = dual_resolver_scores(
                            long_df,
                            correct_scores,
                            wrong_scores,
                            wrong_penalty=wrong_penalty,
                            rank_weight=rank_weight,
                            agreement_bonus=agreement_bonus,
                            conflict_penalty=conflict_penalty,
                            candidate_k=candidate_k,
                        )
                        rows.append(
                            {
                                "wrong_penalty": wrong_penalty,
                                "rank_weight": rank_weight,
                                "agreement_bonus": agreement_bonus,
                                "conflict_penalty": conflict_penalty,
                                "candidate_k": candidate_k,
                                "split": "dev",
                                "Exact Match Diagnostic": exact_match_for_scores(wide_df, long_df, scores),
                            }
                        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(
        ["Exact Match Diagnostic", "wrong_penalty", "rank_weight", "agreement_bonus"],
        ascending=False,
    ).iloc[0]
    return grid, best


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a dual correct/wrong Model A ranker.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--dev-csv", default="data/raw/dev.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--model-dir", default="models/model_a/traditional")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print("Loading relative-feature classical artifacts...", flush=True)
    ohe_vectorizer = joblib.load(model_dir / "ohe_vectorizer_final.joblib")
    tfidf_vectorizer = joblib.load(model_dir / "tfidf_vectorizer_final.joblib")
    lr_correct = joblib.load(model_dir / "lr_model_final.joblib")
    xgb_correct = joblib.load(model_dir / "xgb_model_final.joblib")

    print("Loading CSV splits...", flush=True)
    train_df = pd.read_csv(args.train_csv)
    dev_df = pd.read_csv(args.dev_csv)
    test_df = pd.read_csv(args.test_csv)

    print("Building relative-feature matrices...", flush=True)
    train_wide, train_long, x_train = prepare_split(train_df, ohe_vectorizer, tfidf_vectorizer)
    dev_wide, dev_long, x_dev = prepare_split(dev_df, ohe_vectorizer, tfidf_vectorizer)
    test_wide, test_long, x_test = prepare_split(test_df, ohe_vectorizer, tfidf_vectorizer)
    labels = train_long["label"].to_numpy(dtype=np.int8)
    print(f"Train matrix: {x_train.shape}", flush=True)

    lr_wrong, xgb_wrong = train_wrong_models(x_train, labels, args.random_state)

    print("Scoring correct and wrong rankers...", flush=True)
    dev_correct = 0.2 * lr_correct.predict_proba(x_dev)[:, 1] + 0.8 * xgb_correct.predict_proba(x_dev)[:, 1]
    test_correct = 0.2 * lr_correct.predict_proba(x_test)[:, 1] + 0.8 * xgb_correct.predict_proba(x_test)[:, 1]
    dev_wrong = 0.2 * lr_wrong.predict_proba(x_dev)[:, 1] + 0.8 * xgb_wrong.predict_proba(x_dev)[:, 1]
    test_wrong = 0.2 * lr_wrong.predict_proba(x_test)[:, 1] + 0.8 * xgb_wrong.predict_proba(x_test)[:, 1]

    print("Tuning dual resolver on dev...", flush=True)
    grid, best = tune_resolver(dev_long, dev_wide, dev_correct, dev_wrong)
    grid.to_csv("bts/results/model_a_dual_ranker_grid.csv", index=False)
    config = {
        "wrong_penalty": float(best["wrong_penalty"]),
        "rank_weight": float(best["rank_weight"]),
        "agreement_bonus": float(best["agreement_bonus"]),
        "conflict_penalty": float(best["conflict_penalty"]),
        "candidate_k": int(best["candidate_k"]),
        "correct_blend": {"lr": 0.2, "xgb": 0.8},
        "wrong_blend": {"lr": 0.2, "xgb": 0.8},
    }

    rows = []
    baseline_dev = evaluate_scores("dev", dev_wide, dev_long, dev_correct)
    baseline_test = evaluate_scores("test", test_wide, test_long, test_correct)
    rows.extend(
        [
            {"model": "Relative-feature correct ranker", **baseline_dev},
            {"model": "Relative-feature correct ranker", **baseline_test},
        ]
    )

    for split_name, wide_df, long_df, correct_scores, wrong_scores in [
        ("dev", dev_wide, dev_long, dev_correct, dev_wrong),
        ("test", test_wide, test_long, test_correct, test_wrong),
    ]:
        dual_scores = dual_resolver_scores(
            long_df,
            correct_scores,
            wrong_scores,
            wrong_penalty=config["wrong_penalty"],
            rank_weight=config["rank_weight"],
            agreement_bonus=config["agreement_bonus"],
            conflict_penalty=config["conflict_penalty"],
            candidate_k=config["candidate_k"],
        )
        metrics = evaluate_scores(
            split_name,
            wide_df,
            long_df,
            dual_scores,
            output_path=f"model_a_predictions_dual_ranker_{split_name}.csv",
        )
        rows.append({"model": "Dual correct/wrong ranker", **metrics})

    results = pd.DataFrame(rows)
    results.to_csv("bts/results/model_a_dual_ranker_results.csv", index=False)

    joblib.dump(lr_wrong, model_dir / "lr_wrong_model_final.joblib")
    joblib.dump(xgb_wrong, model_dir / "xgb_wrong_model_final.joblib")
    joblib.dump(config, model_dir / "dual_ranker_config.joblib")
    with open(model_dir / "dual_ranker_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("Best resolver config:")
    print(json.dumps(config, indent=2), flush=True)
    print("\nDual-ranker results")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)


if __name__ == "__main__":
    main()
