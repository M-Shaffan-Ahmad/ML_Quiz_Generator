import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import paired_cosine_distances

from model_a_unsupervised import load_unsupervised_artifacts, predict_answer_clusters


OPTION_COLUMNS = ["A", "B", "C", "D"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSICAL_DIR = PROJECT_ROOT / "models" / "model_a" / "traditional"
DEFAULT_BERT_CHECKPOINT = PROJECT_ROOT / "models" / "model_a" / "neural" / "bert_mc_30000q_plus_new30000q"


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def jaccard_overlap(left_text, right_text):
    left_tokens = set(tokenize(left_text))
    right_tokens = set(tokenize(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def build_candidate_frame(article, question, options):
    row = {
        "qid": 0,
        "example_id": "manual_input",
        "question_id": "manual_input__q0",
        "split": "manual",
        "level": "manual",
        "article": article,
        "question": question,
        "answer": "A",
        "correct_answer_text": options.get("A", ""),
        **{label: options.get(label, "") for label in OPTION_COLUMNS},
    }
    wide_df = pd.DataFrame([row])
    long_df = wide_df.melt(
        id_vars=[
            "qid",
            "example_id",
            "question_id",
            "split",
            "level",
            "article",
            "question",
            "answer",
            "correct_answer_text",
        ],
        value_vars=OPTION_COLUMNS,
        var_name="option_letter",
        value_name="option",
    )
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


def dual_resolver_scores(long_df, correct_scores, wrong_scores, config):
    qids = long_df["qid"].to_numpy()
    correct_norm = minmax_by_question(correct_scores, qids)
    wrong_norm = minmax_by_question(wrong_scores, qids)
    correct_rank, correct_rank_score = ranks_by_question(correct_scores, qids, ascending=False)
    low_wrong_rank, low_wrong_rank_score = ranks_by_question(wrong_scores, qids, ascending=True)
    high_wrong_rank, _ = ranks_by_question(wrong_scores, qids, ascending=False)

    candidate_k = int(config.get("candidate_k", 2))
    agrees = ((correct_rank <= candidate_k) & (low_wrong_rank <= candidate_k)).astype(np.float32)
    conflicts = ((correct_rank <= candidate_k) & (high_wrong_rank <= candidate_k)).astype(np.float32)

    return (
        correct_norm
        - float(config.get("wrong_penalty", 1.0)) * wrong_norm
        + float(config.get("rank_weight", 0.0)) * (correct_rank_score + low_wrong_rank_score)
        + float(config.get("agreement_bonus", 0.0)) * agrees
        - float(config.get("conflict_penalty", 0.0)) * conflicts
    )


def dense_handcrafted_features(df_long, question, option, article):
    qo_sim = 1 - paired_cosine_distances(question, option)
    ao_sim = 1 - paired_cosine_distances(article, option)
    qa_sim = 1 - paired_cosine_distances(question, article)

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


class ClassicalModelA:
    def __init__(
        self,
        model_dir=".",
        lr_weight=0.2,
        xgb_weight=0.8,
        unsupervised_weight=0.0,
        use_dual_ranker=False,
    ):
        model_dir = Path(model_dir)
        self.ohe_vectorizer = joblib.load(model_dir / "ohe_vectorizer_final.joblib")
        self.tfidf_vectorizer = joblib.load(model_dir / "tfidf_vectorizer_final.joblib")
        self.lr_model = joblib.load(model_dir / "lr_model_final.joblib")
        self.xgb_model = joblib.load(model_dir / "xgb_model_final.joblib")
        self.lr_weight = lr_weight
        self.xgb_weight = xgb_weight
        self.unsupervised_weight = float(unsupervised_weight)
        self.unsupervised_artifacts = load_unsupervised_artifacts()
        self.use_dual_ranker = bool(use_dual_ranker)
        self.lr_wrong_model = None
        self.xgb_wrong_model = None
        self.dual_ranker_config = None
        if self.use_dual_ranker:
            self.lr_wrong_model = joblib.load(model_dir / "lr_wrong_model_final.joblib")
            self.xgb_wrong_model = joblib.load(model_dir / "xgb_wrong_model_final.joblib")
            self.dual_ranker_config = joblib.load(model_dir / "dual_ranker_config.joblib")

    def build_features(self, df_long):
        ohe = self.ohe_vectorizer.transform(df_long["combined_text"])
        question, option, article = get_sparse_parts(df_long, self.tfidf_vectorizer)
        dense = dense_handcrafted_features(df_long, question, option, article)
        return hstack([ohe, dense]).tocsr()

    def predict(self, article, question, options):
        df_long = build_candidate_frame(article, question, options)
        features = self.build_features(df_long)
        lr_scores = self.lr_model.predict_proba(features)[:, 1]
        xgb_scores = self.xgb_model.predict_proba(features)[:, 1]
        supervised_scores = self.lr_weight * lr_scores + self.xgb_weight * xgb_scores
        scores = supervised_scores.copy()
        wrong_scores = None
        raw_dual_scores = None
        if self.use_dual_ranker:
            wrong_blend = self.dual_ranker_config.get("wrong_blend", {"lr": 0.2, "xgb": 0.8})
            wrong_lr_scores = self.lr_wrong_model.predict_proba(features)[:, 1]
            wrong_xgb_scores = self.xgb_wrong_model.predict_proba(features)[:, 1]
            wrong_scores = (
                float(wrong_blend.get("lr", 0.2)) * wrong_lr_scores
                + float(wrong_blend.get("xgb", 0.8)) * wrong_xgb_scores
            )
            raw_dual_scores = dual_resolver_scores(df_long, supervised_scores, wrong_scores, self.dual_ranker_config)
            scores = minmax_by_question(raw_dual_scores, df_long["qid"].to_numpy())
        cluster_rows = predict_answer_clusters(df_long, self.unsupervised_artifacts)
        if cluster_rows:
            cluster_prior = np.asarray(
                [row["cluster_correct_rate"] - 0.01 * row["distance"] for row in cluster_rows],
                dtype=np.float32,
            )
            if self.unsupervised_weight > 0:
                weight = min(max(self.unsupervised_weight, 0.0), 1.0)
                scores = (1.0 - weight) * supervised_scores + weight * cluster_prior
        df_long = df_long[["option_letter", "option"]].copy()
        df_long["score"] = scores
        df_long["supervised_score"] = supervised_scores
        best = df_long.sort_values("score", ascending=False).iloc[0]
        result = {
            "backend": "classical",
            "predicted_letter": best["option_letter"],
            "predicted_answer_text": best["option"],
            "scores": {
                row["option_letter"]: float(row["score"])
                for _, row in df_long.iterrows()
            },
        }
        if self.unsupervised_weight > 0:
            result["backend"] = "classical+unsupervised-cluster-prior"
            result["supervised_scores"] = {
                row["option_letter"]: float(row["supervised_score"])
                for _, row in df_long.iterrows()
            }
            result["unsupervised_weight"] = self.unsupervised_weight
        if self.use_dual_ranker:
            result["backend"] = "classical-dual-correct-wrong-ranker"
            result["correct_scores"] = {
                row["option_letter"]: float(row["supervised_score"])
                for _, row in df_long.iterrows()
            }
            result["wrong_scores"] = {
                row["option_letter"]: float(score)
                for (_, row), score in zip(df_long.iterrows(), wrong_scores)
            }
            result["dual_raw_scores"] = {
                row["option_letter"]: float(score)
                for (_, row), score in zip(df_long.iterrows(), raw_dual_scores)
            }
            result["dual_ranker_config"] = self.dual_ranker_config
        if cluster_rows:
            result["unsupervised_clusters"] = {
                label: {
                    "cluster_id": int(cluster["cluster_id"]),
                    "cluster_correct_rate": float(cluster["cluster_correct_rate"]),
                    "cluster_purity": float(cluster["cluster_purity"]),
                    "distance": float(cluster["distance"]),
                }
                for label, cluster in zip(df_long["option_letter"], cluster_rows)
            }
        return result


class BertModelA:
    def __init__(self, checkpoint_dir=DEFAULT_BERT_CHECKPOINT, device=None):
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForMultipleChoice,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "BERT backend requires optional dependencies. Install them with "
                "`pip install -r bts/requirements-bert.txt` and train or place a checkpoint "
                "under models/model_a/neural/bert_mc_30000q_plus_new30000q."
            ) from exc

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"BERT checkpoint not found at {checkpoint_dir}. "
                "Run train_bert_multiple_choice_model_a.py first or provide --bert-checkpoint."
            )

        self.torch = torch
        self.config = AutoConfig.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        architectures = getattr(self.config, "architectures", []) or []
        self.is_multiple_choice = any("MultipleChoice" in architecture for architecture in architectures)
        if self.is_multiple_choice:
            self.model = AutoModelForMultipleChoice.from_pretrained(checkpoint_dir)
            self.model_kind = "bert-multiple-choice"
            self.max_length = 256
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
            self.model_kind = "bert-sequence-verifier"
            self.max_length = 384
        self.device = self.resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    def resolve_device(self, device):
        requested = str(device or "auto").strip().lower()
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for BERT, but PyTorch cannot access a CUDA device. "
                "Check the NVIDIA driver, CUDA-compatible torch build, or use --device auto/cpu."
            )
        return requested

    def predict(self, article, question, options):
        if self.is_multiple_choice:
            return self.predict_multiple_choice(article, question, options)
        return self.predict_sequence_verifier(article, question, options)

    def predict_multiple_choice(self, article, question, options):
        pairs = [
            (
                label,
                options.get(label, ""),
                f"{question} [SEP] {options.get(label, '')}",
            )
            for label in OPTION_COLUMNS
        ]
        encoded = self.tokenizer(
            [article for _label, _option, _text in pairs],
            [text for _label, _option, text in pairs],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.unsqueeze(0).to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(0)
            probabilities = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()

        best_index = int(np.argmax(probabilities))
        best_label, best_text, _ = pairs[best_index]
        return {
            "backend": self.model_kind,
            "device": self.device,
            "predicted_letter": best_label,
            "predicted_answer_text": best_text,
            "scores": {
                label: float(score)
                for (label, _option, _text), score in zip(pairs, probabilities)
            },
        }

    def predict_sequence_verifier(self, article, question, options):
        pairs = [
            (
                label,
                options.get(label, ""),
                f"{question} [SEP] {options.get(label, '')}",
            )
            for label in OPTION_COLUMNS
        ]
        encoded = self.tokenizer(
            [article for _label, _option, _text in pairs],
            [text for _label, _option, text in pairs],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = self.torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        best_index = int(np.argmax(probabilities))
        best_label, best_text, _ = pairs[best_index]
        return {
            "backend": self.model_kind,
            "device": self.device,
            "predicted_letter": best_label,
            "predicted_answer_text": best_text,
            "scores": {
                label: float(score)
                for (label, _option, _text), score in zip(pairs, probabilities)
            },
        }


def load_backend(
    name,
    classical_dir=DEFAULT_CLASSICAL_DIR,
    bert_checkpoint=DEFAULT_BERT_CHECKPOINT,
    device=None,
    unsupervised_weight=0.0,
    use_dual_ranker=False,
):
    if name == "classical":
        return ClassicalModelA(
            model_dir=classical_dir,
            unsupervised_weight=unsupervised_weight,
            use_dual_ranker=use_dual_ranker,
        )
    if name == "bert":
        return BertModelA(checkpoint_dir=bert_checkpoint, device=device)
    if name == "auto":
        try:
            return BertModelA(checkpoint_dir=bert_checkpoint, device=device)
        except Exception:
            return ClassicalModelA(
                model_dir=classical_dir,
                unsupervised_weight=unsupervised_weight,
                use_dual_ranker=use_dual_ranker,
            )
    raise ValueError(f"Unknown backend: {name}")


def parse_options(raw_options):
    if Path(raw_options).exists():
        data = json.loads(Path(raw_options).read_text(encoding="utf-8"))
    else:
        data = json.loads(raw_options)
    return {label: str(data.get(label, "")) for label in OPTION_COLUMNS}


def main():
    parser = argparse.ArgumentParser(description="Run Model A with classical ML or optional BERT backend.")
    parser.add_argument("--backend", choices=["classical", "bert", "auto"], default="classical")
    parser.add_argument("--article", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--options",
        required=True,
        help='JSON string or path containing {"A": "...", "B": "...", "C": "...", "D": "..."}',
    )
    parser.add_argument("--classical-dir", default=str(DEFAULT_CLASSICAL_DIR))
    parser.add_argument("--bert-checkpoint", default=str(DEFAULT_BERT_CHECKPOINT))
    parser.add_argument("--device", default="auto", help="BERT device: auto, cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--unsupervised-weight",
        type=float,
        default=0.0,
        help="Optional classical score blend weight for the K-Means cluster prior. Default keeps final supervised scores unchanged.",
    )
    parser.add_argument(
        "--use-dual-ranker",
        action="store_true",
        help="Use the experimental dual correct/wrong classical resolver if wrong-ranker artifacts are available.",
    )
    args = parser.parse_args()

    try:
        model = load_backend(
            args.backend,
            args.classical_dir,
            args.bert_checkpoint,
            args.device,
            unsupervised_weight=args.unsupervised_weight,
            use_dual_ranker=args.use_dual_ranker,
        )
        result = model.predict(args.article, args.question, parse_options(args.options))
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
