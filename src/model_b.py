import argparse
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from model_a_generation import (
    OPTION_COLUMNS,
    answer_tokens,
    build_candidate_frame,
    clean_answer,
    find_answer_span,
    find_source_sentence,
    infer_answer_type,
    infer_question_type,
    jaccard_overlap,
    normalize_text,
    split_sentences,
    tokenize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_B_DIR = PROJECT_ROOT / "models" / "model_b" / "traditional"
MODEL_B_ARTIFACT_FILE = "model_b_artifacts.joblib"

DISTRACTOR_NUMERIC_FEATURES = [
    "same_answer_type",
    "compatible_question_type",
    "candidate_word_count",
    "correct_word_count",
    "candidate_char_count",
    "correct_char_count",
    "length_ratio",
    "length_difference_abs",
    "candidate_correct_jaccard",
    "candidate_question_jaccard",
    "candidate_article_jaccard",
    "candidate_sentence_jaccard",
    "correct_question_jaccard",
    "char_similarity_to_correct",
    "candidate_frequency",
    "candidate_in_question",
    "candidate_in_article",
    "sentence_position_ratio",
]

HINT_NUMERIC_FEATURES = [
    "question_sentence_jaccard",
    "answer_sentence_jaccard",
    "sentence_word_count",
    "sentence_position_ratio",
    "contains_answer",
    "contains_question_term",
]

DISTRACTOR_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "could",
    "every",
    "first",
    "from",
    "good",
    "great",
    "have",
    "here",
    "into",
    "just",
    "like",
    "little",
    "many",
    "more",
    "much",
    "must",
    "only",
    "other",
    "same",
    "some",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "very",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
ACTION_START_WORDS = {
    "ask",
    "asked",
    "buy",
    "bought",
    "call",
    "called",
    "chat",
    "chats",
    "collect",
    "collects",
    "communicate",
    "come",
    "deal",
    "deals",
    "finish",
    "finishes",
    "get",
    "gets",
    "give",
    "gives",
    "go",
    "goes",
    "hand",
    "hands",
    "help",
    "helps",
    "listen",
    "listens",
    "make",
    "makes",
    "play",
    "plays",
    "put",
    "puts",
    "read",
    "reads",
    "see",
    "share",
    "shared",
    "take",
    "takes",
    "talk",
    "talks",
    "think",
    "thinks",
    "watch",
    "watches",
    "wear",
    "wears",
}
PASSAGE_BAD_EDGE_WORDS = {
    "a",
    "an",
    "according",
    "after",
    "asked",
    "at",
    "became",
    "began",
    "before",
    "bought",
    "by",
    "called",
    "came",
    "come",
    "could",
    "does",
    "doing",
    "done",
    "each",
    "gave",
    "gets",
    "give",
    "goes",
    "going",
    "gone",
    "helped",
    "in",
    "into",
    "kept",
    "liked",
    "looked",
    "made",
    "makes",
    "near",
    "on",
    "played",
    "said",
    "says",
    "shared",
    "started",
    "stopped",
    "told",
    "to",
    "took",
    "used",
    "wanted",
    "went",
    "were",
    "with",
    "will",
    "would",
    "the",
}
PREPOSITIONAL_PHRASE_PATTERN = re.compile(
    r"\b(?:in|on|at|from|to|with|near|inside|into|after|before|during|by)\s+"
    r"(?P<object>(?:(?:the|a|an|his|her|their|our|my|your)\s+)?"
    r"[A-Za-z0-9][A-Za-z0-9'-]*(?:\s+(?:of|the|and|for|in|on|at|after|before|"
    r"his|her|their|our|my|your|[A-Za-z0-9][A-Za-z0-9'-]*)){0,4})",
    re.IGNORECASE,
)
ACTION_PHRASE_PATTERN = re.compile(
    r"\b(?P<action>(?:make|makes|made|take|takes|took|collect|collects|hand|hands|help|helps|"
    r"give|gives|get|gets|deal|deals|think|thinks|play|plays|watch|watches|listen|listens|"
    r"chat|chats|finish|finishes|care)\s+(?:sure\s+that\s+)?"
    r"[A-Za-z][A-Za-z'-]*(?:\s+(?:that|to|of|in|with|their|his|her|the|a|an|and|"
    r"[A-Za-z][A-Za-z'-]*)){0,7})",
    re.IGNORECASE,
)


@dataclass
class Distractor:
    text: str
    score: float
    answer_type: str
    source_sentence: str
    method: str


@dataclass
class Hint:
    level: int
    text: str
    score: float
    source_sentence: str
    method: str


@dataclass
class ModelBOutput:
    question_type: str
    correct_answer: str
    distractors: list[Distractor]
    hints: list[Hint]


def char_similarity(left, right):
    return SequenceMatcher(None, str(left).lower(), str(right).lower()).ratio()


def mask_answer(sentence, answer):
    answer = clean_answer(answer)
    if not answer:
        return sentence
    original = str(sentence)
    masked = re.sub(re.escape(answer), "____", original, count=1, flags=re.IGNORECASE)
    if masked == original:
        light_stopwords = {"that", "than", "with", "from", "this", "there", "their", "about", "into"}
        for token in answer_tokens(answer):
            if len(token) <= 3 or token in light_stopwords:
                continue
            masked = re.sub(rf"\b{re.escape(token)}s?\b", "____", masked, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", masked).strip()


def answer_type_for_text(text, sentence=""):
    start, _end = find_answer_span(sentence or str(text), text)
    return infer_answer_type(text, sentence or str(text), max(start, 0))


def clean_passage_phrase(phrase):
    phrase = clean_answer(phrase)
    phrase = re.sub(r"^(?:his|her|their|our|my|your)\s+", "", phrase, flags=re.IGNORECASE)
    return clean_answer(phrase)


def is_action_like_phrase(text):
    tokens = answer_tokens(text)
    if not tokens:
        return False
    return tokens[0] in ACTION_START_WORDS or tokens[0].endswith(("ing", "ed"))


def is_phrase_level_distractor(candidate, correct_answer, question_type):
    text = clean_answer(candidate.get("answer", ""))
    tokens = answer_tokens(text)
    if not tokens:
        return False
    if candidate.get("source") == "option":
        return True
    if re.search(r"[,.;:!?]", text):
        return False
    candidate_type = candidate.get("answer_type", "fact")
    correct_tokens = answer_tokens(correct_answer)
    if len(tokens) == 1:
        if tokens[0] in DISTRACTOR_STOPWORDS:
            return False
        if question_type in {"how_many", "when", "where", "who"}:
            return candidate_type in {"number", "date", "place", "named_entity", "person_or_named_entity"}
        return len(correct_tokens) <= 1 and candidate_type in {"number", "date", "named_entity", "person_or_named_entity"}
    if len(tokens) > 8:
        return False
    if candidate.get("source") == "passage":
        if normalize_candidate_key(text) in normalize_candidate_key(correct_answer):
            return False
        if jaccard_overlap(text, correct_answer) > 0.70 or char_similarity(text, correct_answer) > 0.90:
            return False
        if is_action_like_phrase(correct_answer) and not is_action_like_phrase(text):
            return False
        if tokens[0] in PASSAGE_BAD_EDGE_WORDS or tokens[-1] in PASSAGE_BAD_EDGE_WORDS:
            return False
        if any(token in {"him", "her", "them", "they"} for token in tokens):
            return False
    return True


def normalize_candidate_key(text):
    return normalize_text(clean_answer(text))


def compatible_with_question_type(candidate_type, correct_type, question_type):
    if question_type == "who":
        return candidate_type in {"person_or_named_entity", "named_entity"}
    if question_type == "where":
        return candidate_type in {"place", "named_entity", "person_or_named_entity"}
    if question_type == "when":
        return candidate_type == "date"
    if question_type == "how_many":
        return candidate_type == "number"
    if question_type in {"what", "which", "cloze"}:
        return candidate_type == correct_type or candidate_type == "fact"
    return True


def add_option_candidate(candidates, article, question, text, correct_answer):
    text = clean_answer(text)
    if not text:
        return
    key = normalize_candidate_key(text)
    if not key or key == normalize_candidate_key(correct_answer):
        return
    source_sentence, start, _end, sentence_index = find_source_sentence(article, text, question)
    sentence_count = max(len(split_sentences(article)), 1)
    answer_type = infer_answer_type(text, source_sentence or text, max(start, 0))
    candidates[key] = {
        "answer": text,
        "answer_type": answer_type,
        "sentence": source_sentence,
        "sentence_index": sentence_index,
        "sentence_count": sentence_count,
        "source": "option",
    }


def add_passage_candidate(candidates, text, sentence, sentence_index, sentence_count, correct_answer):
    text = clean_passage_phrase(text)
    key = normalize_candidate_key(text)
    if not key or key == normalize_candidate_key(correct_answer):
        return
    if key in normalize_candidate_key(correct_answer) or normalize_candidate_key(correct_answer) in key:
        return
    start, _end = find_answer_span(sentence, text)
    candidates.setdefault(
        key,
        {
            "answer": text,
            "answer_type": infer_answer_type(text, sentence or text, max(start, 0)),
            "sentence": sentence,
            "sentence_index": sentence_index,
            "sentence_count": sentence_count,
            "source": "passage",
        },
    )


def add_pattern_phrase_candidates(candidates, article, correct_answer, pattern, group_name):
    sentences = split_sentences(article)
    sentence_count = max(len(sentences), 1)
    for sentence_index, sentence in enumerate(sentences):
        for match in pattern.finditer(sentence):
            phrase = match.group(group_name)
            add_passage_candidate(candidates, phrase, sentence, sentence_index, sentence_count, correct_answer)
            phrase_tokens = phrase.split()
            if len(phrase_tokens) > 2 and phrase_tokens[0].lower() in {"the", "a", "an"}:
                add_passage_candidate(
                    candidates,
                    " ".join(phrase_tokens[1:]),
                    sentence,
                    sentence_index,
                    sentence_count,
                    correct_answer,
                )


def extract_distractor_candidates(article, question, correct_answer, extra_options=None, max_candidates=180):
    correct_key = normalize_candidate_key(correct_answer)
    candidates = {}

    for candidate in build_candidate_frame(article):
        text = clean_answer(candidate["answer"])
        key = normalize_candidate_key(text)
        if not key or key == correct_key:
            continue
        candidates[key] = {
            "answer": text,
            "answer_type": candidate.get("answer_type", "fact"),
            "sentence": candidate.get("sentence", ""),
            "sentence_index": int(candidate.get("sentence_index", 0)),
            "sentence_count": int(candidate.get("sentence_count", 1)),
            "source": "passage",
        }

    add_pattern_phrase_candidates(candidates, article, correct_answer, PREPOSITIONAL_PHRASE_PATTERN, "object")
    add_pattern_phrase_candidates(candidates, article, correct_answer, ACTION_PHRASE_PATTERN, "action")

    for option in extra_options or []:
        add_option_candidate(candidates, article, question, option, correct_answer)

    ranked = list(candidates.values())
    ranked.sort(
        key=lambda item: (
            item["source"] == "option",
            len(answer_tokens(item["answer"])) <= 5,
            -item.get("sentence_index", 0),
        ),
        reverse=True,
    )
    return ranked[:max_candidates]


def build_distractor_feature(candidate, article, question, correct_answer, question_type):
    candidate_text = clean_answer(candidate["answer"])
    correct_answer = clean_answer(correct_answer)
    source_sentence = candidate.get("sentence", "")
    candidate_type = candidate.get("answer_type", "fact")
    correct_type = answer_type_for_text(correct_answer, source_sentence)
    sentence_count = max(int(candidate.get("sentence_count", 1)), 1)
    sentence_index = int(candidate.get("sentence_index", 0))
    candidate_frequency = str(article).lower().count(candidate_text.lower())
    candidate_words = len(answer_tokens(candidate_text))
    correct_words = len(answer_tokens(correct_answer))

    return {
        "feature_text": (
            f"{question_type} [QUESTION] {question} "
            f"[CORRECT] {correct_answer} [CANDIDATE] {candidate_text} "
            f"[SOURCE] {source_sentence}"
        ),
        "question_type": question_type,
        "candidate_type": candidate_type,
        "correct_type": correct_type,
        "candidate_source": candidate.get("source", "passage"),
        "same_answer_type": float(candidate_type == correct_type),
        "compatible_question_type": float(
            compatible_with_question_type(candidate_type, correct_type, question_type)
        ),
        "candidate_word_count": candidate_words,
        "correct_word_count": correct_words,
        "candidate_char_count": len(candidate_text),
        "correct_char_count": len(correct_answer),
        "length_ratio": len(candidate_text) / max(len(correct_answer), 1),
        "length_difference_abs": abs(len(candidate_text) - len(correct_answer)),
        "candidate_correct_jaccard": jaccard_overlap(candidate_text, correct_answer),
        "candidate_question_jaccard": jaccard_overlap(candidate_text, question),
        "candidate_article_jaccard": jaccard_overlap(candidate_text, article),
        "candidate_sentence_jaccard": jaccard_overlap(candidate_text, source_sentence),
        "correct_question_jaccard": jaccard_overlap(correct_answer, question),
        "char_similarity_to_correct": char_similarity(candidate_text, correct_answer),
        "candidate_frequency": float(candidate_frequency),
        "candidate_in_question": float(normalize_candidate_key(candidate_text) in normalize_text(question)),
        "candidate_in_article": float(normalize_candidate_key(candidate_text) in normalize_text(article)),
        "sentence_position_ratio": sentence_index / max(sentence_count - 1, 1),
    }


def distractor_features_to_matrix(rows, artifacts):
    cat = artifacts["distractor_category_vectorizer"].transform(
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
    numeric = csr_matrix(
        np.asarray(
            [[float(row.get(name, 0.0)) for name in DISTRACTOR_NUMERIC_FEATURES] for row in rows],
            dtype=np.float32,
        )
    )
    feature_mode = artifacts.get("distractor_feature_mode", "full")
    if feature_mode == "structured":
        return hstack([cat, numeric]).tocsr()
    text = artifacts["distractor_text_vectorizer"].transform([row["feature_text"] for row in rows])
    return hstack([text, cat, numeric]).tocsr()


def heuristic_distractor_score(feature):
    score = 0.25
    score += 0.25 * float(feature["compatible_question_type"])
    score += 0.15 * float(feature["same_answer_type"])
    score += 0.15 * min(float(feature["candidate_frequency"]), 3.0) / 3.0
    score += 0.10 * min(float(feature["candidate_article_jaccard"]) * 8.0, 1.0)
    score += 0.08 * max(0.0, 1.0 - min(abs(float(feature["length_ratio"]) - 1.0), 1.0))
    if feature["candidate_word_count"] >= 2:
        score += 0.10
    elif feature["correct_word_count"] >= 2 and feature["candidate_type"] not in {"number", "date", "named_entity", "person_or_named_entity"}:
        score -= 0.20
    score -= 0.25 * float(feature["char_similarity_to_correct"] > 0.86)
    score -= 0.30 * float(feature["candidate_correct_jaccard"] > 0.45)
    score -= 0.15 * float(feature["candidate_question_jaccard"] > 0.5)
    return max(0.0, min(1.0, score))


def score_distractor_candidates(candidates, article, question, correct_answer, artifacts=None):
    question_type = infer_question_type(question)
    rows = [build_distractor_feature(item, article, question, correct_answer, question_type) for item in candidates]
    if artifacts and "distractor_ranker" in artifacts and rows:
        matrix = distractor_features_to_matrix(rows, artifacts)
        ranker = artifacts["distractor_ranker"]
        ml_scores = ranker.predict_proba(matrix)[:, 1] if hasattr(ranker, "predict_proba") else ranker.predict(matrix)
        heuristic_scores = np.asarray([heuristic_distractor_score(row) for row in rows], dtype=np.float32)
        has_option_pool = any(candidate.get("source") == "option" for candidate in candidates)
        ml_weight = 0.75 if has_option_pool else 0.0
        scores = (ml_weight * np.asarray(ml_scores, dtype=np.float32)) + ((1.0 - ml_weight) * heuristic_scores)
    else:
        scores = np.asarray([heuristic_distractor_score(row) for row in rows], dtype=np.float32)
    return question_type, rows, scores


def select_diverse_distractors(candidates, scores, rows, top_k=3, correct_answer="", question_type=""):
    length_bonus = np.asarray(
        [0.003 * min(len(answer_tokens(candidate.get("answer", ""))), 6) for candidate in candidates],
        dtype=np.float32,
    )
    order = np.argsort(np.asarray(scores, dtype=np.float32) + length_bonus)[::-1]
    selected = []

    def try_add(index, strict_phrase_filter):
        candidate = candidates[int(index)]
        text = clean_answer(candidate["answer"])
        if strict_phrase_filter and not is_phrase_level_distractor(candidate, correct_answer, question_type):
            return False
        text_tokens = set(answer_tokens(text))
        if any(normalize_candidate_key(text) == normalize_candidate_key(item.text) for item in selected):
            return False
        if any(char_similarity(text, item.text) > 0.82 or jaccard_overlap(text, item.text) > 0.65 for item in selected):
            return False
        if any(
            text_tokens
            and set(answer_tokens(item.text))
            and (text_tokens <= set(answer_tokens(item.text)) or set(answer_tokens(item.text)) <= text_tokens)
            for item in selected
        ):
            return False
        selected.append(
            Distractor(
                text=text,
                score=round(float(scores[int(index)]), 4),
                answer_type=candidate.get("answer_type", "fact"),
                source_sentence=candidate.get("sentence", ""),
                method="ml-ranked" if rows else "heuristic",
            )
        )
        return True

    for index in order:
        try_add(index, strict_phrase_filter=True)
        if len(selected) == top_k:
            return selected
    return selected


def build_hint_feature(sentence, sentence_index, sentence_count, article, question, correct_answer):
    normalized_answer = normalize_candidate_key(correct_answer)
    normalized_sentence = normalize_text(sentence)
    question_terms = set(token.lower() for token in tokenize(question) if len(token) > 3)
    sentence_terms = set(token.lower() for token in tokenize(sentence))
    return {
        "feature_text": f"[QUESTION] {question} [ANSWER] {correct_answer} [SENTENCE] {sentence}",
        "question_sentence_jaccard": jaccard_overlap(question, sentence),
        "answer_sentence_jaccard": jaccard_overlap(correct_answer, sentence),
        "sentence_word_count": len(tokenize(sentence)),
        "sentence_position_ratio": sentence_index / max(sentence_count - 1, 1),
        "contains_answer": float(bool(normalized_answer and normalized_answer in normalized_sentence)),
        "contains_question_term": float(bool(question_terms & sentence_terms)),
    }


def hint_features_to_matrix(rows, artifacts):
    text = artifacts["hint_text_vectorizer"].transform([row["feature_text"] for row in rows])
    numeric = csr_matrix(
        np.asarray(
            [[float(row.get(name, 0.0)) for name in HINT_NUMERIC_FEATURES] for row in rows],
            dtype=np.float32,
        )
    )
    return hstack([text, numeric]).tocsr()


def score_hint_sentences(article, question, correct_answer, artifacts=None):
    sentences = split_sentences(article)
    rows = [
        build_hint_feature(sentence, index, len(sentences), article, question, correct_answer)
        for index, sentence in enumerate(sentences)
    ]
    if not rows:
        return [], [], np.asarray([])
    if artifacts and "hint_ranker" in artifacts:
        matrix = hint_features_to_matrix(rows, artifacts)
        ranker = artifacts["hint_ranker"]
        scores = ranker.predict_proba(matrix)[:, 1] if hasattr(ranker, "predict_proba") else ranker.predict(matrix)
    else:
        scores = np.asarray(
            [
                0.55 * row["question_sentence_jaccard"]
                + 0.35 * row["answer_sentence_jaccard"]
                + 0.10 * row["contains_question_term"]
                - 0.03 * row["sentence_position_ratio"]
                for row in rows
            ],
            dtype=np.float32,
        )
    return sentences, rows, scores


def question_focus_terms(question, max_terms=4):
    terms = []
    blocked = {
        "according",
        "answer",
        "are",
        "because",
        "can",
        "did",
        "does",
        "done",
        "following",
        "from",
        "give",
        "many",
        "much",
        "passage",
        "question",
        "see",
        "story",
        "that",
        "the",
        "there",
        "what",
        "when",
        "where",
        "which",
        "while",
        "about",
    }
    for token in tokenize(question):
        lowered = token.lower()
        if len(lowered) < 3 or lowered in blocked or lowered in terms:
            continue
        terms.append(lowered)
        if len(terms) == max_terms:
            break
    return terms


def context_keywords(sentence, correct_answer, question, max_terms=7):
    answer_terms = set(answer_tokens(correct_answer))
    question_terms = set(question_focus_terms(question, max_terms=8))
    chosen = []
    for token in tokenize(sentence):
        lowered = token.lower()
        if len(lowered) <= 3 or lowered in answer_terms or lowered in DISTRACTOR_STOPWORDS:
            continue
        if lowered not in chosen:
            chosen.append(lowered)
    chosen.sort(key=lambda term: (term not in question_terms, sentence.lower().find(term)))
    return chosen[:max_terms]


def sentence_number_hint(sentence_index, sentence_count):
    if sentence_count <= 1:
        return "Look in the main sentence of the passage."
    return f"Look around sentence {sentence_index + 1} of {sentence_count}."


def passage_region(sentence_index, sentence_count):
    if sentence_count <= 1:
        return "the passage"
    ratio = sentence_index / max(sentence_count - 1, 1)
    if ratio < 0.34:
        return "the beginning of the passage"
    if ratio < 0.67:
        return "the middle of the passage"
    return "the ending part of the passage"


def generate_hints(article, question, correct_answer, question_type, artifacts=None):
    sentences, _rows, scores = score_hint_sentences(article, question, correct_answer, artifacts)
    if not sentences:
        return []
    order = np.argsort(scores)[::-1]
    best_sentence = sentences[int(order[0])]
    source_sentence, _start, _end, source_index = find_source_sentence(article, correct_answer, question)
    if not source_sentence:
        source_sentence = best_sentence
        source_index = int(order[0])

    hint_type = {
        "who": "a person or named entity",
        "where": "a place or location",
        "when": "a time or date",
        "how_many": "a number or quantity",
    }.get(question_type, "a phrase from the passage")
    focus_terms = question_focus_terms(question)
    focus_text = ", ".join(focus_terms) if focus_terms else "the key idea in the question"
    clue_terms = context_keywords(source_sentence, correct_answer, question)
    clue_text = ", ".join(clue_terms) if clue_terms else "the surrounding sentence context"
    masked_source = mask_answer(source_sentence, correct_answer)

    hints = [
        Hint(
            level=1,
            text=f"Look in {passage_region(source_index, len(sentences))}. The answer is {hint_type}.",
            score=round(float(scores[int(order[0])]), 4),
            source_sentence="",
            method="broad-typed-clue",
        ),
        Hint(
            level=2,
            text=f"{sentence_number_hint(source_index, len(sentences))} Match the question focus ({focus_text}) with these context words: {clue_text}.",
            score=round(float(scores[int(order[0])]), 4),
            source_sentence=source_sentence,
            method="context-keywords",
        ),
        Hint(
            level=3,
            text=f"Near-explicit clue: {masked_source}",
            score=1.0,
            source_sentence=source_sentence,
            method="near-explicit-masked",
        ),
    ]
    return hints


@lru_cache(maxsize=2)
def load_model_b_artifacts(model_dir=None):
    model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_B_DIR
    artifact_path = model_path / MODEL_B_ARTIFACT_FILE
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)


def generate_model_b(article, question, correct_answer, extra_options=None, top_k=3, model_dir=None):
    correct_answer = clean_answer(correct_answer)
    artifacts = load_model_b_artifacts(str(model_dir)) if model_dir else load_model_b_artifacts()
    candidates = extract_distractor_candidates(article, question, correct_answer, extra_options=extra_options)
    question_type, rows, scores = score_distractor_candidates(
        candidates,
        article,
        question,
        correct_answer,
        artifacts=artifacts,
    )
    selected = select_diverse_distractors(
        candidates,
        scores,
        rows,
        top_k=top_k,
        correct_answer=correct_answer,
        question_type=question_type,
    )
    hints = generate_hints(article, question, correct_answer, question_type, artifacts=artifacts)
    return ModelBOutput(
        question_type=question_type,
        correct_answer=correct_answer,
        distractors=selected,
        hints=hints,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Model B distractors and hints.")
    parser.add_argument("--article", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--correct-answer", required=True)
    parser.add_argument("--extra-options", default="", help="Optional JSON list of answer options to rank as candidate distractors.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    extra_options = json.loads(args.extra_options) if args.extra_options else None

    output = generate_model_b(
        args.article,
        args.question,
        args.correct_answer,
        extra_options=extra_options,
        top_k=args.top_k,
        model_dir=args.model_dir,
    )
    print(json.dumps(asdict(output), indent=2))


if __name__ == "__main__":
    main()
