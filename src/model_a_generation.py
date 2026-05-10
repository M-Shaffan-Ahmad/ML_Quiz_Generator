import argparse
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from model_a_unsupervised import load_unsupervised_artifacts, predict_question_cluster


OPTION_COLUMNS = ["A", "B", "C", "D"]
QUESTION_TYPES = ["cloze", "who", "what", "where", "when", "why", "how_many", "which", "other"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATOR_DIR = PROJECT_ROOT / "models" / "model_a" / "traditional" / "generator"
ARTIFACT_FILE = "generator_artifacts.joblib"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
MONTH_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
WEEKDAY_PATTERN = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"\b(?:(?:more than|less than|about|over|under)\s+)?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(?:%|percent|years?|days?|months?|weeks?|hours?|minutes?|dollars?|pounds?|"
    r"meters?|kilometers?|miles?|people|students|children|books|times))?\b",
    re.IGNORECASE,
)
CAPITALIZED_SPAN_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z]+(?:\s+(?:of|the|and|for|in|on|at|[A-Z][A-Za-z]+)){0,4})\b"
)
QUOTED_PATTERN = re.compile(r'"([^"\n]{3,90})"')

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "more",
    "than",
    "less",
    "about",
}
LOCATION_PREPOSITIONS = {"at", "from", "in", "inside", "into", "near", "on", "to", "within"}
TIME_PREPOSITIONS = {"after", "at", "before", "by", "during", "in", "on", "since", "until"}
PERSON_TITLES = {"mr", "mrs", "ms", "miss", "dr", "professor", "teacher"}
PERSON_CONTEXT_TOKENS = {
    "asked",
    "bought",
    "called",
    "came",
    "could",
    "gave",
    "get",
    "gets",
    "got",
    "had",
    "has",
    "have",
    "invited",
    "is",
    "liked",
    "likes",
    "lived",
    "lives",
    "made",
    "paid",
    "said",
    "says",
    "shared",
    "should",
    "told",
    "took",
    "wanted",
    "was",
    "went",
    "will",
    "would",
}
NON_PERSON_CAPITALIZED = {
    "american",
    "americans",
    "art",
    "biology",
    "chemistry",
    "chinese",
    "christmas",
    "english",
    "french",
    "geography",
    "german",
    "history",
    "japanese",
    "math",
    "maths",
    "music",
    "physics",
    "science",
    "spanish",
}
NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
}

CLASSIFIER_NUMERIC_FEATURES = [
    "answer_word_count",
    "answer_char_count",
    "sentence_word_count",
    "sentence_char_count",
    "article_word_count_log",
    "answer_position_ratio",
    "sentence_index_ratio",
    "answer_sentence_overlap",
    "answer_is_title",
    "answer_is_number",
    "answer_is_date",
    "answer_in_sentence",
]
RANKER_NUMERIC_FEATURES = [
    "classifier_confidence",
    "heuristic_score",
    "question_word_count",
    "answer_word_count",
    "sentence_word_count",
    "answer_position_ratio",
    "sentence_index_ratio",
    "answer_sentence_overlap",
    "question_answer_overlap",
    "question_sentence_overlap",
]


@dataclass
class GeneratedQuestion:
    question: str
    answer: str
    answer_type: str
    source_sentence: str
    score: float
    method: str
    temporary_options: list[str]
    question_type: str = "cloze"
    classifier_confidence: float = 0.0
    ranker_score: float = 0.0
    heuristic_score: float = 0.0
    cluster_id: int | None = None
    cluster_question_type: str = ""
    cluster_purity: float = 0.0
    cluster_distance: float = 0.0


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text))


def normalize_text(text):
    return " ".join(token.lower() for token in tokenize(text))


def jaccard_overlap(left_text, right_text):
    left = set(token.lower() for token in tokenize(left_text))
    right = set(token.lower() for token in tokenize(right_text))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def split_sentences(article):
    cleaned = re.sub(r"\s+", " ", str(article)).strip()
    if not cleaned:
        return []
    parts = SENTENCE_SPLIT_PATTERN.split(cleaned)
    sentences = []
    for part in parts:
        sentence = part.strip()
        token_count = len(tokenize(sentence))
        if token_count >= 6:
            sentences.append(sentence)
    return sentences


def clean_answer(answer):
    answer = re.sub(r"\s+", " ", str(answer)).strip(" ,.;:!?()[]{}")
    return answer.strip()


def answer_tokens(answer):
    return [token.lower() for token in tokenize(answer)]


def is_bad_answer(answer):
    tokens = answer_tokens(answer)
    if not tokens:
        return True
    if '"' in answer or "''" in answer or "``" in answer:
        return True
    if len(tokens) > 8:
        return True
    if all(token in STOPWORDS for token in tokens):
        return True
    if len(answer) < 2:
        return True
    return False


def previous_token(sentence, start_index):
    prefix_tokens = list(TOKEN_PATTERN.finditer(sentence[:start_index]))
    for match in reversed(prefix_tokens):
        token = match.group(0).lower()
        if token not in {"a", "an", "the"}:
            return token
    return ""


def next_token(sentence, end_index):
    suffix_tokens = list(TOKEN_PATTERN.finditer(sentence[end_index:]))
    for match in suffix_tokens:
        token = match.group(0).lower()
        if token not in {"a", "an", "the"}:
            return token
    return ""


def find_answer_span(sentence, answer):
    answer = clean_answer(answer)
    if not answer:
        return -1, -1
    match = re.search(re.escape(answer), sentence, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return -1, -1
    normalized_sentence = normalize_text(sentence)
    if normalized_answer in normalized_sentence:
        return 0, len(answer)
    return -1, -1


def infer_answer_type(answer, sentence, start_index):
    if MONTH_PATTERN.fullmatch(answer) or WEEKDAY_PATTERN.fullmatch(answer) or re.fullmatch(r"\d{4}", answer):
        return "date"
    if NUMBER_PATTERN.fullmatch(answer) or all(token in NUMBER_WORDS for token in answer_tokens(answer)):
        return "number"
    previous = previous_token(sentence, start_index)
    if previous in LOCATION_PREPOSITIONS:
        return "place"
    if re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+$", answer):
        return "person_or_named_entity"
    if answer[:1].isupper() and len(answer_tokens(answer)) == 1:
        if answer.lower() in NON_PERSON_CAPITALIZED:
            return "fact"
        answer_end = start_index + len(answer)
        following = next_token(sentence, answer_end)
        if previous in PERSON_TITLES or following in PERSON_CONTEXT_TOKENS:
            return "person_or_named_entity"
        return "named_entity"
    return "fact"


def infer_question_type(question):
    question_text = str(question).strip().lower()
    question_text = re.sub(r"^\W+", "", question_text)
    if "_" in question_text or " blank " in f" {question_text} ":
        return "cloze"
    if question_text.startswith("how many") or question_text.startswith("how much"):
        return "how_many"
    if question_text.startswith("how old") or question_text.startswith("how long"):
        return "how_many"
    for label in ("who", "where", "when", "why", "which", "what"):
        if question_text.startswith(label):
            return label
    return "other"


def add_candidate(candidates, sentence, answer, start_index, end_index):
    answer = clean_answer(answer)
    if is_bad_answer(answer):
        return
    if len(answer_tokens(answer)) == 1 and answer.istitle():
        before = sentence[:start_index].rstrip()
        after = sentence[end_index:].lstrip()
        if re.search(r"\b[A-Z][A-Za-z]+$", before) or re.match(r"[A-Z][A-Za-z]+\b", after):
            return
    answer_type = infer_answer_type(answer, sentence, start_index)
    key = (sentence.lower(), answer.lower())
    if key in candidates:
        return
    candidates[key] = {
        "sentence": sentence,
        "answer": answer,
        "start": start_index,
        "end": end_index,
        "answer_type": answer_type,
    }


def extract_candidates_from_sentence(sentence):
    candidates = {}

    for pattern in (MONTH_PATTERN, WEEKDAY_PATTERN, NUMBER_PATTERN, QUOTED_PATTERN, CAPITALIZED_SPAN_PATTERN):
        for match in pattern.finditer(sentence):
            answer = match.group(1) if pattern is QUOTED_PATTERN else match.group(0)
            start = match.start(1) if pattern is QUOTED_PATTERN else match.start()
            end = match.end(1) if pattern is QUOTED_PATTERN else match.end()
            add_candidate(candidates, sentence, answer, start, end)

    token_matches = list(TOKEN_PATTERN.finditer(sentence))
    content_spans = []
    current = []
    for match in token_matches:
        token = match.group(0)
        if token.lower() not in STOPWORDS and len(token) > 3:
            current.append(match)
        else:
            if current:
                content_spans.append(current)
                current = []
    if current:
        content_spans.append(current)

    for span in content_spans:
        for width in range(min(4, len(span)), 0, -1):
            for start_pos in range(0, len(span) - width + 1):
                group = span[start_pos : start_pos + width]
                start = group[0].start()
                end = group[-1].end()
                answer = sentence[start:end]
                add_candidate(candidates, sentence, answer, start, end)

    return list(candidates.values())


def find_source_sentence(article, answer, question=None):
    sentences = split_sentences(article)
    if not sentences:
        return "", -1, -1, 0

    normalized_answer = normalize_text(answer)
    for index, sentence in enumerate(sentences):
        start, end = find_answer_span(sentence, answer)
        if start >= 0:
            return sentence, start, end, index

    best_index = 0
    best_score = -1.0
    for index, sentence in enumerate(sentences):
        score = 0.75 * jaccard_overlap(sentence, answer)
        if question:
            score += 0.25 * jaccard_overlap(sentence, question)
        if normalized_answer and normalized_answer in normalize_text(sentence):
            score += 0.5
        if score > best_score:
            best_score = score
            best_index = index

    sentence = sentences[best_index]
    start, end = find_answer_span(sentence, answer)
    return sentence, start, end, best_index


def build_candidate_frame(article):
    sentences = split_sentences(article)
    raw_candidates = []
    for sentence_index, sentence in enumerate(sentences):
        for candidate in extract_candidates_from_sentence(sentence):
            candidate["sentence_index"] = sentence_index
            candidate["sentence_count"] = len(sentences)
            candidate["heuristic_score"] = rank_candidate(candidate, sentence_index, len(sentences))
            raw_candidates.append(candidate)
    return sorted(raw_candidates, key=lambda item: item["heuristic_score"], reverse=True)


def make_cloze_question(sentence, answer):
    blanked = re.sub(re.escape(answer), " _ ", sentence, count=1, flags=re.IGNORECASE)
    blanked = re.sub(r"\s+([,.;:!?])", r"\1", blanked)
    blanked = re.sub(r"\s+", " ", blanked).strip()
    if not blanked.endswith((".", "?", "!")):
        blanked = f"{blanked}."
    return f"According to the passage, {blanked}"


def cleanup_question(question):
    question = re.sub(r"\s+([,.;:!?])", r"\1", str(question))
    question = re.sub(r"\s+", " ", question).strip()
    question = question.rstrip(".!")
    if not question.endswith("?"):
        question = f"{question}?"
    if question:
        question = question[0].upper() + question[1:]
    return question


def strip_previous_preposition(prefix, question_type):
    tokens = list(TOKEN_PATTERN.finditer(prefix))
    if not tokens:
        return prefix
    token_index = len(tokens) - 1
    while token_index >= 0 and tokens[token_index].group(0).lower() in {"a", "an", "the"}:
        token_index -= 1
    if token_index < 0:
        return prefix
    previous = tokens[token_index].group(0).lower()
    removable = set()
    if question_type == "where":
        removable = LOCATION_PREPOSITIONS
    elif question_type == "when":
        removable = TIME_PREPOSITIONS
    if previous not in removable:
        return prefix
    return prefix[: tokens[token_index].start()].rstrip()


def make_typed_question(sentence, answer, question_type):
    if question_type in {"cloze", "other"}:
        return make_cloze_question(sentence, answer)

    start, end = find_answer_span(sentence, answer)
    if start < 0:
        return make_cloze_question(sentence, answer)

    prefix = sentence[:start].strip()
    suffix = sentence[end:].strip()
    answer_prompt = {
        "who": "who",
        "what": "what",
        "where": "where",
        "when": "when",
        "why": "why",
        "how_many": "how many",
        "which": "which answer",
    }.get(question_type, "what")

    if question_type in {"where", "when"}:
        prefix = strip_previous_preposition(prefix, question_type)
    if question_type == "why":
        cloze = make_cloze_question(sentence, answer)
        return cleanup_question(f"Why is this statement true according to the passage: {cloze}")
    if question_type == "what":
        copula_match = re.match(r"^(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*$", prefix, re.I)
        if copula_match:
            return cleanup_question(f"What is {copula_match.group('subject').strip()}")
    if not prefix:
        return cleanup_question(f"{answer_prompt} {suffix}")
    return cleanup_question(f"{prefix} {answer_prompt} {suffix}")


def rank_candidate(candidate, sentence_index, sentence_count):
    sentence = candidate["sentence"]
    answer = candidate["answer"]
    sentence_tokens = tokenize(sentence)
    answer_token_count = len(answer_tokens(answer))

    score = 0.20
    if 8 <= len(sentence_tokens) <= 32:
        score += 0.30
    elif len(sentence_tokens) <= 45:
        score += 0.15

    if 1 <= answer_token_count <= 5:
        score += 0.25
    elif answer_token_count <= 8:
        score += 0.10

    if candidate["answer_type"] in {"date", "number", "person_or_named_entity", "place"}:
        score += 0.18
    elif candidate["answer_type"] == "named_entity":
        score += 0.12

    if sentence.lower().count(answer.lower()) == 1:
        score += 0.08
    if sentence.endswith("?"):
        score -= 0.20
    if answer.lower() in STOPWORDS:
        score -= 0.25
    if sentence_count:
        position_ratio = sentence_index / max(sentence_count - 1, 1)
        score += max(0.0, 0.08 * (1.0 - position_ratio))

    return round(max(score, 0.0), 4)


def build_classifier_feature(candidate, article):
    sentence = candidate["sentence"]
    answer = candidate["answer"]
    sentence_count = max(int(candidate.get("sentence_count", 1)), 1)
    sentence_index = int(candidate.get("sentence_index", 0))
    start = int(candidate.get("start", -1))
    answer_word_count = len(answer_tokens(answer))
    sentence_word_count = len(tokenize(sentence))
    article_word_count = len(tokenize(article))
    answer_position_ratio = 0.0 if start < 0 or not sentence else start / max(len(sentence), 1)

    return {
        "feature_text": f"{candidate['answer_type']} {answer} [SEP] {sentence} [SEP] {str(article)[:1200]}",
        "answer_type": candidate["answer_type"],
        "answer_word_count": answer_word_count,
        "answer_char_count": len(answer),
        "sentence_word_count": sentence_word_count,
        "sentence_char_count": len(sentence),
        "article_word_count_log": float(np.log1p(article_word_count)),
        "answer_position_ratio": answer_position_ratio,
        "sentence_index_ratio": sentence_index / max(sentence_count - 1, 1),
        "answer_sentence_overlap": jaccard_overlap(answer, sentence),
        "answer_is_title": float(str(answer).istitle()),
        "answer_is_number": float(candidate["answer_type"] == "number"),
        "answer_is_date": float(candidate["answer_type"] == "date"),
        "answer_in_sentence": float(find_answer_span(sentence, answer)[0] >= 0),
    }


def classifier_features_to_matrix(rows, artifacts):
    text_vectorizer = artifacts["text_vectorizer"]
    category_vectorizer = artifacts["category_vectorizer"]
    X_text = text_vectorizer.transform([row["feature_text"] for row in rows])
    X_cat = category_vectorizer.transform(
        [{"answer_type": row["answer_type"]} for row in rows]
    )
    X_num = csr_matrix(
        np.asarray(
            [[float(row.get(name, 0.0)) for name in CLASSIFIER_NUMERIC_FEATURES] for row in rows],
            dtype=np.float32,
        )
    )
    return hstack([X_text, X_cat, X_num]).tocsr()


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def predict_question_type(candidate, article, artifacts):
    if not artifacts:
        return "cloze", 0.0
    row = build_classifier_feature(candidate, article)
    X = classifier_features_to_matrix([row], artifacts)
    classifier = artifacts["question_type_classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X)[0]
        classes = list(classifier.classes_)
    else:
        probabilities = softmax(classifier.decision_function(X))[0]
        classes = list(classifier.classes_)
    best_index = int(np.argmax(probabilities))
    return str(classes[best_index]), float(probabilities[best_index])


def build_ranker_feature(candidate, question, question_type, classifier_confidence):
    sentence = candidate["sentence"]
    answer = candidate["answer"]
    sentence_count = max(int(candidate.get("sentence_count", 1)), 1)
    sentence_index = int(candidate.get("sentence_index", 0))
    start = int(candidate.get("start", -1))
    row = {
        "question_type": question_type,
        "answer_type": candidate["answer_type"],
        "classifier_confidence": float(classifier_confidence),
        "heuristic_score": float(candidate.get("heuristic_score", 0.0)),
        "question_word_count": len(tokenize(question)),
        "answer_word_count": len(answer_tokens(answer)),
        "sentence_word_count": len(tokenize(sentence)),
        "answer_position_ratio": 0.0 if start < 0 or not sentence else start / max(len(sentence), 1),
        "sentence_index_ratio": sentence_index / max(sentence_count - 1, 1),
        "answer_sentence_overlap": jaccard_overlap(answer, sentence),
        "question_answer_overlap": jaccard_overlap(question, answer),
        "question_sentence_overlap": jaccard_overlap(question, sentence),
    }
    return row


def ranker_features_to_matrix(rows, artifacts):
    vectorizer = artifacts["ranker_vectorizer"]
    return vectorizer.transform(rows)


def predict_ranker_score(candidate, question, question_type, classifier_confidence, artifacts):
    if not artifacts or "ranker" not in artifacts:
        return 0.65 * float(candidate.get("heuristic_score", 0.0)) + 0.35 * float(classifier_confidence)
    row = build_ranker_feature(candidate, question, question_type, classifier_confidence)
    X = ranker_features_to_matrix([row], artifacts)
    ranker = artifacts["ranker"]
    if hasattr(ranker, "predict_proba"):
        return float(ranker.predict_proba(X)[0, 1])
    return float(ranker.predict(X)[0])


def selection_score(candidate, question_type, ranker_score, artifacts):
    heuristic_score = float(candidate.get("heuristic_score", 0.0))
    if artifacts and "ranker" in artifacts:
        base_score = (0.70 * float(ranker_score)) + (0.30 * heuristic_score)
    else:
        base_score = heuristic_score
    if question_type == "cloze":
        return base_score
    answer_type = candidate.get("answer_type", "fact")
    typed_bonus = 0.20
    type_matches_answer = (
        (answer_type == "number" and question_type == "how_many")
        or (answer_type == "date" and question_type == "when")
        or (answer_type == "place" and question_type == "where")
        or (answer_type == "person_or_named_entity" and question_type == "who")
    )
    if type_matches_answer:
        typed_bonus = 0.50
    return min(1.0, base_score + typed_bonus)


def answer_type_question_type(candidate):
    answer_type = candidate.get("answer_type", "fact")
    if answer_type == "number":
        return "how_many"
    if answer_type == "date":
        return "when"
    if answer_type == "place":
        return "where"
    if answer_type == "person_or_named_entity":
        return "who"
    sentence = candidate.get("sentence", "")
    answer = candidate.get("answer", "")
    start, _end = find_answer_span(sentence, answer)
    if start >= 0:
        prefix = sentence[:start].strip()
        if re.match(r"^.+?\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*$", prefix, re.I):
            return "what"
    return None


def coerce_question_type(question_type, candidate):
    answer_type = candidate.get("answer_type", "fact")
    suggested = answer_type_question_type(candidate)
    if question_type in {"cloze", "other"} and suggested:
        return suggested
    if answer_type == "person_or_named_entity" and question_type != "cloze":
        return "who"
    if question_type == "cloze":
        return "cloze"
    if answer_type == "number":
        return "how_many"
    if answer_type == "date":
        return "when"
    if answer_type == "place":
        return "where"
    if question_type == "who" and answer_type != "person_or_named_entity":
        return "what"
    if question_type in {"where", "when", "how_many"}:
        return "what"
    if question_type == "other":
        return "cloze"
    return question_type


def select_temporary_options(answer, answer_type, ranked_candidates):
    options = [answer]
    answer_key = answer.lower()
    preferred = []
    fallback = []
    for candidate in ranked_candidates:
        candidate_answer = candidate["answer"]
        candidate_key = candidate_answer.lower()
        if candidate_key == answer_key or candidate_key in {item.lower() for item in options}:
            continue
        if candidate["answer_type"] == answer_type:
            preferred.append(candidate_answer)
        else:
            fallback.append(candidate_answer)

    for candidate_answer in preferred + fallback:
        options.append(candidate_answer)
        if len(options) == 4:
            break
    return options


@lru_cache(maxsize=4)
def load_generator_artifacts(model_dir=None):
    model_path = Path(model_dir) if model_dir else DEFAULT_GENERATOR_DIR
    artifact_path = model_path / ARTIFACT_FILE
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)


def requested_question_types(candidate, article, artifacts, mode):
    mode = str(mode or "ml").strip().lower()
    if mode == "cloze":
        return [("cloze", 1.0, "cloze-fallback")]

    if artifacts:
        predicted_type, confidence = predict_question_type(candidate, article, artifacts)
        predicted_type = coerce_question_type(predicted_type, candidate)
        types = [(predicted_type, confidence, "supervised-typed-generation")]
        if mode == "both" and predicted_type != "cloze":
            types.append(("cloze", 1.0, "cloze-fallback"))
        return types

    return [("cloze", 1.0, "cloze-fallback")]


def generate_questions(article, top_k=5, mode="ml", model_dir=None):
    candidates = build_candidate_frame(article)
    artifacts = load_generator_artifacts(str(model_dir)) if model_dir else load_generator_artifacts()
    unsupervised_artifacts = load_unsupervised_artifacts()
    generated = []
    seen = set()

    for candidate in candidates:
        for question_type, confidence, method in requested_question_types(candidate, article, artifacts, mode):
            question = make_typed_question(candidate["sentence"], candidate["answer"], question_type)
            if not question or len(tokenize(question)) < 4:
                question_type = "cloze"
                question = make_cloze_question(candidate["sentence"], candidate["answer"])
            key = (question.lower(), candidate["answer"].lower())
            if key in seen:
                continue
            ranker_score = predict_ranker_score(candidate, question, question_type, confidence, artifacts)
            score = selection_score(candidate, question_type, ranker_score, artifacts)
            cluster_info = predict_question_cluster(
                candidate,
                article,
                question=question,
                artifacts=unsupervised_artifacts,
            )
            if cluster_info and cluster_info.get("dominant_question_type") == question_type:
                score = min(1.0, score + 0.03 * float(cluster_info.get("question_type_purity", 0.0)))
            temporary_options = select_temporary_options(candidate["answer"], candidate["answer_type"], candidates)
            generated.append(
                GeneratedQuestion(
                    question=question,
                    answer=candidate["answer"],
                    answer_type=candidate["answer_type"],
                    source_sentence=candidate["sentence"],
                    score=round(score, 4),
                    method=method,
                    temporary_options=temporary_options,
                    question_type=question_type,
                    classifier_confidence=round(float(confidence), 4),
                    ranker_score=round(float(ranker_score), 4),
                    heuristic_score=round(float(candidate.get("heuristic_score", 0.0)), 4),
                    cluster_id=None if not cluster_info else int(cluster_info["cluster_id"]),
                    cluster_question_type="" if not cluster_info else str(cluster_info["dominant_question_type"]),
                    cluster_purity=0.0 if not cluster_info else round(float(cluster_info["question_type_purity"]), 4),
                    cluster_distance=0.0 if not cluster_info else round(float(cluster_info["distance"]), 4),
                )
            )
            seen.add(key)

    generated = sorted(
        generated,
        key=lambda item: (item.score, item.classifier_confidence, item.heuristic_score),
        reverse=True,
    )
    return generated[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Generate Model A question-answer candidates from a passage.")
    parser.add_argument("--article", default=None)
    parser.add_argument("--article-file", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["ml", "cloze", "both"], default="ml")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    if args.article_file:
        with open(args.article_file, "r", encoding="utf-8") as file:
            article = file.read()
    elif args.article:
        article = args.article
    else:
        raise SystemExit("Provide --article or --article-file.")

    result = [
        asdict(item)
        for item in generate_questions(
            article,
            top_k=args.top_k,
            mode=args.mode,
            model_dir=args.model_dir,
        )
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
