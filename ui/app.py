import json
import random
import sys
import time
from dataclasses import asdict
from html import escape
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "Streamlit is not installed. Install UI dependencies with `pip install streamlit`."
    ) from exc

from model_a_generation import generate_questions
from model_a_inference import OPTION_COLUMNS, load_backend
from model_b import generate_model_b


BERT_CHECKPOINT = PROJECT_DIR / "models" / "model_a" / "neural" / "bert_mc_30000q_plus_new30000q"
CLASSICAL_DIR = PROJECT_DIR / "models" / "model_a" / "traditional"

BACKENDS = {
    "BERT multiple-choice": {
        "key": "bert",
        "caption": "Best measured Model A. Trained on 60k RACE train questions.",
        "metrics": {"BLEU": 0.5431, "ROUGE-L": 0.6073, "METEOR": 0.5526, "Exact": 0.5172},
    },
    "Classical LR + XGBoost": {
        "key": "classical",
        "caption": "Traditional supervised baseline with TF-IDF, cosine, lexical, and question-relative features.",
        "metrics": {"BLEU": 0.4106, "ROUGE-L": 0.4922, "METEOR": 0.4438, "Exact": 0.3739},
    },
}


st.set_page_config(page_title="RACE Quiz Lab", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --page-bg: #f4f6f3;
        --panel-bg: #ffffff;
        --panel-soft: #edf2ee;
        --text-main: #202923;
        --text-muted: #647067;
        --border-soft: #d6ddd7;
        --accent: #4f7d6b;
        --accent-strong: #3d6656;
        --accent-soft: #e3eee8;
        --shadow-soft: none;
    }
    .stApp {
        background: var(--page-bg);
        color: var(--text-main);
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }
    [data-testid="stSidebar"] {
        background: var(--panel-bg);
        border-right: 1px solid var(--border-soft);
    }
    .hero-shell {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        background: var(--panel-bg);
        box-shadow: var(--shadow-soft);
        padding: 1.15rem 1.25rem;
        margin-bottom: 1.1rem;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 1rem;
        align-items: center;
    }
    .hero-eyebrow {
        color: var(--accent-strong);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 0.35rem;
    }
    .app-title {
        font-size: 2.15rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
        color: var(--text-main);
    }
    .subtle {
        color: var(--text-muted);
        font-size: 0.98rem;
        margin: 0.45rem 0 0;
        max-width: 760px;
    }
    .hero-status {
        border: 1px solid #bdd1c6;
        background: var(--accent-soft);
        color: #234238;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-width: 230px;
        font-size: 0.9rem;
    }
    .hero-status strong {
        display: block;
        font-size: 1rem;
        margin-bottom: 0.15rem;
    }
    .screen-heading {
        border-top: 1px solid var(--border-soft);
        padding-top: 1rem;
        margin-top: 0.9rem;
        margin-bottom: 0.85rem;
    }
    .screen-heading span {
        display: inline-flex;
        align-items: center;
        border-radius: 8px;
        background: var(--accent-soft);
        color: var(--accent-strong);
        font-size: 0.78rem;
        font-weight: 800;
        padding: 0.22rem 0.5rem;
        margin-bottom: 0.45rem;
    }
    .screen-title {
        margin: 0;
        color: var(--text-main);
        font-size: 1.38rem;
        line-height: 1.2;
        font-weight: 760;
    }
    .screen-heading p {
        margin: 0.3rem 0 0;
        color: var(--text-muted);
        font-size: 0.94rem;
        max-width: 820px;
    }
    .inline-note {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        background: var(--panel-soft);
        padding: 0.78rem 0.9rem;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .prediction-box {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        padding: 1rem;
        background: var(--panel-soft);
        color: var(--text-main);
        box-shadow: none;
    }
    .answer-pill {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        border: 1px solid #bdd1c6;
        background: var(--accent-soft);
        color: var(--text-main);
        font-weight: 700;
        margin-right: 0.4rem;
    }
    .prediction-muted {
        color: var(--text-muted);
    }
    .generation-note {
        border: 1px solid var(--border-soft);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: var(--panel-soft);
        color: var(--text-main);
        font-size: 0.92rem;
    }
    div[data-testid="stButton"] button {
        border-radius: 8px;
        border-color: var(--border-soft);
        font-weight: 700;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
    }
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        border-radius: 8px;
        border-color: var(--border-soft);
        background: var(--panel-bg);
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border-color: var(--border-soft);
        background: var(--panel-bg);
    }
    @media (max-width: 760px) {
        .hero-shell {
            grid-template-columns: 1fr;
        }
        .hero-status {
            min-width: 0;
        }
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --page-bg: #111411;
            --panel-bg: #191d1a;
            --panel-soft: #20261f;
            --text-main: #f1f3ef;
            --text-muted: #aab3aa;
            --border-soft: #343b34;
            --accent: #79a894;
            --accent-strong: #9cc5b3;
            --accent-soft: #233329;
            --shadow-soft: none;
        }
        .stApp {
            background: var(--page-bg);
        }
        [data-testid="stSidebar"] {
            background: var(--panel-bg);
            border-right-color: var(--border-soft);
        }
        .hero-shell {
            background: var(--panel-bg);
        }
        .hero-status {
            border-color: #405647;
            background: var(--accent-soft);
            color: #d9eee4;
        }
        .screen-heading span {
            background: var(--accent-soft);
            color: var(--accent-strong);
        }
        .subtle {
            color: var(--text-muted);
        }
        .prediction-box {
            border-color: var(--border-soft);
            background: var(--panel-soft);
            color: var(--text-main);
        }
        .answer-pill {
            border-color: #405647;
            background: var(--accent-soft);
            color: var(--text-main);
        }
        .prediction-muted {
            color: var(--text-muted);
        }
        .generation-note {
            border-color: var(--border-soft);
            background: var(--panel-soft);
            color: var(--text-main);
        }
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.08rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value):
    return escape(str(value or ""))


def render_app_header():
    st.markdown(
        """
        <section class="hero-shell">
            <div>
                <div class="hero-eyebrow">Section 6.2 aligned interface</div>
                <div class="app-title" role="heading" aria-level="1">RACE Question Generator and Verifier</div>
                <p class="subtle">
                    A passage-first quiz workflow for Model A question/answer generation,
                    Model B distractors and graduated hints, plus developer analytics.
                </p>
            </div>
            <aside class="hero-status">
                <strong>Transparency note</strong>
                AI-generated questions, options, and hints can be wrong; keep a human review step before real exam use.
            </aside>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_screen_heading(number, title, copy):
    st.markdown(
        f"""
        <div class="screen-heading">
            <span>Screen {safe_text(number)}</span>
            <div class="screen-title" role="heading" aria-level="2">{safe_text(title)}</div>
            <p>{safe_text(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_test_data():
    path = PROJECT_DIR / "data" / "raw" / "test.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Loading Model A backend...")
def cached_backend(backend_key, bert_device, use_dual_ranker=False):
    return load_backend(
        backend_key,
        classical_dir=CLASSICAL_DIR,
        bert_checkpoint=BERT_CHECKPOINT,
        device=bert_device,
        use_dual_ranker=use_dual_ranker,
    )


@st.cache_data(show_spinner=False)
def cached_generation(article, top_k, mode):
    return generate_questions(article, top_k=top_k, mode=mode)


@st.cache_data(show_spinner=False)
def cached_model_b_generation(article, question, correct_answer, extra_options, top_k):
    output = generate_model_b(
        article,
        question,
        correct_answer,
        extra_options=list(extra_options) if extra_options else None,
        top_k=top_k,
    )
    return asdict(output)


@st.cache_data(show_spinner=False)
def load_model_b_metrics():
    model_b_dir = PROJECT_DIR / "models" / "model_b" / "traditional"
    distractor_path = model_b_dir / "model_b_distractor_results.csv"
    hint_path = model_b_dir / "model_b_hint_results.csv"
    distractor = pd.read_csv(distractor_path) if distractor_path.exists() else pd.DataFrame()
    hint = pd.read_csv(hint_path) if hint_path.exists() else pd.DataFrame()
    return distractor, hint


def set_input_state(article, question, options):
    st.session_state.article_input = str(article)
    st.session_state.question_input = str(question)
    for label in OPTION_COLUMNS:
        st.session_state[f"option_{label}_input"] = str(options.get(label, ""))
    st.session_state.options = {label: str(options.get(label, "")) for label in OPTION_COLUMNS}


def blank_options():
    return {label: "" for label in OPTION_COLUMNS}


def clear_generated_turn_state():
    for key in [
        "generated_questions",
        "active_generated_question",
        "model_b_output",
        "last_result",
        "last_options",
        "quiz_checked",
        "quiz_selected",
    ]:
        st.session_state.pop(key, None)
    st.session_state.model_b_hints_seen = 0
    st.session_state.model_b_answer_revealed = False
    st.session_state.using_generated_qa = False
    st.session_state.pop("pre_generated_verifier", None)


def capture_verifier_state():
    return {
        "article": st.session_state.get("article_input", ""),
        "question": st.session_state.get("question_input", ""),
        "options": {
            label: st.session_state.get(f"option_{label}_input", "")
            for label in OPTION_COLUMNS
        },
        "gold_letter": st.session_state.get("gold_letter", ""),
        "gold_text": st.session_state.get("gold_text", ""),
        "sample_id": st.session_state.get("sample_id", "manual"),
        "sample_index": int(st.session_state.get("sample_index", 0)),
    }


def restore_verifier_state(state):
    set_input_state(state.get("article", ""), state.get("question", ""), state.get("options", {}))
    st.session_state.gold_letter = state.get("gold_letter", "")
    st.session_state.gold_text = state.get("gold_text", "")
    st.session_state.model_b_correct_answer_input = state.get("gold_text", "")
    st.session_state.sample_id = state.get("sample_id", "manual")
    st.session_state.sample_index = int(state.get("sample_index", 0))
    st.session_state.using_generated_qa = False
    st.session_state.pop("pre_generated_verifier", None)
    st.session_state.pop("last_result", None)


def sample_to_state(row, sample_index=None):
    options = {label: str(row[label]) for label in OPTION_COLUMNS}
    article = str(row["article"])
    set_input_state(article, "", blank_options())
    st.session_state.gold_letter = ""
    st.session_state.gold_text = ""
    st.session_state.model_b_correct_answer_input = ""
    st.session_state.sample_id = str(row.get("question_id", row.get("example_id", "RACE sample")))
    st.session_state.race_article_snapshot = article
    st.session_state.race_question = str(row.get("question", ""))
    st.session_state.race_options = options
    st.session_state.race_gold_letter = str(row.get("answer", ""))
    st.session_state.race_gold_text = str(row.get("correct_answer_text", ""))
    st.session_state.race_available = True
    st.session_state.passage_cleared = False
    st.session_state.using_generated_qa = False
    st.session_state.pop("pre_generated_verifier", None)
    clear_generated_turn_state()
    if sample_index is not None:
        st.session_state.sample_index = int(sample_index)


def load_sample_at(index):
    test_df = load_test_data()
    if test_df.empty:
        return
    safe_index = int(index) % len(test_df)
    sample_to_state(test_df.iloc[safe_index], sample_index=safe_index)
    st.session_state.pop("last_result", None)


def ensure_initial_state():
    if "article_input" in st.session_state:
        return

    test_df = load_test_data()
    if not test_df.empty:
        sample_to_state(test_df.iloc[0], sample_index=0)
        return

    set_input_state("", "", {label: "" for label in OPTION_COLUMNS})
    st.session_state.gold_letter = ""
    st.session_state.gold_text = ""
    st.session_state.model_b_correct_answer_input = ""
    st.session_state.sample_id = "manual"
    st.session_state.sample_index = 0
    st.session_state.race_article_snapshot = ""
    st.session_state.race_question = ""
    st.session_state.race_options = blank_options()
    st.session_state.race_gold_letter = ""
    st.session_state.race_gold_text = ""
    st.session_state.race_available = False
    st.session_state.passage_cleared = True
    st.session_state.using_generated_qa = False


def score_table(result):
    displayed_options = st.session_state.get("last_options", st.session_state.options)
    rows = []
    for label in OPTION_COLUMNS:
        row = {
            "Option": label,
            "Answer text": displayed_options.get(label, ""),
            "Model score": float(result["scores"].get(label, 0.0)),
        }
        if "wrong_scores" in result:
            row["Wrong-answer score"] = float(result["wrong_scores"].get(label, 0.0))
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("Model score", ascending=False).reset_index(drop=True)


def validate_inputs(article, question, options):
    missing = []
    if not article.strip():
        missing.append("article")
    if not question.strip():
        missing.append("question")
    empty_options = [label for label, value in options.items() if not str(value).strip()]
    if empty_options:
        missing.append("options " + ", ".join(empty_options))
    return missing


def current_options_from_state():
    return {
        label: st.session_state.get(f"option_{label}_input", "")
        for label in OPTION_COLUMNS
    }


def current_model_b_default_answer(options, gold_text):
    if str(gold_text).strip():
        return str(gold_text).strip()
    model_b_answer = str(st.session_state.get("model_b_correct_answer_input", "")).strip()
    if model_b_answer:
        return model_b_answer
    last_result = st.session_state.get("last_result")
    if last_result and str(last_result.get("predicted_answer_text", "")).strip():
        return str(last_result["predicted_answer_text"]).strip()
    gold_letter = st.session_state.get("gold_letter", "")
    if gold_letter in options and str(options[gold_letter]).strip():
        return str(options[gold_letter]).strip()
    return str(options.get("A", "")).strip()


def race_context_available(article):
    if not st.session_state.get("race_available", False):
        return False
    if st.session_state.get("passage_cleared", False):
        return False
    return str(article).strip() == str(st.session_state.get("race_article_snapshot", "")).strip()


def generated_question_to_dict(item):
    return asdict(item)


def load_race_question_into_verifier(article):
    race_options = st.session_state.get("race_options", blank_options())
    for key in ["model_b_output", "last_result", "last_options", "quiz_checked", "quiz_selected"]:
        st.session_state.pop(key, None)
    st.session_state.model_b_hints_seen = 0
    st.session_state.model_b_answer_revealed = False
    st.session_state.pending_verifier_update = {
        "article": article,
        "question": st.session_state.get("race_question", ""),
        "options": race_options,
        "gold_letter": st.session_state.get("race_gold_letter", ""),
        "gold_text": st.session_state.get("race_gold_text", ""),
        "sample_id": st.session_state.get("sample_id", "race_sample"),
    }
    st.session_state.active_question_source = "RACE original"


def load_model_b_options_into_verifier(correct_answer, distractors, article=None, question=None):
    generated_values = [("correct", correct_answer)]
    generated_values.extend(("distractor", item["text"]) for item in distractors[:3])
    while len(generated_values) < len(OPTION_COLUMNS):
        generated_values.append(("distractor", ""))
    random.shuffle(generated_values)
    generated_options = {}
    gold_letter = "A"
    for label, (kind, value) in zip(OPTION_COLUMNS, generated_values):
        generated_options[label] = value
        if kind == "correct":
            gold_letter = label
    st.session_state.pending_verifier_update = {
        "article": st.session_state.get("article_input", "") if article is None else article,
        "question": st.session_state.get("question_input", "") if question is None else question,
        "options": generated_options,
        "gold_letter": gold_letter,
        "gold_text": correct_answer,
        "sample_id": "model_b_quiz",
    }
    return generated_options, gold_letter


def apply_pending_verifier_update():
    pending = st.session_state.pop("pending_verifier_update", None)
    if not pending:
        return
    set_input_state(
        pending.get("article", st.session_state.get("article_input", "")),
        pending.get("question", st.session_state.get("question_input", "")),
        pending.get("options", {}),
    )
    st.session_state.gold_letter = pending.get("gold_letter", "")
    st.session_state.gold_text = pending.get("gold_text", "")
    st.session_state.model_b_correct_answer_input = pending.get("gold_text", "")
    st.session_state.sample_id = pending.get("sample_id", st.session_state.get("sample_id", "manual"))
    st.session_state.quiz_checked = False
    st.session_state.pop("last_result", None)


def record_model_a_history(result):
    gold_letter = st.session_state.get("gold_letter", "")
    if not gold_letter:
        return
    row = {
        "sample_id": st.session_state.get("sample_id", "manual"),
        "backend": st.session_state.get("last_backend_label", ""),
        "predicted": result.get("predicted_letter", ""),
        "gold": gold_letter,
        "correct": int(result.get("predicted_letter", "") == gold_letter),
        "latency_ms": round(float(result.get("latency_ms", 0.0)), 2),
    }
    history = st.session_state.get("model_a_history", [])
    history.append(row)
    st.session_state.model_a_history = history[-50:]


def history_metrics(history):
    if not history:
        return {"n": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "latency_ms": 0.0}
    precisions = []
    recalls = []
    f1s = []
    for label in OPTION_COLUMNS:
        tp = sum(1 for row in history if row["predicted"] == label and row["gold"] == label)
        fp = sum(1 for row in history if row["predicted"] == label and row["gold"] != label)
        fn = sum(1 for row in history if row["predicted"] != label and row["gold"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "n": len(history),
        "accuracy": sum(row["correct"] for row in history) / len(history),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
        "latency_ms": sum(float(row.get("latency_ms", 0.0)) for row in history) / len(history),
    }


def confusion_matrix_frame(history):
    matrix = pd.DataFrame(0, index=OPTION_COLUMNS, columns=OPTION_COLUMNS)
    matrix.index.name = "Gold \\ Predicted"
    for row in history:
        gold = row.get("gold", "")
        predicted = row.get("predicted", "")
        if gold in OPTION_COLUMNS and predicted in OPTION_COLUMNS:
            matrix.loc[gold, predicted] += 1
    return matrix.reset_index()


def record_model_b_history(output, latency_ms, used_option_pool):
    row = {
        "sample_id": st.session_state.get("sample_id", "manual"),
        "question_type": output.get("question_type", ""),
        "distractors": len(output.get("distractors", [])),
        "hints": len(output.get("hints", [])),
        "used_option_pool": bool(used_option_pool),
        "latency_ms": round(float(latency_ms), 2),
    }
    history = st.session_state.get("model_b_history", [])
    history.append(row)
    st.session_state.model_b_history = history[-50:]
    st.session_state.last_model_b_latency_ms = round(float(latency_ms), 2)


def run_model_b_pipeline(article, question, correct_answer, extra_options=tuple(), top_k=3, used_option_pool=False):
    started = time.perf_counter()
    output = cached_model_b_generation(article, question, correct_answer, extra_options, top_k)
    latency_ms = (time.perf_counter() - started) * 1000.0
    st.session_state.model_b_output = output
    st.session_state.model_b_used_option_pool = used_option_pool
    st.session_state.model_b_hints_seen = 0
    st.session_state.model_b_answer_revealed = False
    record_model_b_history(output, latency_ms, used_option_pool)
    return output


def submit_model_a_only(article, generation_mode="both", generation_top_k=5):
    if not article.strip():
        st.error("Please enter or load a passage before running Model A.")
        return

    with st.spinner("Generating a Model A question and answer..."):
        generated_questions = cached_generation(article, generation_top_k, generation_mode)
        st.session_state.generated_questions = generated_questions
        if not generated_questions:
            st.error("Model A could not generate a usable question from this passage.")
            return

    selected_generated = generated_questions[0]
    generated_payload = generated_question_to_dict(selected_generated)
    st.session_state.active_generated_question = generated_payload
    st.session_state.active_question_source = "Model A generated"
    st.session_state.pending_verifier_update = {
        "article": article,
        "question": selected_generated.question,
        "options": blank_options(),
        "gold_letter": "",
        "gold_text": selected_generated.answer,
        "sample_id": "generated_model_a_question",
    }
    st.session_state.using_generated_qa = True
    st.session_state.pop("model_b_output", None)
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_options", None)
    st.session_state.quiz_checked = False
    st.session_state.model_b_hints_seen = 0
    st.session_state.model_b_answer_revealed = False
    st.rerun()


def submit_model_b_only(article):
    if not article.strip():
        st.error("Please enter or load a passage before running Model B.")
        return

    options = current_options_from_state()
    active_generated = st.session_state.get("active_generated_question", {})
    question = str(st.session_state.get("question_input", "")).strip()
    if not question and active_generated:
        question = str(active_generated.get("question", "")).strip()

    correct_answer = current_model_b_default_answer(options, st.session_state.get("gold_text", ""))
    if not correct_answer and active_generated:
        correct_answer = str(active_generated.get("answer", "")).strip()

    missing = []
    if not question:
        missing.append("question")
    if not correct_answer:
        missing.append("correct answer")
    if missing:
        st.error("Model B needs a " + " and ".join(missing) + ". Run Model A only first or fill the quiz fields.")
        return

    with st.spinner("Generating Model B distractors and hints..."):
        model_b_output = run_model_b_pipeline(
            article,
            question,
            correct_answer,
            extra_options=tuple(),
            top_k=3,
            used_option_pool=False,
        )

    if len(model_b_output.get("distractors", [])) < 3:
        st.warning(
            "Model B found fewer than three passage-grounded distractors, so the quiz may not have a full four-option set."
        )

    load_model_b_options_into_verifier(
        model_b_output["correct_answer"],
        model_b_output["distractors"],
        article=article,
        question=question,
    )
    st.rerun()


def submit_model_a_b(article, generation_mode="both", generation_top_k=5):
    if not article.strip():
        st.error("Please enter or load a passage before submitting.")
        return

    with st.spinner("Generating a Model A question, then Model B distractors and hints..."):
        generated_questions = cached_generation(article, generation_top_k, generation_mode)
        st.session_state.generated_questions = generated_questions
        if not generated_questions:
            st.error("Model A could not generate a usable question from this passage.")
            return

        selected_generated = generated_questions[0]
        generated_payload = generated_question_to_dict(selected_generated)
        st.session_state.active_generated_question = generated_payload
        st.session_state.active_question_source = "Model A generated"
        model_b_output = run_model_b_pipeline(
            article,
            selected_generated.question,
            selected_generated.answer,
            extra_options=tuple(),
            top_k=3,
            used_option_pool=False,
        )

    if len(model_b_output.get("distractors", [])) < 3:
        st.warning(
            "Model B found fewer than three passage-grounded distractors. The question and hints were generated, but the quiz needs three distractors for a full four-option view."
        )

    load_model_b_options_into_verifier(
        model_b_output["correct_answer"],
        model_b_output["distractors"],
        article=article,
        question=selected_generated.question,
    )
    st.session_state.pending_autorun = True
    st.rerun()


def run_model_a(backend_key, backend_label, device, article, question, options, use_dual_ranker=False):
    missing_fields = validate_inputs(article, question, options)
    if missing_fields:
        st.error("Please fill in: " + ", ".join(missing_fields))
        return

    try:
        st.session_state.options = options.copy()
        started = time.perf_counter()
        model = cached_backend(backend_key, device, use_dual_ranker)
        with st.spinner(f"Running {backend_label}..."):
            st.session_state.last_result = model.predict(article, question, options)
            st.session_state.last_result["latency_ms"] = (time.perf_counter() - started) * 1000.0
            st.session_state.last_backend_label = backend_label
            st.session_state.last_options = options.copy()
            record_model_a_history(st.session_state.last_result)
    except Exception as exc:
        st.error(str(exc))


ensure_initial_state()
apply_pending_verifier_update()

render_app_header()

with st.sidebar:
    st.header("Model A")
    backend_names = list(BACKENDS.keys())
    backend_label = st.radio(
        "Backend",
        backend_names,
        index=backend_names.index("Classical LR + XGBoost"),
        captions=[BACKENDS[name]["caption"] for name in BACKENDS],
    )
    backend_key = BACKENDS[backend_label]["key"]

    device = "auto"
    if backend_key == "bert":
        device = st.selectbox(
            "BERT device",
            ["auto", "cuda", "cpu"],
            help="Auto uses CUDA when PyTorch can access the RTX GPU.",
        )
    use_dual_ranker = False
    if backend_key == "classical":
        use_dual_ranker = st.checkbox(
            "Experimental wrong-answer resolver",
            value=False,
            help="Uses a second ranker to score likely distractors, then combines correct and wrong rankings. Not the default because it improved dev but not held-out test.",
        )
    auto_run_on_sample_change = st.checkbox(
        "Run after Next/Previous",
        value=False,
        help="Useful for fast testing. Leave off for slower BERT runs if you want manual control.",
    )

    st.divider()
    st.subheader("Reported Test Metrics")
    metric_cols = st.columns(2)
    for index, (name, value) in enumerate(BACKENDS[backend_label]["metrics"].items()):
        metric_cols[index % 2].metric(name, f"{value:.4f}")

    st.divider()
    st.caption("BERT checkpoint")
    st.code(str(BERT_CHECKPOINT.relative_to(PROJECT_DIR)), language="text")


render_screen_heading(
    "1",
    "Article Input",
    "Paste or upload a passage, or load a RACE sample. Submit runs Model A generation followed by Model B distractors and hints.",
)
test_df = load_test_data()
sample_total = len(test_df)
sample_index = int(st.session_state.get("sample_index", 0))

sample_header_cols = st.columns([0.44, 0.24, 0.32])
with sample_header_cols[0]:
    st.subheader("Passage")
with sample_header_cols[1]:
    st.caption(f"{sample_total:,} RACE test questions loaded" if sample_total else "Manual passage mode")
with sample_header_cols[2]:
    uploaded_passage = st.file_uploader(
        "Upload passage file",
        type=["txt", "md"],
        accept_multiple_files=False,
        help="Upload a plain-text passage for the Article Input screen.",
    )

if uploaded_passage is not None:
    raw_bytes = uploaded_passage.getvalue()
    upload_token = f"{uploaded_passage.name}:{len(raw_bytes)}"
    if st.session_state.get("last_uploaded_passage_token") != upload_token:
        try:
            uploaded_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            uploaded_text = raw_bytes.decode("latin-1", errors="replace")
        if not uploaded_text.strip():
            st.error("That uploaded file is empty. Please choose a passage file with text.")
        else:
            set_input_state(uploaded_text.strip(), "", blank_options())
            st.session_state.gold_letter = ""
            st.session_state.gold_text = ""
            st.session_state.model_b_correct_answer_input = ""
            st.session_state.sample_id = uploaded_passage.name
            st.session_state.race_available = False
            st.session_state.passage_cleared = True
            st.session_state.last_uploaded_passage_token = upload_token
            clear_generated_turn_state()
            st.success(f"Loaded uploaded passage: {uploaded_passage.name}")

nav_cols = st.columns([0.14, 0.14, 0.14, 0.16, 0.42])
with nav_cols[0]:
    previous_clicked = st.button("Previous", width="stretch", disabled=test_df.empty)
with nav_cols[1]:
    next_clicked = st.button("Next", width="stretch", disabled=test_df.empty)
with nav_cols[2]:
    random_clicked = st.button("Random", width="stretch", disabled=test_df.empty)
with nav_cols[3]:
    clear_passage_clicked = st.button("Clear Passage", width="stretch")
with nav_cols[4]:
    if sample_total and st.session_state.get("race_available", False) and not st.session_state.get("passage_cleared", False):
        st.caption(
            f"Sample {sample_index + 1:,} of {sample_total:,} | {st.session_state.get('sample_id', 'RACE sample')}"
        )
    else:
        st.caption("Paste a passage below to use the generator.")

sample_changed = False
if previous_clicked:
    load_sample_at(sample_index - 1)
    sample_changed = True
elif next_clicked:
    load_sample_at(sample_index + 1)
    sample_changed = True
elif random_clicked:
    load_sample_at(random.randint(0, max(sample_total - 1, 0)))
    sample_changed = True

if clear_passage_clicked:
    set_input_state("", "", blank_options())
    st.session_state.gold_letter = ""
    st.session_state.gold_text = ""
    st.session_state.model_b_correct_answer_input = ""
    st.session_state.sample_id = "manual"
    st.session_state.race_available = False
    st.session_state.passage_cleared = True
    clear_generated_turn_state()
    st.rerun()

if sample_changed:
    st.session_state.pending_autorun = auto_run_on_sample_change
    st.rerun()

article = st.text_area(
    "Passage",
    key="article_input",
    height=265,
    placeholder="Paste or type a reading passage here...",
)

with st.expander("Generation Settings", expanded=False):
    settings_cols = st.columns([0.32, 0.18, 0.50])
    with settings_cols[0]:
        generation_mode_label = st.radio(
            "Model A generation mode",
            ["Both", "ML typed generation", "Cloze fallback"],
            index=0,
            horizontal=False,
        )
        generation_mode = {
            "Both": "both",
            "ML typed generation": "ml",
            "Cloze fallback": "cloze",
        }[generation_mode_label]
    with settings_cols[1]:
        generation_top_k = st.slider("Candidates", min_value=1, max_value=10, value=5)
    with settings_cols[2]:
        st.caption(
            "Both is the default: supervised WH/question-type generation is tried first, with cloze fallback kept for weak templates."
        )

submit_cols = st.columns([0.22, 0.22, 0.22, 0.34])
with submit_cols[0]:
    if st.button("Submit A + B", type="primary", width="stretch", key="screen1_submit_new"):
        submit_model_a_b(article, generation_mode=generation_mode, generation_top_k=generation_top_k)
with submit_cols[1]:
    if st.button("Model A Only", width="stretch", key="screen1_model_a_only"):
        submit_model_a_only(article, generation_mode=generation_mode, generation_top_k=generation_top_k)
with submit_cols[2]:
    if st.button("Model B Only", width="stretch", key="screen1_model_b_only"):
        submit_model_b_only(article)
with submit_cols[3]:
    st.caption("Run both models together, or run Model A question generation and Model B distractor/hint generation separately.")

active_generated = st.session_state.get("active_generated_question")
if active_generated:
    with st.expander("Model A Generated Question", expanded=False):
        gen_cols = st.columns([0.18, 0.18, 0.18, 0.46])
        gen_cols[0].metric("Type", active_generated.get("question_type", ""))
        gen_cols[1].metric("Score", f"{float(active_generated.get('score', 0.0)):.4f}")
        gen_cols[2].metric(
            "Confidence",
            f"{float(active_generated.get('classifier_confidence', 0.0)):.4f}",
        )
        gen_cols[3].markdown(f"**Correct answer**  \n{active_generated.get('answer', '')}")
        st.markdown("**Question**")
        st.write(active_generated.get("question", ""))
        st.markdown("**Source sentence**")
        st.write(active_generated.get("source_sentence", ""))
        if active_generated.get("cluster_id") is not None:
            st.caption(
                "Unsupervised cluster: "
                f"#{active_generated.get('cluster_id')} | dominant type: {active_generated.get('cluster_question_type') or 'n/a'} "
                f"| purity: {float(active_generated.get('cluster_purity', 0.0)):.4f}"
            )

render_screen_heading(
    "2",
    "Question & Answer Quiz View",
    "Review the generated or RACE original question, answer the four-option quiz, and let Model A verify the selection.",
)
race_available_for_article = race_context_available(article)
if race_available_for_article:
    with st.expander("View Exact RACE Dataset Question", expanded=False):
        st.markdown("**RACE question**")
        st.write(st.session_state.get("race_question", ""))
        race_gold_letter = st.session_state.get("race_gold_letter", "")
        race_gold_text = st.session_state.get("race_gold_text", "")
        if race_gold_letter and race_gold_text:
            st.success(f"Correct option: {race_gold_letter} - {race_gold_text}")
        race_cols = st.columns([0.28, 0.28, 0.44])
        with race_cols[0]:
            if st.button("Use RACE Question", width="stretch"):
                load_race_question_into_verifier(article)
                st.rerun()
        with race_cols[1]:
            if st.button("View RACE Options", width="stretch"):
                st.session_state.show_race_options = not st.session_state.get("show_race_options", False)
        with race_cols[2]:
            st.caption("Available because the passage still matches the loaded RACE sample and was not cleared.")
        if st.session_state.get("show_race_options", False):
            race_option_rows = [
                {
                    "Option": label,
                    "Text": st.session_state.get("race_options", {}).get(label, ""),
                    "Correct": label == race_gold_letter,
                }
                for label in OPTION_COLUMNS
            ]
            st.dataframe(pd.DataFrame(race_option_rows), hide_index=True, width="stretch")
elif st.session_state.get("passage_cleared", False):
    st.caption("RACE original question/options are hidden for this turn because the passage was cleared.")

question = st.text_area(
    "Question",
    key="question_input",
    height=96,
    placeholder="Submit the passage to generate a Model A question, or enter your own question.",
)

options = {}
first_option_row = st.columns(2)
second_option_row = st.columns(2)
option_slots = {
    "A": first_option_row[0],
    "B": first_option_row[1],
    "C": second_option_row[0],
    "D": second_option_row[1],
}
for label in OPTION_COLUMNS:
    with option_slots[label]:
        options[label] = st.text_area(f"Option {label}", key=f"option_{label}_input", height=108)

gold_letter = st.session_state.get("gold_letter", "")
gold_text = st.session_state.get("gold_text", "")
if gold_letter and gold_text:
    st.caption(f"Current correct option: {gold_letter} - {gold_text}")

st.markdown("#### Distractors, Options, and Hints")
action_cols = st.columns([0.26, 0.23, 0.22, 0.29])
with action_cols[0]:
    generate_distractors_clicked = st.button("Generate Distractors", width="stretch")
with action_cols[1]:
    use_race_options_clicked = st.button(
        "Use RACE Options",
        width="stretch",
        disabled=not race_available_for_article,
    )
with action_cols[2]:
    generate_hints_clicked = st.button("Generate Hints", width="stretch")
with action_cols[3]:
    st.caption("Distractors are passage-grounded by default; hints stay hidden until the hint panel is used.")

current_correct_answer = current_model_b_default_answer(options, gold_text)
if generate_distractors_clicked:
    missing_model_b = []
    if not article.strip():
        missing_model_b.append("passage")
    if not question.strip():
        missing_model_b.append("question")
    if not current_correct_answer.strip():
        missing_model_b.append("correct answer")
    if missing_model_b:
        st.error("Please provide: " + ", ".join(missing_model_b))
    else:
        with st.spinner("Generating passage-grounded distractors..."):
            refreshed_model_b = run_model_b_pipeline(
                article,
                question,
                current_correct_answer,
                extra_options=tuple(),
                top_k=3,
                used_option_pool=False,
            )
        load_model_b_options_into_verifier(
            refreshed_model_b["correct_answer"],
            refreshed_model_b["distractors"],
            article=article,
            question=question,
        )
        st.rerun()

if use_race_options_clicked:
    load_race_question_into_verifier(article)
    st.rerun()

if generate_hints_clicked:
    missing_hint_fields = []
    if not article.strip():
        missing_hint_fields.append("passage")
    if not question.strip():
        missing_hint_fields.append("question")
    if not current_correct_answer.strip():
        missing_hint_fields.append("correct answer")
    if missing_hint_fields:
        st.error("Please provide: " + ", ".join(missing_hint_fields))
    else:
        with st.spinner("Generating graduated hints..."):
            run_model_b_pipeline(
                article,
                question,
                current_correct_answer,
                extra_options=tuple(),
                top_k=3,
                used_option_pool=False,
            )

model_b_output = st.session_state.get("model_b_output")
if model_b_output:
    with st.expander(
        f"Model B Distractors ({model_b_output.get('question_type', '')})",
        expanded=False,
    ):
        distractor_rows = [
            {
                "Rank": index + 1,
                "Distractor": item["text"],
                "Score": f"{float(item['score']):.4f}",
                "Answer type": item["answer_type"],
                "Source sentence": item["source_sentence"],
            }
            for index, item in enumerate(model_b_output.get("distractors", []))
        ]
        if distractor_rows:
            st.dataframe(
                pd.DataFrame(distractor_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Source sentence": st.column_config.TextColumn("Source sentence", width="large"),
                },
            )
        else:
            st.info("No strong distractors were found for this passage/question pair.")

pending_autorun = bool(st.session_state.pop("pending_autorun", False))
if pending_autorun:
    run_model_a(backend_key, backend_label, device, article, question, options, use_dual_ranker)

if all(str(options[label]).strip() for label in OPTION_COLUMNS):
    selected_quiz_option = st.radio(
        "Choose an answer",
        options=OPTION_COLUMNS,
        format_func=lambda label: f"{label}. {options[label]}",
        horizontal=False,
        key="quiz_selected_option",
    )
    if st.button("Check Answer With Model A", width="content"):
        st.session_state.quiz_checked = True
        st.session_state.quiz_selected = selected_quiz_option
        run_model_a(backend_key, backend_label, device, article, question, options, use_dual_ranker)

    if st.session_state.get("quiz_checked", False):
        selected_letter = st.session_state.get("quiz_selected", selected_quiz_option)
        latest_result = st.session_state.get("last_result")
        verifier_letter = latest_result.get("predicted_letter", "") if latest_result else ""
        comparison_letter = gold_letter or verifier_letter
        explanation_source = "the current answer key" if gold_letter else "Model A's verifier prediction"
        if comparison_letter and selected_letter == comparison_letter:
            st.success(f"Correct. Selected {selected_letter}.")
            st.caption(
                f"Explanation: your selected option matches {explanation_source}. "
                f"{comparison_letter}. {options.get(comparison_letter, '')}"
            )
        elif comparison_letter:
            st.error(f"Incorrect. Selected {selected_letter}; expected {comparison_letter}.")
            st.caption(
                f"Explanation: {comparison_letter}. {options.get(comparison_letter, '')} "
                f"is currently treated as correct by {explanation_source}."
            )
        else:
            st.info("Model A needs to run before the selected answer can be checked.")
        if latest_result:
            st.caption(
                f"Model A verifier prediction: {verifier_letter} - {latest_result.get('predicted_answer_text', '')}"
            )
else:
    st.info("Submit a passage or generate distractors to create the four-option quiz.")

run_col, clear_col = st.columns([0.22, 0.78])
run_clicked = run_col.button("Run Model A", type="primary", width="stretch")
if clear_col.button("Clear Result", width="content"):
    st.session_state.pop("last_result", None)
    st.rerun()

if run_clicked:
    run_model_a(backend_key, backend_label, device, article, question, options, use_dual_ranker)

result = st.session_state.get("last_result")
if result:
    st.divider()
    left, right = st.columns([0.46, 0.54])

    with left:
        predicted_letter = result["predicted_letter"]
        predicted_text = result["predicted_answer_text"]
        device_label = result.get("device", "CPU / sklearn")
        backend_display = st.session_state.get("last_backend_label", backend_label)
        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-muted" style="font-size:0.86rem;margin-bottom:0.45rem;">Prediction</div>
                <div style="font-size:1.18rem;">
                    <span class="answer-pill">{safe_text(predicted_letter)}</span>{safe_text(predicted_text)}
                </div>
                <div class="prediction-muted" style="font-size:0.82rem;margin-top:0.7rem;">
                    Backend: {safe_text(backend_display)} | Device: {safe_text(device_label)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if gold_letter:
            if predicted_letter == gold_letter:
                st.success(f"Matches current correct option: {gold_letter}")
            else:
                st.warning(f"Current correct option: {gold_letter} - {gold_text}")

        result_payload = {
            "backend": result.get("backend"),
            "device": result.get("device", "sklearn"),
            "predicted_letter": predicted_letter,
            "predicted_answer_text": predicted_text,
            "scores": result["scores"],
        }
        if "wrong_scores" in result:
            result_payload["wrong_scores"] = result["wrong_scores"]
            result_payload["dual_ranker_config"] = result.get("dual_ranker_config", {})
        st.download_button(
            "Download Result JSON",
            data=json.dumps(result_payload, indent=2),
            file_name="model_a_result.json",
            mime="application/json",
            width="stretch",
        )

    with right:
        scores = score_table(result)
        st.subheader("Option Scores")
        st.dataframe(
            scores,
            hide_index=True,
            width="stretch",
            column_config={
                "Model score": st.column_config.NumberColumn(
                    "Model score",
                    format="%.4f",
                ),
                "Wrong-answer score": st.column_config.NumberColumn(
                    "Wrong-answer score",
                    format="%.4f",
                ),
            },
        )

render_screen_heading(
    "3",
    "Hint Panel",
    "Use the collapsible panel to reveal Model B hints one level at a time. The answer stays locked until all hints are used.",
)
model_b_output = st.session_state.get("model_b_output")
with st.expander("Graduated Hints", expanded=bool(model_b_output)):
    if not model_b_output:
        st.info("Submit the passage or generate hints to prepare the three-level hint panel.")
    else:
        hints = model_b_output.get("hints", [])
        hints_seen = int(st.session_state.get("model_b_hints_seen", 0))
        hint_cols = st.columns([0.18, 0.18, 0.64])
        with hint_cols[0]:
            next_hint_label = "Show Hint" if hints_seen == 0 else "Next Hint"
            if st.button(
                next_hint_label,
                disabled=hints_seen >= len(hints),
                width="stretch",
            ):
                st.session_state.model_b_hints_seen = min(hints_seen + 1, len(hints))
                st.rerun()
        with hint_cols[1]:
            if st.button("Reset Hints", width="stretch"):
                st.session_state.model_b_hints_seen = 0
                st.session_state.model_b_answer_revealed = False
                st.rerun()
        with hint_cols[2]:
            st.caption("Hint 1 is vague, Hint 2 is moderate, Hint 3 is near-explicit. The answer reveal appears after all hints.")

        hints_seen = int(st.session_state.get("model_b_hints_seen", 0))
        if hints_seen == 0:
            st.info("No hints revealed yet.")
        for hint in hints[:hints_seen]:
            with st.container(border=True):
                st.markdown(f"**Hint {hint['level']}**")
                st.write(hint["text"])
                st.caption(f"Score: {hint['score']:.4f} | Method: {hint['method']}")

        if hints and hints_seen >= len(hints):
            if st.button("Reveal Answer", width="content"):
                st.session_state.model_b_answer_revealed = True
            if st.session_state.get("model_b_answer_revealed", False):
                st.success(f"Answer: {model_b_output.get('correct_answer', '')}")

render_screen_heading(
    "4",
    "Developer / Analytics Dashboard",
    "Inspect last-N Model A metrics, Model B ranking reports, latency tracking, and CSV exports.",
)
with st.expander("Developer / Analytics Dashboard", expanded=False):
    model_a_history = st.session_state.get("model_a_history", [])
    history_summary = history_metrics(model_a_history)
    st.markdown("#### Model A Last-N Inferences")
    metric_cols = st.columns(6)
    metric_cols[0].metric("N", f"{history_summary['n']}")
    metric_cols[1].metric("Accuracy", f"{history_summary['accuracy']:.4f}")
    metric_cols[2].metric("Precision", f"{history_summary['precision']:.4f}")
    metric_cols[3].metric("Recall", f"{history_summary['recall']:.4f}")
    metric_cols[4].metric("F1", f"{history_summary['f1']:.4f}")
    metric_cols[5].metric("Avg latency", f"{history_summary['latency_ms']:.0f} ms")
    if model_a_history:
        st.dataframe(pd.DataFrame(model_a_history[-10:]), hide_index=True, width="stretch")
        st.markdown("#### Model A Confusion Matrix")
        st.dataframe(confusion_matrix_frame(model_a_history), hide_index=True, width="stretch")
        st.download_button(
            "Download Model A Session Log CSV",
            data=pd.DataFrame(model_a_history).to_csv(index=False),
            file_name="model_a_session_log.csv",
            mime="text/csv",
            width="content",
        )
    else:
        st.caption("Run Model A on samples with a correct option to populate last-N analytics.")

    model_b_history = st.session_state.get("model_b_history", [])
    st.markdown("#### Model B Last-N Inferences")
    if model_b_history:
        st.dataframe(pd.DataFrame(model_b_history[-10:]), hide_index=True, width="stretch")
        st.download_button(
            "Download Model B Session Log CSV",
            data=pd.DataFrame(model_b_history).to_csv(index=False),
            file_name="model_b_session_log.csv",
            mime="text/csv",
            width="content",
        )
    else:
        st.caption("Generate Model B distractors to populate Model B session analytics.")

    st.markdown("#### Model B Reported Metrics")
    distractor_metrics, hint_metrics = load_model_b_metrics()
    if not distractor_metrics.empty:
        display_columns = [
            "model",
            "split",
            "candidate_accuracy",
            "candidate_precision",
            "candidate_recall",
            "candidate_f1",
            "precision_at_3",
            "recall_at_3",
            "f1_at_3",
        ]
        st.dataframe(
            distractor_metrics[[col for col in display_columns if col in distractor_metrics.columns]],
            hide_index=True,
            width="stretch",
        )
    if not hint_metrics.empty:
        st.dataframe(hint_metrics, hide_index=True, width="stretch")
