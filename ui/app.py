"""
app.py — Streamlit UI for the Intelligent Reading Comprehension & Quiz Generation System
Screens: Article Input | Quiz | Hints | Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — add src/ to path so imports work from ui/ or root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import (
    load_models,
    run_full_pipeline,
    verify_answer,
    identify_best_answer,
    generate_distractors,
    generate_hints,
    get_session_stats,
    export_session_log,
    clear_session_log,
)

# ===========================================================================
# Page config — must be first Streamlit call
# ===========================================================================
st.set_page_config(
    page_title="RC Quiz System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Global CSS — dark purple / black theme
# ===========================================================================
st.markdown("""
<style>
/* ---- fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ---- root palette ---- */
:root {
    --bg-base:       #09080f;
    --bg-surface:    #110f1e;
    --bg-card:       #17142b;
    --bg-hover:      #1e1a36;
    --border:        #2a2545;
    --border-bright: #3d3670;
    --purple-dim:    #4b3d8f;
    --purple-mid:    #6c5ce7;
    --purple-bright: #9b8fff;
    --accent:        #a78bfa;
    --text-primary:  #e8e4ff;
    --text-secondary:#a49dbf;
    --text-muted:    #5c5480;
    --correct:       #34d399;
    --incorrect:     #f87171;
    --warning:       #fbbf24;
}

/* ---- base ---- */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ---- headings ---- */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

/* ---- paragraphs & labels ---- */
p, li, label, .stMarkdown {
    color: var(--text-secondary) !important;
    font-family: 'Syne', sans-serif !important;
}

/* ---- text area & text input ---- */
textarea, input[type="text"] {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--purple-mid) !important;
    box-shadow: 0 0 0 2px rgba(108,92,231,0.25) !important;
}

/* ---- primary buttons ---- */
.stButton > button {
    background-color: var(--purple-mid) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background-color: var(--purple-bright) !important;
    color: #09080f !important;
}
.stButton > button:disabled {
    background-color: var(--bg-hover) !important;
    color: var(--text-muted) !important;
}

/* ---- radio buttons ---- */
.stRadio > label {
    color: var(--text-secondary) !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
}

/* ---- select box ---- */
.stSelectbox [data-baseweb="select"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}

/* ---- expander ---- */
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ---- divider ---- */
hr {
    border-color: var(--border) !important;
}

/* ---- metric ---- */
[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.6rem !important;
    font-family: 'DM Mono', monospace !important;
}

/* ---- dataframe ---- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ---- info / success / error / warning boxes ---- */
.stAlert {
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
}

/* ---- spinner ---- */
.stSpinner > div {
    border-top-color: var(--purple-mid) !important;
}

/* ---- scrollbar ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--purple-dim); border-radius: 3px; }

/* ---- custom card ---- */
.rc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.rc-card-accent {
    border-left: 3px solid var(--purple-mid);
}
.rc-tag {
    display: inline-block;
    background: var(--purple-dim);
    color: var(--text-primary);
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.rc-correct {
    color: var(--correct) !important;
    font-weight: 700;
}
.rc-incorrect {
    color: var(--incorrect) !important;
    font-weight: 700;
}
.rc-hint-badge {
    display: inline-block;
    background: var(--bg-hover);
    border: 1px solid var(--border-bright);
    color: var(--purple-bright);
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin-right: 0.5rem;
    vertical-align: middle;
}
.rc-score-bar-wrap {
    background: var(--bg-hover);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.rc-score-bar {
    background: var(--purple-mid);
    border-radius: 4px;
    height: 6px;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session state initialisation
# ===========================================================================
def _init_state():
    defaults = {
        "bundle":           None,
        "load_error":       None,
        "screen":           "article",   # article | quiz | hints | dashboard
        "article":          "",
        "question":         "",
        "options":          {},          # {A:..., B:..., C:..., D:...}
        "correct_label":    "",
        "all_scores":       {},
        "distractors":      [],
        "hints":            [],
        "hints_revealed":   0,           # how many hints shown so far
        "answer_checked":   False,
        "selected_option":  None,
        "is_correct":       None,
        "source":           "race_original",
        "pipeline_latency": 0.0,
        "mode":             "RACE",      # RACE | Generated
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ===========================================================================
# Model loading — cached so it only runs once per session
# ===========================================================================
@st.cache_resource(show_spinner=False)
def _load_bundle():
    try:
        bundle = load_models(verbose=False)
        return bundle, None
    except Exception as e:
        return None, str(e)


def _ensure_bundle():
    if st.session_state.bundle is None:
        with st.spinner("Loading model artifacts..."):
            bundle, err = _load_bundle()
        st.session_state.bundle    = bundle
        st.session_state.load_error = err


# ===========================================================================
# Sidebar navigation
# ===========================================================================
def _sidebar():
    with st.sidebar:
        st.markdown("## RC Quiz System")
        st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)

        screens = {
            "article":   "Article Input",
            "quiz":      "Quiz View",
            "hints":     "Hint Panel",
            "dashboard": "Analytics",
        }
        for key, label in screens.items():
            active = st.session_state.screen == key
            style  = "font-weight:700;color:#a78bfa;" if active else "color:#a49dbf;"
            prefix = ">" if active else " "
            if st.button(f"{prefix}  {label}", key=f"nav_{key}",
                         use_container_width=True):
                st.session_state.screen = key
                st.rerun()

        st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)

        # Bundle status
        if st.session_state.bundle is not None:
            st.markdown(
                "<span style='color:#34d399;font-size:0.78rem;font-family:DM Mono,monospace'>"
                "Models loaded</span>", unsafe_allow_html=True)
        elif st.session_state.load_error:
            st.markdown(
                f"<span style='color:#f87171;font-size:0.78rem;font-family:DM Mono,monospace'>"
                f"Load error</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<span style='color:#fbbf24;font-size:0.78rem;font-family:DM Mono,monospace'>"
                "Not loaded</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("FAST-NUCES  |  AL2002  |  Spring 2026")


# ===========================================================================
# Screen 1 — Article Input
# ===========================================================================
def _screen_article():
    st.markdown("# Article Input")
    st.markdown(
        "<p style='color:#a49dbf'>Paste a reading passage or load a random sample "
        "from the RACE dataset, then submit to generate the quiz.</p>",
        unsafe_allow_html=True,
    )

    _ensure_bundle()

    if st.session_state.load_error:
        st.error(f"Model loading failed: {st.session_state.load_error}")
        st.info("Ensure all model artifacts are present in models/ and data/processed/.")
        return

    if st.session_state.bundle is None:
        st.warning("Models are still loading. Please wait.")
        return

    # ---- Mode selector ----
    mode = st.radio(
        "Mode",
        ["RACE (use dataset question)", "Generated (generate question from passage)"],
        index=0,
        horizontal=True,
    )
    st.session_state.mode = "RACE" if mode.startswith("RACE") else "Generated"

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1], gap="medium")

    with col_left:
        article_input = st.text_area(
            "Reading Passage",
            value=st.session_state.article,
            height=280,
            placeholder="Paste your reading passage here...",
            key="article_textarea",
        )

    with col_right:
        st.markdown("#### Options")

        if st.button("Load Random RACE Sample", use_container_width=True):
            _load_random_sample()
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.mode == "RACE":
            st.markdown(
                "<p style='font-size:0.8rem'>In RACE mode, the original question "
                "and four options from the dataset are used directly.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='font-size:0.8rem'>In Generated mode, Model A generates "
                "a question from the passage and identifies the best answer.</p>",
                unsafe_allow_html=True,
            )

    # ---- If RACE mode, show question + options fields ----
    if st.session_state.mode == "RACE":
        st.markdown("#### Question & Options")
        q_col, _ = st.columns([3, 1])
        with q_col:
            q_input = st.text_input(
                "Question",
                value=st.session_state.question,
                placeholder="Enter the question...",
            )

        opt_cols = st.columns(4)
        opt_vals = {}
        for i, label in enumerate(["A", "B", "C", "D"]):
            with opt_cols[i]:
                opt_vals[label] = st.text_input(
                    f"Option {label}",
                    value=st.session_state.options.get(label, ""),
                    key=f"opt_{label}",
                )

        correct_input = st.selectbox(
            "Correct Answer",
            ["A", "B", "C", "D"],
            index=["A", "B", "C", "D"].index(st.session_state.correct_label)
                  if st.session_state.correct_label in ["A", "B", "C", "D"] else 0,
        )
    else:
        q_input      = ""
        opt_vals     = st.session_state.options if st.session_state.options else {}
        correct_input = st.session_state.correct_label

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Submit ----
    if st.button("Submit  —  Run Pipeline", use_container_width=False):
        article_text = article_input.strip()
        if not article_text:
            st.error("Please enter a reading passage before submitting.")
            return

        if st.session_state.mode == "RACE":
            if not q_input.strip():
                st.error("Please enter a question.")
                return
            missing = [l for l, v in opt_vals.items() if not v.strip()]
            if missing:
                st.error(f"Please fill in option(s): {', '.join(missing)}")
                return

        with st.spinner("Running pipeline..."):
            _run_pipeline(article_text, q_input, opt_vals, correct_input)

        st.session_state.screen = "quiz"
        st.rerun()


def _load_random_sample():
    """Load a random row from val.csv into session state."""
    val_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'raw', 'val.csv'
    )
    try:
        df  = pd.read_csv(val_path)
        row = df.sample(1, random_state=int(time.time()) % 10000).iloc[0]
        st.session_state.article       = str(row["article"])
        st.session_state.question      = str(row["question"])
        st.session_state.options       = {k: str(row[k]) for k in ("A","B","C","D")}
        st.session_state.correct_label = str(row["answer"]).strip().upper()
    except Exception as e:
        st.error(f"Could not load RACE sample: {e}")


def _run_pipeline(article, question, options, correct_label):
    """Call inference and store results in session state."""
    bundle = st.session_state.bundle
    st.session_state.article       = article
    st.session_state.question      = question
    st.session_state.options       = options
    st.session_state.correct_label = correct_label

    # Reset quiz state
    st.session_state.answer_checked  = False
    st.session_state.selected_option = None
    st.session_state.is_correct       = None
    st.session_state.hints_revealed   = 0

    try:
        if st.session_state.mode == "RACE" and options:
            result = run_full_pipeline(
                bundle,
                article           = article,
                options           = options,
                gold_question     = question,
                gold_answer_label = correct_label,
            )
        else:
            result = run_full_pipeline(
                bundle,
                article = article,
                options = options if options else None,
            )

        st.session_state.question         = result.question
        st.session_state.options          = result.options
        st.session_state.correct_label    = result.correct_label
        st.session_state.all_scores       = result.all_scores
        st.session_state.distractors      = result.distractors
        st.session_state.hints            = result.hints
        st.session_state.pipeline_latency = result.latency_total
        st.session_state.source           = result.source

    except Exception as e:
        st.error(f"Pipeline error: {e}")


# ===========================================================================
# Screen 2 — Quiz View
# ===========================================================================
def _screen_quiz():
    st.markdown("# Quiz")

    if not st.session_state.question:
        st.info("No article submitted yet. Go to Article Input to get started.")
        if st.button("Go to Article Input"):
            st.session_state.screen = "article"
            st.rerun()
        return

    # Source badge
    badge = "RACE Original" if st.session_state.source == "race_original" else "AI Generated"
    st.markdown(
        f"<span class='rc-tag'>{badge}</span>",
        unsafe_allow_html=True,
    )

    # ---- Article (collapsed) ----
    with st.expander("Reading Passage", expanded=False):
        st.markdown(
            f"<div style='font-family:DM Mono,monospace;font-size:0.82rem;"
            f"color:#c4bde0;line-height:1.7'>{st.session_state.article}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Question ----
    st.markdown(
        f"<div class='rc-card rc-card-accent'>"
        f"<p style='color:#e8e4ff;font-size:1.05rem;font-weight:600;margin:0'>"
        f"{st.session_state.question}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- Build display options (correct + distractors, shuffled consistently) ----
    correct_label = st.session_state.correct_label
    correct_text  = st.session_state.options.get(correct_label, "")
    distractors   = st.session_state.distractors

    # Compose 4 display options: correct + up to 3 distractors
    display_options = _build_display_options(correct_text, distractors)

    # ---- Option radio ----
    if not st.session_state.answer_checked:
        chosen = st.radio(
            "Select your answer:",
            options=list(display_options.keys()),
            format_func=lambda k: f"{k}.  {display_options[k]}",
            index=None,
            key="quiz_radio",
        )
        st.session_state.selected_option = chosen

        col_check, col_hint, _ = st.columns([1, 1, 3])
        with col_check:
            if st.button("Check Answer"):
                if not chosen:
                    st.warning("Please select an option first.")
                else:
                    _check_answer(chosen, display_options)
                    st.rerun()
        with col_hint:
            if st.button("Show Hint"):
                st.session_state.screen = "hints"
                st.rerun()
    else:
        # Show result
        chosen      = st.session_state.selected_option
        is_correct  = st.session_state.is_correct

        display_correct = st.session_state.get("display_correct_label", "")
        for k, v in display_options.items():
            if k == display_correct:
                colour = "#34d399"
                marker = "  [Correct]"
            elif k == chosen and not is_correct:
                colour = "#f87171"
                marker = "  [Your answer]"
            else:
                colour = "#5c5480"
                marker = ""
            st.markdown(
                f"<div style='padding:0.5rem 0.8rem;margin:0.3rem 0;"
                f"border-radius:6px;background:#17142b;"
                f"border:1px solid #2a2545;"
                f"color:{colour};font-size:0.9rem'>"
                f"<strong>{k}.</strong>  {v}{marker}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if is_correct:
            st.markdown(
                "<div style='background:#052e16;border:1px solid #34d399;"
                "border-radius:6px;padding:0.8rem 1rem;"
                "color:#34d399;font-weight:600'>Correct.</div>",
                unsafe_allow_html=True,
            )
        else:
            display_correct = st.session_state.get("display_correct_label", "")
            st.markdown(
                f"<div style='background:#2d0a0a;border:1px solid #f87171;"
                f"border-radius:6px;padding:0.8rem 1rem;"
                f"color:#f87171;font-weight:600'>"
                f"Incorrect. The correct answer is <strong>{display_correct}</strong>: "
                f"{correct_text}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Verifier score table
        if st.session_state.all_scores:
            st.markdown("**Verifier Confidence Scores**")
            for label, score in sorted(st.session_state.all_scores.items()):
                pct = int(score * 100)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:0.8rem;"
                    f"margin:0.25rem 0;font-size:0.82rem;font-family:DM Mono,monospace'>"
                    f"<span style='width:20px;color:#a78bfa'>{label}</span>"
                    f"<div class='rc-score-bar-wrap' style='flex:1'>"
                    f"<div class='rc-score-bar' style='width:{pct}%'></div></div>"
                    f"<span style='width:42px;color:#a49dbf'>{score:.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        col_retry, col_hints, col_new = st.columns([1, 1, 1])
        with col_retry:
            if st.button("Try Again"):
                st.session_state.answer_checked  = False
                st.session_state.selected_option = None
                st.session_state.is_correct       = None
                st.rerun()
        with col_hints:
            if st.button("View Hints"):
                st.session_state.screen = "hints"
                st.rerun()
        with col_new:
            if st.button("New Article"):
                _reset_all()
                st.session_state.screen = "article"
                st.rerun()


def _build_display_options(correct_text, distractors):
    """
    Combine correct answer and up to 3 distractors into labelled options A-D.
    Uses a fixed shuffle seed per question for consistency across reruns.
    """
    import random
    items = [correct_text] + distractors[:3]
    # pad if fewer than 4
    while len(items) < 4:
        items.append("(no distractor generated)")
    rng = random.Random(hash(correct_text) % (2**31))
    rng.shuffle(items)
    labels = ["A", "B", "C", "D"]
    display_options = {labels[i]: items[i] for i in range(4)}
    
    # after shuffle, find which label now holds the correct text
    for label, text in display_options.items():
        if text == correct_text:
            st.session_state.display_correct_label = label
            break
            
    return display_options


def _check_answer(chosen, display_options):
    is_correct = chosen == st.session_state.display_correct_label

    st.session_state.answer_checked  = True
    st.session_state.selected_option = chosen
    st.session_state.is_correct       = is_correct


# ===========================================================================
# Screen 3 — Hint Panel
# ===========================================================================
def _screen_hints():
    st.markdown("# Hint Panel")

    if not st.session_state.question:
        st.info("No quiz loaded. Submit an article first.")
        if st.button("Go to Article Input"):
            st.session_state.screen = "article"
            st.rerun()
        return

    st.markdown(
        f"<div class='rc-card'>"
        f"<p style='color:#a49dbf;font-size:0.8rem;margin:0 0 0.3rem 0'>Question</p>"
        f"<p style='color:#e8e4ff;font-weight:600;margin:0'>{st.session_state.question}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    hints = st.session_state.hints
    if not hints:
        st.warning("No hints available for this question.")
        return

    hint_labels = ["General Clue", "Narrowed Context", "Near-Explicit Clue"]
    hint_descriptions = [
        "A broad clue about the topic of the answer.",
        "A more specific sentence from the passage.",
        "A sentence with the answer redacted.",
    ]

    revealed = st.session_state.hints_revealed

    for i, (label, desc) in enumerate(zip(hint_labels, hint_descriptions)):
        if i < revealed:
            st.markdown(
                f"<div class='rc-card rc-card-accent'>"
                f"<span class='rc-hint-badge'>Hint {i+1}</span>"
                f"<span style='color:#5c5480;font-size:0.72rem'>{label}</span>"
                f"<p style='color:#c4bde0;margin:0.5rem 0 0 0;font-size:0.9rem'>"
                f"{hints[i]}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='rc-card' style='opacity:0.45'>"
                f"<span class='rc-hint-badge'>Hint {i+1}</span>"
                f"<span style='color:#5c5480;font-size:0.72rem'>{label}</span>"
                f"<p style='color:#5c5480;margin:0.5rem 0 0 0;font-size:0.85rem;font-style:italic'>"
                f"{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_hint_btn, col_reveal, col_quiz = st.columns([1, 1, 1])

    with col_hint_btn:
        if revealed < len(hints):
            if st.button(f"Reveal Hint {revealed + 1}"):
                st.session_state.hints_revealed += 1
                st.rerun()
        else:
            st.markdown(
                "<p style='color:#5c5480;font-size:0.8rem'>All hints revealed.</p>",
                unsafe_allow_html=True,
            )

    with col_reveal:
        # Reveal answer only after all hints used
        if revealed >= len(hints):
            correct_text = st.session_state.options.get(
                st.session_state.correct_label, "")
            st.markdown(
                f"<div style='background:#052e16;border:1px solid #34d399;"
                f"border-radius:6px;padding:0.6rem 0.9rem;"
                f"color:#34d399;font-size:0.85rem'>"
                f"Answer: <strong>{st.session_state.correct_label}</strong> — {correct_text}"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_quiz:
        if st.button("Back to Quiz"):
            st.session_state.screen = "quiz"
            st.rerun()


# ===========================================================================
# Screen 4 — Analytics Dashboard
# ===========================================================================
def _screen_dashboard():
    st.markdown("# Analytics Dashboard")
    st.markdown(
        "<p style='color:#a49dbf'>Live session metrics and pre-trained "
        "model performance.</p>",
        unsafe_allow_html=True,
    )

    from inference import get_model_a_metrics, get_model_b_metrics

    bundle = st.session_state.bundle
    if bundle is None:
        st.info("Models not loaded yet. Submit an article to begin.")
        return

    stats = get_session_stats(bundle)

    # ================================================================
    # Section 1 — Session Overview
    # ================================================================
    st.markdown("## Session Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Requests",  stats["total_requests"])
    m2.metric("Avg Latency (s)", f"{stats['avg_latency']:.3f}")
    m3.metric("Hint Calls",      stats["per_task_counts"].get("generate_hints", 0))
    m4.metric("Verify Calls",    stats["per_task_counts"].get("verify_answer", 0))

    # ================================================================
    # Section 2 — Model A Performance (last N inferences)
    # ================================================================
    st.markdown("---")
    st.markdown("## Model A — Answer Verifier Performance")

    last_n = st.slider("Last N inferences", min_value=5,
                        max_value=200, value=50, step=5)
    ma = get_model_a_metrics(bundle, last_n=last_n)

    if ma is None:
        st.info(
            "Not enough labeled inferences yet. Use RACE mode "
            "(load a random sample) to accumulate gold-labeled results."
        )
    else:
        st.caption(f"Based on last {ma['n']} labeled inferences")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{ma['accuracy']:.4f}")
        c2.metric("Macro F1",  f"{ma['f1']:.4f}")
        c3.metric("Precision", f"{ma['precision']:.4f}")
        c4.metric("Recall",    f"{ma['recall']:.4f}")

        st.markdown("**Confusion Matrix** (rows = gold, cols = predicted)")
        cm_df = pd.DataFrame(
            ma['confusion_matrix'],
            index=[f"Gold {l}" for l in ma['labels']],
            columns=[f"Pred {l}" for l in ma['labels']],
        )
        st.dataframe(cm_df, use_container_width=True)

    # ================================================================
    # Section 3 — Model B Performance (from training results)
    # ================================================================
    st.markdown("---")
    st.markdown("## Model B — Distractor Ranker Performance")
    mb = get_model_b_metrics(bundle)

    if mb is None:
        st.warning(
            "model_b_results.csv not found. Run model_b_train.py first."
        )
    else:
        st.caption("Metrics from training evaluation on RACE val set")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Ranker Accuracy",  f"{mb['ranker_accuracy']:.4f}")
        b2.metric("Ranker F1",        f"{mb['ranker_f1']:.4f}")
        b3.metric("Ranker Precision", f"{mb['ranker_precision']:.4f}")
        b4.metric("Ranker Recall",    f"{mb['ranker_recall']:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Gen Precision",    f"{mb['gen_precision']:.4f}")
        g2.metric("Gen Recall",       f"{mb['gen_recall']:.4f}")
        g3.metric("Gen F1",           f"{mb['gen_f1']:.4f}")
        g4.metric("Hint Precision",   f"{mb['hint_precision']:.4f}")

    # ================================================================
    # Section 4 — Per-Request Latency Table
    # ================================================================
    st.markdown("---")
    st.markdown("## Inference Latency — Per Request")

    log = bundle.session_log
    if not log:
        st.info("No requests logged yet.")
    else:
        log_df = pd.DataFrame(log)
        # Show relevant columns only
        show_cols = [c for c in
                     ['task', 'latency', 'gold_label', 'pred_label']
                     if c in log_df.columns]
        log_df = log_df[show_cols].copy()
        if 'latency' in log_df.columns:
            log_df['latency'] = log_df['latency'].map(lambda x: f"{x:.4f}s")
        st.dataframe(log_df.tail(50), use_container_width=True)

        # Latency chart
        if 'latency' in pd.DataFrame(log).columns:
            chart_df = pd.DataFrame(log)[['latency']].tail(50)
            st.line_chart(chart_df, use_container_width=True)

    # ================================================================
    # Section 5 — Last Pipeline Result + Export
    # ================================================================
    st.markdown("---")
    if st.session_state.get("question"):
        st.markdown("## Last Pipeline Result")
        r1, r2 = st.columns([2, 1])
        with r1:
            st.markdown(
                f"<div class='rc-card'>"
                f"<p style='color:#a49dbf;font-size:0.78rem;margin:0'>Question</p>"
                f"<p style='color:#e8e4ff;margin:0.3rem 0 0 0'>"
                f"{st.session_state.question}</p></div>",
                unsafe_allow_html=True,
            )
        with r2:
            label = st.session_state.get(
                "display_correct_label", st.session_state.correct_label)
            st.markdown(
                f"<div class='rc-card'>"
                f"<p style='color:#a49dbf;font-size:0.78rem;margin:0'>"
                f"Correct Label</p>"
                f"<p style='color:#a78bfa;font-size:1.4rem;"
                f"font-family:DM Mono,monospace;font-weight:700;"
                f"margin:0.2rem 0 0 0'>{label}</p></div>",
                unsafe_allow_html=True,
            )

        if st.session_state.all_scores:
            st.markdown("**Verifier Scores**")
            label = st.session_state.get(
                "display_correct_label", st.session_state.correct_label)
            for lbl, score in sorted(st.session_state.all_scores.items()):
                pct    = int(score * 100)
                colour = "#34d399" if lbl == label else "#6c5ce7"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:0.8rem;"
                    f"margin:0.3rem 0;font-size:0.82rem;"
                    f"font-family:DM Mono,monospace'>"
                    f"<span style='width:20px;color:#a78bfa'>{lbl}</span>"
                    f"<div class='rc-score-bar-wrap' style='flex:1'>"
                    f"<div class='rc-score-bar' "
                    f"style='width:{pct}%;background:{colour}'></div></div>"
                    f"<span style='width:42px;color:#a49dbf'>{score:.3f}</span>"
                    f"{'<span style=\"color:#34d399;margin-left:4px\">'
                       'correct</span>' if lbl == label else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"<p style='color:#5c5480;font-size:0.78rem;"
            f"font-family:DM Mono,monospace;margin-top:0.5rem'>"
            f"Pipeline latency: "
            f"{st.session_state.pipeline_latency:.3f}s</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_exp, col_clr, _ = st.columns([1, 1, 3])
    with col_exp:
        if st.button("Export Session Log", use_container_width=True):
            if bundle and bundle.session_log:
                export_session_log(bundle)
                st.success("Exported to data/processed/session_log.csv")
            else:
                st.info("Session log is empty.")
    with col_clr:
        if st.button("Clear Session Log", use_container_width=True):
            if bundle:
                clear_session_log(bundle)
                st.success("Session log cleared.")


# ===========================================================================
# Helpers
# ===========================================================================
def _reset_all():
    keys = [
        "article", "question", "options", "correct_label",
        "all_scores", "distractors", "hints", "hints_revealed",
        "answer_checked", "selected_option", "is_correct",
        "pipeline_latency", "source",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    _init_state()


# ===========================================================================
# Main router
# ===========================================================================
def main():
    _sidebar()

    screen = st.session_state.screen

    if screen == "article":
        _screen_article()
    elif screen == "quiz":
        _screen_quiz()
    elif screen == "hints":
        _screen_hints()
    elif screen == "dashboard":
        _screen_dashboard()
    else:
        st.session_state.screen = "article"
        st.rerun()


if __name__ == "__main__":
    main()