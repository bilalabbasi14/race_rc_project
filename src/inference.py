"""
inference.py — Unified Inference API
=====================================
Loads all pre-trained Model A and Model B artifacts once at startup and
exposes clean functions that the UI (app.py) calls at runtime.

Public API
----------
load_models()                        -> ModelBundle
verify_answer(bundle, article, question, option) -> (is_correct, confidence)
identify_best_answer(bundle, article, question, options_dict) -> (label, confidence, all_scores)
generate_question(bundle, article, options_dict) -> (question, pred_label, confidence, all_scores)
generate_distractors(bundle, article, question, correct_answer) -> list[str]
generate_hints(bundle, article, question, correct_answer) -> list[str]
run_full_pipeline(bundle, article, options_dict) -> PipelineResult
"""

import joblib
import numpy as np
import pandas as pd
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.sparse import hstack, csr_matrix

# ---------------------------------------------------------------------------
# Paths — must match the paths used during training
# ---------------------------------------------------------------------------
PROCESSED_DIR    = 'data/processed/'
MODEL_A_DIR      = 'models/model_a/traditional/'
MODEL_B_DIR      = 'models/model_b/traditional/'

# ---------------------------------------------------------------------------
# Import generation helpers from model_a_generate and model_b_train.
# We import the helper functions directly so inference.py stays thin and
# doesn't duplicate logic.
# ---------------------------------------------------------------------------
# Model A helpers
from model_a_generate import (
    extract_candidate_sentences,
    generate_candidate_questions,
    question_features,
    identify_answer,
    clean_text    as _clean_a,
    split_sentences as _split_sentences_a,
)

# Model B helpers
from model_b_train import (
    generate_distractors as _gen_distractors,
    generate_hints       as _gen_hints,
    ohe_cosine,
    content_words,
    split_sentences      as _split_sentences_b,
)

# ===========================================================================
# ModelBundle — holds every loaded artifact
# ===========================================================================
@dataclass
class ModelBundle:
    """Container for all loaded model artifacts."""
    # Model A — supervised classifiers
    logistic_regression: object
    svm:                 object
    naive_bayes:         object
    random_forest:       object

    # Model A — ensemble meta-classifier (stacking)
    stacking_meta_clf:   Optional[object]

    # Model A — question generation ranker
    question_ranker:     Optional[object]

    # Shared — text vectorizer (CountVectorizer fitted in preprocessing.py)
    vectorizer:          object

    # Model B — hint scorer (LogisticRegression)
    hint_scorer:         Optional[object]

    # Model B — Word2Vec (optional; None if gensim not installed)
    w2v_model:           Optional[object] = None

    # Session-level performance log (populated by inference calls)
    session_log: List[dict] = field(default_factory=list)


# ===========================================================================
# Model loading
# ===========================================================================
def load_models(verbose: bool = True) -> ModelBundle:
    """
    Load all trained artifacts from disk and return a ModelBundle.

    Artifacts loaded
    ----------------
    Model A supervised  : logistic_regression, svm, naive_bayes, random_forest
    Model A ensemble    : stacking_meta_clf (optional)
    Model A generation  : question_ranker (optional)
    Shared              : vectorizer (CountVectorizer)
    Model B             : hint_scorer
    Model B (optional)  : word2vec KeyedVectors

    Any optional artifact that is missing is loaded as None with a warning,
    so the rest of the pipeline keeps working.
    """
    def _load(path, name, required=True):
        if os.path.exists(path):
            obj = joblib.load(path)
            if verbose:
                print(f"  [OK] {name}")
            return obj
        else:
            msg = f"  [{'ERROR' if required else 'WARN'}] {name} not found at {path}"
            print(msg)
            if required:
                raise FileNotFoundError(msg)
            return None

    if verbose:
        print("=" * 55)
        print("Loading model artifacts...")
        print("=" * 55)

    # Required Model A classifiers
    lr  = _load(MODEL_A_DIR + 'logistic_regression.pkl', 'Logistic Regression')
    svm = _load(MODEL_A_DIR + 'svm.pkl',                 'Linear SVM')
    nb  = _load(MODEL_A_DIR + 'naive_bayes.pkl',         'Naive Bayes')
    rf  = _load(MODEL_A_DIR + 'random_forest.pkl',       'Random Forest')

    # Optional Model A artifacts
    meta_clf = _load(MODEL_A_DIR + 'stacking_meta_clf.pkl',
                     'Stacking Meta-Classifier', required=False)
    ranker   = _load(MODEL_A_DIR + 'question_ranker.pkl',
                     'Question Ranker', required=False)

    # Required shared vectorizer
    vectorizer = _load(PROCESSED_DIR + 'vectorizer.pkl', 'CountVectorizer')

    # Optional Model B artifacts
    hint_scorer = _load(MODEL_B_DIR + 'hint_scorer.pkl',
                        'Hint Scorer (LR)', required=False)

    # Word2Vec — graceful skip
    w2v_model = None
    w2v_path  = MODEL_B_DIR + 'word2vec_kv.bin'
    if os.path.exists(w2v_path):
        try:
            from gensim.models import KeyedVectors
            w2v_model = KeyedVectors.load(w2v_path)
            if verbose:
                print("  [OK] Word2Vec KeyedVectors")
        except Exception as e:
            if verbose:
                print(f"  [WARN] Word2Vec found but could not load: {e}")
    else:
        if verbose:
            print("  [WARN] Word2Vec not cached — distractor generation "
                  "will use phrase-extraction + frequency sources only")

    if verbose:
        print("=" * 55)
        print("All required artifacts loaded.\n")

    return ModelBundle(
        logistic_regression = lr,
        svm                 = svm,
        naive_bayes         = nb,
        random_forest       = rf,
        stacking_meta_clf   = meta_clf,
        question_ranker     = ranker,
        vectorizer          = vectorizer,
        hint_scorer         = hint_scorer,
        w2v_model           = w2v_model,
    )


# ===========================================================================
# Internal helpers
# ===========================================================================
def _build_lexical_features(article: str, question: str, option: str) -> csr_matrix:
    """
    Reproduce the 5 handcrafted lexical features from model_a_train.py so
    that inference uses the same combined feature space as training.

    Features
    --------
    f1 : article-option word overlap
    f2 : question-option word overlap
    f3 : option length ratio relative to total text
    f4 : article lexical coverage
    f5 : cosine similarity between (article + question) and option
    """
    article_words  = set(article.split())
    question_words = set(question.split())
    option_words   = set(option.split())

    opt_len   = len(option_words)
    total_len = len(article_words) + len(question_words) + opt_len

    f1 = len(article_words  & option_words) / (opt_len + 1)
    f2 = len(question_words & option_words) / (opt_len + 1)
    f3 = opt_len / (total_len + 1)
    f4 = len(article_words) / (total_len + 1)

    vocab_aq  = article_words | question_words
    vocab_all = vocab_aq | option_words
    v1 = np.array([1 if w in vocab_aq    else 0 for w in vocab_all], dtype=np.float32)
    v2 = np.array([1 if w in option_words else 0 for w in vocab_all], dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    f5 = float(np.dot(v1, v2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0

    return csr_matrix(np.array([[f1, f2, f3, f4, f5]], dtype=np.float32))


def _vectorize_single(article: str, question: str, option: str,
                      vectorizer) -> csr_matrix:
    """
    Build the combined OHE + lexical feature vector for ONE (article, question,
    option) triple — matches the feature space used in model_a_train.py.
    """
    combined_text = article + ' ' + question + ' ' + option
    X_ohe         = vectorizer.transform([combined_text])
    X_lex         = _build_lexical_features(article, question, option)
    return hstack([X_ohe, X_lex])


def _get_model_score(model, X) -> float:
    """Return probability of class-1 for a single sample."""
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X)[0, 1])
    else:
        # LinearSVC — sigmoid of decision function
        score = float(model.decision_function(X)[0])
        return 1.0 / (1.0 + np.exp(-score))


def _ensemble_score(bundle: ModelBundle, X) -> float:
    """
    Compute ensemble score for a single feature vector X.
    Uses stacking meta-classifier if available; falls back to soft voting
    over (LR, SVM, RF).
    """
    models      = [bundle.logistic_regression, bundle.svm, bundle.random_forest]
    model_names = ['LR', 'SVM', 'RF']

    if bundle.stacking_meta_clf is not None:
        # Build meta-features (hard preds + soft probs) for ONE sample
        hard_preds  = np.array([m.predict(X)[0]          for m in models])
        soft_probs  = np.array([_get_model_score(m, X)   for m in models])
        meta_feat   = np.hstack([hard_preds, soft_probs]).reshape(1, -1)
        if hasattr(bundle.stacking_meta_clf, 'predict_proba'):
            return float(bundle.stacking_meta_clf.predict_proba(meta_feat)[0, 1])
        else:
            s = float(bundle.stacking_meta_clf.decision_function(meta_feat)[0])
            return 1.0 / (1.0 + np.exp(-s))
    else:
        # Soft voting fallback
        scores = [_get_model_score(m, X) for m in models]
        return float(np.mean(scores))


# ===========================================================================
# Public API — Model A
# ===========================================================================

def verify_answer(bundle: ModelBundle,
                  article: str,
                  question: str,
                  option: str) -> Tuple[bool, float]:
    """
    Verify whether a single option is the correct answer.

    Parameters
    ----------
    bundle   : loaded ModelBundle
    article  : reading passage (raw text)
    question : question text
    option   : candidate answer text to verify

    Returns
    -------
    is_correct  : bool   — True if model predicts this option is correct
    confidence  : float  — model confidence in the positive (correct) prediction
    """
    t0 = time.time()

    article_c  = _clean_a(article)
    question_c = _clean_a(question)
    option_c   = _clean_a(option)

    X = _vectorize_single(article_c, question_c, option_c, bundle.vectorizer)

    score      = _ensemble_score(bundle, X)
    is_correct = score >= 0.5
    latency    = time.time() - t0

    bundle.session_log.append({
        'task': 'verify_answer', 'latency': latency,
        'prediction': int(is_correct), 'confidence': score
    })

    return bool(is_correct), float(score)


def identify_best_answer(bundle: ModelBundle,
                         article: str,
                         question: str,
                         options: Dict[str, str]) -> Tuple[str, float, Dict[str, float]]:
    """
    Given article, question, and a dict of options {label: text}, return the
    label the ensemble scores highest.

    Parameters
    ----------
    bundle   : loaded ModelBundle
    article  : reading passage
    question : question text
    options  : dict mapping label → option text, e.g. {'A': '...', 'B': '...'}

    Returns
    -------
    best_label  : str            — predicted correct option label (A/B/C/D)
    confidence  : float          — score for best_label
    all_scores  : Dict[str, float] — scores for all labels
    """
    t0 = time.time()

    article_c  = _clean_a(article)
    question_c = _clean_a(question)

    all_scores = {}
    for label, text in options.items():
        option_c          = _clean_a(text)
        X                 = _vectorize_single(article_c, question_c,
                                              option_c, bundle.vectorizer)
        all_scores[label] = _ensemble_score(bundle, X)

    best_label = max(all_scores, key=all_scores.get)
    confidence = all_scores[best_label]
    latency    = time.time() - t0

    bundle.session_log.append({
        'task': 'identify_best_answer', 'latency': latency,
        'prediction': best_label, 'confidence': confidence,
        'all_scores': all_scores
    })

    return best_label, confidence, all_scores


def generate_question(bundle: ModelBundle,
                      article: str,
                      options: Dict[str, str]
                      ) -> Tuple[str, str, float, Dict[str, float]]:
    """
    Generate a question for the article and identify the best answer.

    Uses the trained question_ranker from model_a_generate.py to score
    template-generated candidate questions, then runs identify_best_answer.

    Parameters
    ----------
    bundle   : loaded ModelBundle
    article  : reading passage (raw text, punctuation preserved)
    options  : dict mapping label → option text

    Returns
    -------
    question    : str   — best generated question
    best_label  : str   — predicted correct label (A/B/C/D)
    confidence  : float — verifier confidence
    all_scores  : dict  — per-label scores
    """
    t0 = time.time()

    if bundle.question_ranker is None:
        # No ranker — fall back to a generic question
        question = "What is the main idea of the passage?"
    else:
        all_option_texts = ' '.join(options.values())
        cand_sents       = extract_candidate_sentences(article, all_option_texts,
                                                       top_k=8)
        if not cand_sents:
            cand_sents = _split_sentences_a(article)[:3]

        candidates = generate_candidate_questions(cand_sents)

        if not candidates:
            question = "What is the main idea of the passage?"
        else:
            feat_matrix = np.array([
                question_features(q, src, article, all_option_texts)
                for q, src in candidates
            ], dtype=np.float32)

            if hasattr(bundle.question_ranker, 'predict_proba'):
                scores = bundle.question_ranker.predict_proba(feat_matrix)[:, 1]
            else:
                scores = bundle.question_ranker.decision_function(feat_matrix)

            best_idx = int(np.argmax(scores))
            question = candidates[best_idx][0]

    # Now identify the best answer using the verifier
    best_label, confidence, all_scores = identify_best_answer(
        bundle, article, question, options
    )

    latency = time.time() - t0
    bundle.session_log.append({
        'task': 'generate_question', 'latency': latency,
        'question': question, 'prediction': best_label, 'confidence': confidence
    })

    return question, best_label, confidence, all_scores


# ===========================================================================
# Public API — Model B
# ===========================================================================

def generate_distractors(bundle: ModelBundle,
                         article: str,
                         question: str,
                         correct_answer: str,
                         n: int = 3) -> List[str]:
    """
    Generate n plausible distractor options using Model B.

    Parameters
    ----------
    bundle         : loaded ModelBundle
    article        : reading passage
    question       : question text
    correct_answer : gold correct answer text
    n              : number of distractors to return (default 3)

    Returns
    -------
    list of distractor strings (length up to n)
    """
    t0 = time.time()

    distractors = _gen_distractors(
        article        = article,
        question       = question,
        correct_answer = correct_answer,
        vectorizer     = bundle.vectorizer,
        w2v_model      = bundle.w2v_model,
        n              = n,
    )

    latency = time.time() - t0
    bundle.session_log.append({
        'task': 'generate_distractors', 'latency': latency,
        'n_generated': len(distractors)
    })

    return distractors


def generate_hints(bundle: ModelBundle,
                   article: str,
                   question: str,
                   correct_answer: str = "") -> List[str]:
    """
    Generate 3 graduated hints using Model B hint scorer.

    Parameters
    ----------
    bundle         : loaded ModelBundle
    article        : reading passage
    question       : question text
    correct_answer : gold answer text (used for redaction in Hint 3)

    Returns
    -------
    list of 3 hint strings in ascending specificity order
    """
    t0 = time.time()

    # Note: _gen_hints uses OHE cosine similarity internally, not the
    # CountVectorizer — vectorizer kwarg is intentionally omitted.
    hints = _gen_hints(
        article        = article,
        question       = question,
        correct_answer = correct_answer,
        hint_scorer    = bundle.hint_scorer,
    )

    latency = time.time() - t0
    # Log whether Hint 3 contains the redaction marker '_____' — a quick proxy
    # for whether the near-explicit cloze hint was actually generated vs the
    # generic fallback string.  Useful in the analytics dashboard.
    hint3_is_cloze = (len(hints) >= 3 and '_____' in hints[2])
    bundle.session_log.append({
        'task': 'generate_hints', 'latency': latency,
        'n_hints': len(hints), 'hint3_is_cloze': hint3_is_cloze,
    })

    return hints


# ===========================================================================
# Full pipeline — convenience wrapper for the UI
# ===========================================================================
@dataclass
class PipelineResult:
    """All outputs from a single run_full_pipeline call."""
    question:          str
    options:           Dict[str, str]          # label -> text
    correct_label:     str                     # gold or predicted
    all_scores:        Dict[str, float]        # per-label verifier scores
    confidence:        float
    distractors:       List[str]               # 3 wrong options from Model B
    hints:             List[str]               # 3 graduated hints
    latency_total:     float                   # wall-clock seconds
    source:            str = "generated"       # "generated" or "race_original"


def run_full_pipeline(bundle: ModelBundle,
                      article: str,
                      options: Optional[Dict[str, str]] = None,
                      gold_question: Optional[str] = None,
                      gold_answer_label: Optional[str] = None
                      ) -> PipelineResult:
    """
    Run the complete Model A + Model B pipeline on one article.

    Two modes
    ---------
    Generated mode (options provided, no gold_question):
        Model A generates a question and picks the best answer.
        Model B generates distractors and hints for that answer.

    RACE mode (gold_question + gold_answer_label provided):
        Uses the gold question and answer directly.
        Model A verifier scores all options (for analytics).
        Model B generates distractors and hints for the gold answer.

    Parameters
    ----------
    bundle            : loaded ModelBundle
    article           : reading passage (raw text)
    options           : dict {label: text} — required for Generated mode
    gold_question     : gold question text (RACE mode)
    gold_answer_label : gold answer label A/B/C/D (RACE mode)

    Returns
    -------
    PipelineResult dataclass
    """
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Determine mode
    # -----------------------------------------------------------------------
    if gold_question is not None and options is not None:
        # RACE mode: use gold question, score options with verifier
        question = gold_question
        pred_label, confidence, all_scores = identify_best_answer(
            bundle, article, question, options
        )
        correct_label = gold_answer_label if gold_answer_label else pred_label
        source        = "race_original"

    elif options is not None:
        # Generated mode: generate question, identify best answer
        question, pred_label, confidence, all_scores = generate_question(
            bundle, article, options
        )
        correct_label = pred_label
        source        = "generated"

    else:
        raise ValueError(
            "run_full_pipeline requires either 'options' (Generated mode) "
            "or both 'gold_question' and 'options' (RACE mode)."
        )

    # -----------------------------------------------------------------------
    # Get the correct answer text for Model B
    # -----------------------------------------------------------------------
    correct_answer_text = options.get(correct_label, "")

    # -----------------------------------------------------------------------
    # Model B: generate distractors and hints
    # -----------------------------------------------------------------------
    distractors = generate_distractors(
        bundle, article, question, correct_answer_text
    )
    hints = generate_hints(
        bundle, article, question, correct_answer_text
    )

    latency = time.time() - t0

    return PipelineResult(
        question      = question,
        options       = options,
        correct_label = correct_label,
        all_scores    = all_scores,
        confidence    = confidence,
        distractors   = distractors,
        hints         = hints,
        latency_total = latency,
        source        = source,
    )


# ===========================================================================
# Analytics helpers (used by the Developer Dashboard screen)
# ===========================================================================

def get_session_stats(bundle: ModelBundle) -> dict:
    """
    Summarise the session log into metrics for the analytics dashboard.

    Returns
    -------
    dict with keys:
        total_requests, avg_latency, per_task_counts, per_task_avg_latency
    """
    log = bundle.session_log
    if not log:
        return {
            'total_requests':        0,
            'avg_latency':           0.0,
            'per_task_counts':       {},
            'per_task_avg_latency':  {},
        }

    total     = len(log)
    avg_lat   = float(np.mean([e['latency'] for e in log]))

    tasks     = set(e['task'] for e in log)
    per_count = {t: sum(1 for e in log if e['task'] == t) for t in tasks}
    per_lat   = {
        t: float(np.mean([e['latency'] for e in log if e['task'] == t]))
        for t in tasks
    }

    # Fraction of hint calls where Hint 3 was a genuine cloze (not a fallback)
    hint_entries  = [e for e in log if e['task'] == 'generate_hints']
    cloze_rate    = (
        float(np.mean([e.get('hint3_is_cloze', False) for e in hint_entries]))
        if hint_entries else 0.0
    )

    return {
        'total_requests':       total,
        'avg_latency':          avg_lat,
        'per_task_counts':      per_count,
        'per_task_avg_latency': per_lat,
        'hint3_cloze_rate':     cloze_rate,
    }


def export_session_log(bundle: ModelBundle,
                       path: str = 'data/processed/session_log.csv'):
    """Export the session log to a CSV file."""
    if not bundle.session_log:
        print("Session log is empty — nothing to export.")
        return
    df = pd.DataFrame(bundle.session_log)
    df.to_csv(path, index=False)
    print(f"Session log exported -> {path}")


def clear_session_log(bundle: ModelBundle):
    """Reset the session log."""
    bundle.session_log.clear()


# ===========================================================================
# Quick smoke-test (run as a script: python src/inference.py)
# ===========================================================================
def _smoke_test():
    """
    Load models and run the full pipeline on 3 samples from val.csv.
    Useful for a quick sanity check after training.
    """
    print("\n" + "=" * 60)
    print("INFERENCE SMOKE TEST")
    print("=" * 60)

    bundle = load_models(verbose=True)

    val_df = pd.read_csv('data/raw/val.csv')
    sample = val_df.sample(3, random_state=99)

    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        article    = str(row['article'])
        question   = str(row['question'])
        gold_label = str(row['answer']).strip().upper()
        options    = {k: str(row[k]) for k in ('A', 'B', 'C', 'D')}

        print(f"\n{'='*55}")
        print(f"[Sample {idx}]")
        print(f"  Gold question : {question[:90]}")

        # --- RACE mode ---
        result = run_full_pipeline(
            bundle,
            article           = article,
            options           = options,
            gold_question     = question,
            gold_answer_label = gold_label,
        )

        print(f"  Gold answer   : {gold_label} — {options[gold_label][:60]}")
        print(f"  Predicted     : {result.correct_label} "
              f"(confidence {result.confidence:.3f})")
        print(f"  Correct?      : {result.correct_label == gold_label}")
        print(f"  All scores    : "
              + "  ".join(f"{k}={v:.3f}" for k, v in result.all_scores.items()))

        print(f"\n  Distractors:")
        for i, d in enumerate(result.distractors, 1):
            print(f"    {i}. {d}")

        print(f"\n  Hints:")
        for i, h in enumerate(result.hints, 1):
            print(f"    Hint {i}: {h[:110]}")

        print(f"\n  Pipeline latency: {result.latency_total:.3f}s")

    print(f"\n{'='*55}")
    print("Session stats:")
    stats = get_session_stats(bundle)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    export_session_log(bundle)
    print("\nSmoke test complete.")


if __name__ == '__main__':
    _smoke_test()