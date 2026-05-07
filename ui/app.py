import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import time
import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix

# Page Config 
st.set_page_config(
    page_title="RACE Reading Comprehension System",
    page_icon="📚",
    layout="wide"
)

# Paths 
PROCESSED_DIR = 'data/processed/'
MODEL_A_DIR   = 'models/model_a/traditional/'
MODEL_B_DIR   = 'models/model_b/traditional/'
RAW_DIR       = 'data/raw/'

# Stop Words 
STOP_WORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'that', 'this', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
    'we', 'you', 'i', 'my', 'your', 'his', 'her', 'their', 'our',
    'not', 'no', 'so', 'if', 'as', 'up', 'out', 'about', 'into',
    'then', 'than', 'also', 'just', 'more', 'there', 'when', 'which'
])

# Load Models 
@st.cache_resource
def load_models():
    models = {}
    try:
        models['vectorizer']    = joblib.load(PROCESSED_DIR + 'vectorizer.pkl')
        models['lr']            = joblib.load(MODEL_A_DIR   + 'logistic_regression.pkl')
        models['svm']           = joblib.load(MODEL_A_DIR   + 'svm.pkl')
        models['nb']            = joblib.load(MODEL_A_DIR   + 'naive_bayes.pkl')
        models['dist_ranker']   = joblib.load(MODEL_B_DIR   + 'distractor_ranker_lr.pkl')
        models['hint_scorer']   = joblib.load(MODEL_B_DIR   + 'hint_scorer.pkl')
        return models, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_race_sample():
    try:
        df = pd.read_csv(RAW_DIR + 'val.csv')
        return df
    except Exception as e:
        return None

# Text Utilities (same as model_b_train.py) 
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return [w for w in clean_text(text).split()
            if w not in STOP_WORDS and len(w) > 2]

def get_content_words(text):
    tokens = tokenize(text)
    return Counter(tokens)

def char_ngram_overlap(s1, s2, n=3):
    if len(s1) < n or len(s2) < n: return 0.0
    ngrams1 = set([s1[i:i+n] for i in range(len(s1)-n+1)])
    ngrams2 = set([s2[i:i+n] for i in range(len(s2)-n+1)])
    if not ngrams1 or not ngrams2: return 0.0
    return len(ngrams1 & ngrams2) / min(len(ngrams1), len(ngrams2))

def cosine_sim_bow(tokens1, tokens2):
    vocab = set(tokens1) | set(tokens2)
    if not vocab: return 0.0
    v1 = np.array([1 if w in tokens1 else 0 for w in vocab])
    v2 = np.array([1 if w in tokens2 else 0 for w in vocab])
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

# Model A — Answer Verification 
def build_features_for_inference(article, question, option, vectorizer):
    combined = clean_text(article) + ' ' + clean_text(question) + ' ' + clean_text(option)

    # One-Hot features
    X_ohe = vectorizer.transform([combined])

    # Lexical features
    parts         = combined.split()
    total         = len(parts)
    article_end   = int(total * 0.80)
    question_end  = int(total * 0.90)
    article_words = set(parts[:article_end])
    question_words= set(parts[article_end:question_end])
    option_words  = set(parts[question_end:])
    opt_len       = len(option_words)

    f1 = len(article_words  & option_words)  / (opt_len + 1)
    f2 = len(question_words & option_words)  / (opt_len + 1)
    f3 = opt_len / (total + 1)
    f4 = len(article_words) / (total + 1)

    vocab_aq = article_words | question_words
    vocab_all = vocab_aq | option_words
    v1 = np.array([1 if w in vocab_aq else 0 for w in vocab_all])
    v2 = np.array([1 if w in option_words else 0 for w in vocab_all])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    f5 = np.dot(v1, v2) / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

    X_lex = csr_matrix(np.array([[f1, f2, f3, f4, f5]], dtype=np.float32))
    return hstack([X_ohe, X_lex])

def verify_answer(article, question, option, model, vectorizer):
    X = build_features_for_inference(article, question, option, vectorizer)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)[0][1]
    else:
        score = model.decision_function(X)[0]
        prob  = 1 / (1 + np.exp(-score))
    pred = int(prob >= 0.5)
    return pred, prob

# Model B — Distractor Generation 
def extract_candidates(article, correct_answer, top_n=20):
    article_clean = clean_text(article)
    answer_tokens = set(clean_text(correct_answer).split())
    article_freq  = get_content_words(article_clean)
    candidates    = {
        w: f for w, f in article_freq.items()
        if w not in answer_tokens and f >= 2
    }
    sorted_cands  = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_cands[:top_n]]

def compute_distractor_features(candidate, article, question, correct_answer):
    cand_tok    = set(tokenize(candidate))
    answer_tok  = set(tokenize(correct_answer))
    question_tok= set(tokenize(question))
    article_tok = tokenize(article)
    article_freq= Counter(article_tok)
    article_len = max(len(article_tok), 1)

    f1 = cosine_sim_bow(cand_tok, answer_tok)
    f2 = sum(article_freq.get(w, 0) for w in cand_tok) / article_len
    f3 = len(candidate) / 50.0
    f4 = len(cand_tok & question_tok) / (len(question_tok) + 1)
    words     = article.lower().split()
    positions = [i for i, w in enumerate(words) if w == candidate]
    f5 = 1.0 - (positions[0] / len(words)) if positions else 0.0
    f6 = char_ngram_overlap(candidate.replace(' ', ''), correct_answer.replace(' ', ''), n=3)

    return [f1, f2, f3, f4, f5, f6]

def generate_distractors(article, question, correct_answer, ranker, n=3):
    candidates = extract_candidates(article, correct_answer, top_n=20)
    if not candidates:
        return ["Option X", "Option Y", "Option Z"][:n]
    features = np.array([
        compute_distractor_features(c, article, question, correct_answer)
        for c in candidates
    ], dtype=np.float32)
    scores   = ranker.predict_proba(features)[:, 1]
    ranked   = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    selected = []
    selected_tokens = []
    for cand, score in ranked:
        cand_tok = set(tokenize(cand))
        too_similar = any(
            len(cand_tok & prev) / (len(cand_tok) + 1) > 0.5
            for prev in selected_tokens
        )
        if not too_similar:
            selected.append(cand)
            selected_tokens.append(cand_tok)
        if len(selected) == n:
            break
    while len(selected) < n:
        selected.append("other option")
    return selected

# Model B — Hint Generation 
def generate_hints(article, question, correct_answer, hint_scorer, n_hints=3):
    sentences    = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 20]
    question_tok = set(tokenize(question))
    answer_tok   = set(tokenize(correct_answer))
    if not sentences:
        return ["Re-read the passage carefully.",
                "Focus on the key topic of the passage.",
                "The answer is directly stated in the passage."]
    scored = []
    for i, sent in enumerate(sentences):
        sent_tok = set(tokenize(sent))
        f1 = cosine_sim_bow(sent_tok, question_tok)
        f2 = len(sent_tok & answer_tok)   / (len(answer_tok) + 1)
        f3 = 1.0 - (i / max(len(sentences), 1))
        f4 = min(len(sent_tok) / 30.0, 1.0)
        f5 = len(sent_tok) / (len(question_tok) + len(answer_tok) + 1)
        
        features = np.array([[f1, f2, f3, f4, f5]], dtype=np.float32)
        if hasattr(hint_scorer, 'predict_proba'):
            score = hint_scorer.predict_proba(features)[0][1]
        else:
            score = hint_scorer.predict(features)[0]
            
        scored.append((sent, score, f2))
    scored.sort(key=lambda x: x[1], reverse=True)
    general  = [s for s, sc, ao in scored if ao < 0.3]
    medium   = [s for s, sc, ao in scored if 0.3 <= ao < 0.6]
    specific = [s for s, sc, ao in scored if ao >= 0.6]
    hint1 = general[0]  if general  else scored[0][0]
    hint2 = medium[0]   if medium   else (scored[1][0] if len(scored) > 1 else hint1)
    hint3 = specific[0] if specific else (scored[2][0] if len(scored) > 2 else hint2)
    return [hint1, hint2, hint3][:n_hints]

# Session State Init 
def init_session_state():
    defaults = {
        'article': '', 'question': '', 'correct_answer': '',
        'options': [], 'correct_index': -1,
        'selected_option': None, 'checked': False,
        'hints_used': 0, 'answer_revealed': False,
        'inference_log': [],
        'model_a_metrics': {'correct': 0, 'total': 0},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Main App 
def main():
    init_session_state()

    # Load models
    models, error = load_models()
    if error:
        st.error(f"⚠️ Failed to load models: {error}")
        st.stop()

    race_df = load_race_sample()

    # Sidebar Navigation 
    st.sidebar.title("📚 RACE RC System")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["📝 Article Input", "🧠 Quiz", "💡 Hints", "📊 Analytics Dashboard"]
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Answers are AI-generated. Errors are possible.")

    # SCREEN 1 — Article Input
    if page == "📝 Article Input":
        st.title("📝 Article Input")
        st.markdown("Paste a reading passage below or load a random RACE sample.")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🎲 Load Random RACE Sample", use_container_width=True):
                if race_df is not None:
                    sample = race_df.sample(1, random_state=int(time.time()) % 10000).iloc[0]
                    st.session_state['article']        = sample['article']
                    st.session_state['question']       = sample['question']
                    correct_col                        = sample['answer']
                    st.session_state['correct_answer'] = str(sample[correct_col])
                    st.session_state['checked']        = False
                    st.session_state['selected_option']= None
                    st.session_state['hints_used']     = 0
                    st.session_state['answer_revealed'] = False
                    st.success("✅ Random sample loaded!")
                else:
                    st.error("Could not load RACE dataset.")

        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                for key in ['article', 'question', 'correct_answer',
                            'options', 'checked', 'selected_option',
                            'hints_used', 'answer_revealed']:
                    st.session_state[key] = '' if isinstance(
                        st.session_state[key], str) else \
                        ([] if isinstance(st.session_state[key], list) else
                         False if isinstance(st.session_state[key], bool) else 0)

        st.markdown("### Reading Passage")
        article = st.text_area(
            "Paste your article here",
            value=st.session_state['article'],
            height=250,
            placeholder="Paste a reading passage here..."
        )

        st.markdown("### Question")
        question = st.text_input(
            "Question",
            value=st.session_state['question'],
            placeholder="Enter the comprehension question..."
        )

        st.markdown("### Correct Answer")
        correct_answer = st.text_input(
            "Correct Answer",
            value=st.session_state['correct_answer'],
            placeholder="Enter the correct answer..."
        )

        st.markdown("---")
        if st.button("🚀 Submit — Generate Quiz", type="primary", use_container_width=True):
            if not article.strip():
                st.error("⚠️ Please enter or load a reading passage.")
            elif not question.strip():
                st.error("⚠️ Please enter a question.")
            elif not correct_answer.strip():
                st.error("⚠️ Please enter the correct answer.")
            else:
                with st.spinner("Generating quiz options and hints..."):
                    start = time.time()

                    # Generate distractors
                    distractors = generate_distractors(
                        article, question, correct_answer,
                        models['dist_ranker']
                    )

                    # Shuffle correct answer into options
                    options      = distractors[:3] + [correct_answer]
                    np.random.shuffle(options)
                    correct_index = options.index(correct_answer)

                    elapsed = time.time() - start

                    # Save to session state
                    st.session_state['article']         = article
                    st.session_state['question']        = question
                    st.session_state['correct_answer']  = correct_answer
                    st.session_state['options']         = options
                    st.session_state['correct_index']   = correct_index
                    st.session_state['checked']         = False
                    st.session_state['selected_option'] = None
                    st.session_state['hints_used']      = 0
                    st.session_state['answer_revealed'] = False

                    st.success(f"✅ Quiz generated in {elapsed:.2f}s! Go to 🧠 Quiz tab.")

    # SCREEN 2 — Quiz
    elif page == "🧠 Quiz":
        st.title("🧠 Quiz")

        if not st.session_state['article']:
            st.warning("⚠️ No article loaded. Go to 📝 Article Input first.")
            st.stop()

        if not st.session_state['options']:
            st.warning("⚠️ Please submit the article first to generate quiz options.")
            st.stop()

        # Show passage
        with st.expander("📖 Reading Passage", expanded=False):
            st.write(st.session_state['article'])

        st.markdown(f"### ❓ {st.session_state['question']}")
        st.markdown("---")

        # Options
        options       = st.session_state['options']
        correct_index = st.session_state['correct_index']
        labels        = ['A', 'B', 'C', 'D']

        selected = st.radio(
            "Select your answer:",
            options=[f"{labels[i]}) {options[i]}" for i in range(len(options))],
            index=None,
            key="quiz_radio"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            check_clicked = st.button("✅ Check Answer", type="primary",
                                       use_container_width=True,
                                       disabled=selected is None)

        if check_clicked and selected is not None:
            selected_idx = [f"{labels[i]}) {options[i]}"
                            for i in range(len(options))].index(selected)
            selected_option = options[selected_idx]

            with st.spinner("Verifying with Model A..."):
                start = time.time()
                pred, prob = verify_answer(
                    st.session_state['article'],
                    st.session_state['question'],
                    selected_option,
                    models['lr'],
                    models['vectorizer']
                )
                elapsed = time.time() - start

            is_correct = (selected_idx == correct_index)
            model_is_correct = (pred == (1 if is_correct else 0))

            st.session_state['checked']         = True
            st.session_state['selected_option'] = selected_option

            # Update metrics
            st.session_state['model_a_metrics']['total'] += 1
            if model_is_correct:
                st.session_state['model_a_metrics']['correct'] += 1

            # Log inference
            st.session_state['inference_log'].append({
                'question'       : st.session_state['question'][:60] + '...',
                'selected'       : selected_option,
                'correct'        : st.session_state['correct_answer'],
                'user_correct'   : is_correct,
                'model_pred'     : pred,
                'model_prob'     : round(prob, 4),
                'model_correct'  : model_is_correct,
                'latency_s'      : round(elapsed, 3)
            })

            if is_correct:
                st.success(f"🎉 Correct! Model confidence: {prob:.2%}")
                st.balloons()
            else:
                st.error(f"❌ Incorrect. The correct answer was: **{st.session_state['correct_answer']}**")
                st.info(f"Model confidence score: {prob:.2%}")

            st.caption(f"⚠️ This answer was verified by an AI model. Inference time: {elapsed:.3f}s")

    # SCREEN 3 — Hints
    elif page == "💡 Hints":
        st.title("💡 Hint Panel")

        if not st.session_state['article']:
            st.warning("⚠️ No article loaded. Go to 📝 Article Input first.")
            st.stop()

        st.markdown(f"**Question:** {st.session_state['question']}")
        st.markdown("---")

        hints = generate_hints(
            st.session_state['article'],
            st.session_state['question'],
            st.session_state['correct_answer'],
            models['hint_scorer']
        )

        hint_labels = [
            "💬 Hint 1 — General Clue",
            "🔍 Hint 2 — More Specific",
            "🎯 Hint 3 — Near Explicit"
        ]

        hints_used = st.session_state['hints_used']

        for i in range(3):
            if i < hints_used:
                with st.expander(hint_labels[i], expanded=True):
                    st.info(hints[i])
            else:
                st.expander(hint_labels[i], expanded=False)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])
        with col1:
            if hints_used < 3:
                if st.button(f"🔓 Reveal Hint {hints_used + 1}", use_container_width=True):
                    st.session_state['hints_used'] += 1
                    st.rerun()
            else:
                st.success("All hints revealed!")

        with col2:
            if hints_used >= 3 and not st.session_state['answer_revealed']:
                if st.button("🎯 Reveal Answer", type="primary", use_container_width=True):
                    st.session_state['answer_revealed'] = True
                    st.rerun()

        if st.session_state['answer_revealed']:
            st.markdown("---")
            st.success(f"✅ The correct answer is: **{st.session_state['correct_answer']}**")
            st.caption("⚠️ This is an AI-generated answer. Please verify with the passage.")

    # SCREEN 4 — Analytics Dashboard
    elif page == "📊 Analytics Dashboard":
        st.title("📊 Analytics Dashboard")
        st.markdown("Performance metrics from this session.")

        # Model A Metrics 
        st.markdown("### Model A — Answer Verifier Performance")
        metrics = st.session_state['model_a_metrics']
        total   = metrics['total']
        correct = metrics['correct']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Checks",  total)
        col2.metric("Correct",       correct)
        col3.metric("Incorrect",     total - correct)
        col4.metric("Session Accuracy",
                    f"{correct/total:.1%}" if total > 0 else "N/A")

        # Pre-computed Model Results 
        st.markdown("### Model A — Training Evaluation Results")
        try:
            model_a_df = pd.read_csv(PROCESSED_DIR + 'model_a_results.csv')
            st.dataframe(model_a_df.style.highlight_max(
                subset=['accuracy', 'f1'], color='lightgreen'), use_container_width=True)
        except Exception:
            st.warning("model_a_results.csv not found.")

        st.markdown("### Model A — Ensemble Results")
        try:
            ensemble_df = pd.read_csv(PROCESSED_DIR + 'model_a_ensemble_results.csv')
            st.dataframe(ensemble_df, use_container_width=True)
        except Exception:
            st.warning("model_a_ensemble_results.csv not found.")

        st.markdown("### Model B — Distractor & Hint Results")
        try:
            model_b_df = pd.read_csv(PROCESSED_DIR + 'model_b_results.csv')
            st.dataframe(model_b_df.style.highlight_max(
                subset=['accuracy', 'f1'], color='lightblue'), use_container_width=True)
        except Exception:
            st.warning("model_b_results.csv not found.")

        # Inference Log 
        st.markdown("### Inference Log")
        if st.session_state['inference_log']:
            log_df = pd.DataFrame(st.session_state['inference_log'])
            st.dataframe(log_df, use_container_width=True)

            # Export button
            csv = log_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Export Log to CSV",
                data=csv,
                file_name='session_inference_log.csv',
                mime='text/csv',
                use_container_width=True
            )
        else:
            st.info("No inferences yet. Answer some questions in the 🧠 Quiz tab first.")

        # Latency Chart 
        if st.session_state['inference_log']:
            st.markdown("### Inference Latency (seconds)")
            log_df = pd.DataFrame(st.session_state['inference_log'])
            st.line_chart(log_df['latency_s'])

if __name__ == '__main__':
    main()