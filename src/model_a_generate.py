import joblib
import numpy as np
import pandas as pd
import os
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

# ===========================================================================
# Paths
# ===========================================================================
PROCESSED_DIR = 'data/processed/'
MODEL_DIR     = 'models/model_a/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================================================================
# Wh-Word Templates (Section 4.2.3)
#
# Each template transforms a candidate sentence into a question.
# The trigger word determines which template is applied based on
# the first meaningful content word found in the sentence.
# ===========================================================================
WH_TEMPLATES = {
    'who':   "Who {predicate}?",
    'what':  "What {predicate}?",
    'where': "Where {predicate}?",
    'when':  "When {predicate}?",
    'why':   "Why {predicate}?",
    'how':   "How {predicate}?",
}

# Words that hint at which Wh-word to use
WH_TRIGGERS = {
    'who':   {'person', 'people', 'man', 'woman', 'boy', 'girl', 'he',
              'she', 'they', 'author', 'teacher', 'student', 'doctor',
              'mother', 'father', 'friend', 'family', 'children', 'child'},
    'where': {'place', 'city', 'country', 'town', 'school', 'home',
              'house', 'room', 'street', 'store', 'park', 'office',
              'here', 'there', 'location', 'area', 'region', 'building'},
    'when':  {'day', 'year', 'time', 'morning', 'evening', 'night',
              'week', 'month', 'hour', 'today', 'yesterday', 'ago',
              'soon', 'later', 'before', 'after', 'during', 'while'},
    'why':   {'because', 'reason', 'therefore', 'cause', 'result',
              'since', 'thus', 'hence', 'purpose', 'goal', 'aim'},
    'how':   {'way', 'method', 'process', 'manner', 'means', 'step',
              'approach', 'technique', 'strategy', 'plan', 'procedure'},
}

# Common verbs used for heuristic predicate extraction
COMMON_VERBS = {
    'is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'did', 'do',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
    'works', 'lives', 'uses', 'makes', 'takes', 'says', 'goes', 'gets',
    'knows', 'thinks', 'sees', 'wants', 'comes', 'looks', 'gives', 'finds',
    'shows', 'tells', 'asks', 'seems', 'feels', 'leaves', 'puts', 'brings',
    'begins', 'keeps', 'holds', 'writes', 'stands', 'hears', 'likes', 'loves',
    'involves', 'means', 'requires'
}

# Common English stopwords to ignore during overlap computation
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'we', 'you',
    'he', 'she', 'they', 'my', 'your', 'his', 'her', 'our', 'their',
    'not', 'no', 'so', 'if', 'as', 'up', 'out', 'about', 'into', 'then'
}

# ===========================================================================
# Text Utilities
# ===========================================================================
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Return list of lowercase tokens, punctuation stripped."""
    return re.findall(r'[a-z0-9]+', text.lower())

def content_words(text):
    """Return set of non-stopword tokens."""
    return {w for w in tokenize(text) if w not in STOPWORDS}

def split_sentences(text):
    """Split article into sentences on period/exclamation/question mark."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]

# ===========================================================================
# Step 1 — Candidate Sentence Extraction (Section 4.2.3)
#
# Rank all sentences in the article by their One-Hot word overlap with the
# correct answer. The top-K sentences are most likely to contain the answer
# and make good question stems.
# ===========================================================================
def extract_candidate_sentences(article, answer, top_k=5):
    """
    Return the top_k sentences from article ranked by word overlap
    with the answer option (One-Hot keyword overlap as per spec).
    """
    sentences    = split_sentences(article)
    answer_words = content_words(answer)

    if not sentences:
        return []

    scored = []
    for sent in sentences:
        sent_words = content_words(sent)
        if not answer_words:
            overlap = 0.0
        else:
            overlap = len(sent_words & answer_words) / len(answer_words)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_k]]

# ===========================================================================
# Step 2 — Template Application (Section 4.2.3)
#
# Transform each candidate sentence into candidate questions using
# Wh-word templates. The Wh-word is chosen based on content words in the
# sentence that match known trigger word sets.
# ===========================================================================
def pick_wh_word(sentence):
    """
    Scan sentence tokens for trigger words and return the best Wh-word.
    Falls back to 'what' if no trigger is found.
    """
    words = set(tokenize(sentence))
    for wh, triggers in WH_TRIGGERS.items():
        if words & triggers:
            return wh
    return 'what'   # safe default

def apply_template(sentence, wh_word=None):
    """
    Transform a sentence into a question using heuristic predicate extraction.
    Identifies the main verb, removes the subject, and determines if 'Who' 
    should be used based on proper nouns or person-words in the subject.
    """
    tokens = sentence.split()
    if not tokens:
        return ""

    # Remove trailing punctuation for cleaner verb matching
    if tokens[-1][-1] in {'.', '!', '?'}:
        tokens[-1] = tokens[-1][:-1]

    # 1. Better Predicate Extraction: Find the first verb
    verb_idx = -1
    for i, tok in enumerate(tokens):
        w_clean = re.sub(r'[^a-z0-9]', '', tok.lower())
        if w_clean in COMMON_VERBS:
            verb_idx = i
            break

    if verb_idx == -1:
        # Fallback if no known verb is found
        clean_tokens = tokenize(sentence)
        start = 0
        for i, tok in enumerate(clean_tokens):
            if tok not in STOPWORDS:
                start = i
                break
        predicate = ' '.join(clean_tokens[start:]) if start < len(clean_tokens) else ""
        if wh_word is None:
            wh_word = pick_wh_word(sentence)
    else:
        subject_tokens = tokens[:verb_idx]
        verb_token = tokens[verb_idx]
        rest_tokens = tokens[verb_idx+1:]

        # 3. Refining triggers for "Who"
        is_who = False
        for st in subject_tokens:
            st_clean = re.sub(r'[^A-Za-z0-9]', '', st)
            if not st_clean: continue
            
            # Proper noun (capitalized, not a common article)
            if st_clean[0].isupper() and st_clean.lower() not in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'many', 'all'}:
                is_who = True
                break
            # Specific person-words
            if st_clean.lower() in {'he', 'she', 'author', 'writer', 'mr', 'mrs', 'ms', 'dr'}:
                is_who = True
                break

        if wh_word is None:
            if is_who:
                wh_word = 'who'
            else:
                # Fallback to trigger words for the REST of the sentence
                wh_word = pick_wh_word(' '.join(rest_tokens))
                # Prevent hallucinated "Who" for general groups
                if wh_word == 'who':
                    wh_word = 'what'

        # 2. Subject-Verb Inversion (The "Is" Rule)
        # By removing the subject and moving the verb to the front of the predicate,
        # we naturally form questions like "What is..." or "Who works..."
        predicate = ' '.join([verb_token] + rest_tokens)

    template = WH_TEMPLATES.get(wh_word, WH_TEMPLATES['what'])
    question = template.format(predicate=predicate)
    return question[0].upper() + question[1:] if question else question

def generate_candidate_questions(sentences):
    """
    Apply Wh-word templates intelligently to each candidate sentence.
    Returns list of (question_string, source_sentence) tuples.
    """
    candidates = []
    for sent in sentences:
        # Let apply_template intelligently determine the Wh-word
        q = apply_template(sent)
        if q:
            candidates.append((q, sent))

            # Also generate a 'what' fallback for diversity if different
            if not q.startswith('What'):
                q_what = apply_template(sent, 'what')
                candidates.append((q_what, sent))

    return candidates

# ===========================================================================
# Step 3 — ML Ranker Feature Engineering
#
# For each candidate question, compute features that measure fluency and
# relevance. These are used to train and run the ML ranker.
# ===========================================================================
def question_features(question, source_sentence, article, answer):
    """
    Compute a feature vector for one candidate question:
      f1 — question length (normalised)
      f2 — overlap between question content words and article
      f3 — overlap between question content words and answer
      f4 — overlap between source sentence and answer (relevance signal)
      f5 — whether question starts with a Wh-word (fluency signal)
      f6 — source sentence position in article (earlier = more important)
      f7 — answer word coverage in source sentence
      f8 — cosine similarity between question and answer word vectors
    """
    q_words      = content_words(question)
    art_words    = content_words(article)
    ans_words    = content_words(answer)
    src_words    = content_words(source_sentence)
    sentences    = split_sentences(article)

    # f1: normalised question length
    f1 = len(q_words) / (len(tokenize(article)) + 1)

    # f2: question-article overlap
    f2 = len(q_words & art_words) / (len(q_words) + 1)

    # f3: question-answer overlap
    f3 = len(q_words & ans_words) / (len(ans_words) + 1)

    # f4: source sentence-answer overlap
    f4 = len(src_words & ans_words) / (len(ans_words) + 1)

    # f5: starts with a Wh-word (1 = fluent question form)
    first_word = tokenize(question)[0] if tokenize(question) else ''
    f5 = 1.0 if first_word in WH_TEMPLATES else 0.0

    # f6: sentence position (earlier sentences score higher — intro/topic)
    try:
        pos = next(i for i, s in enumerate(sentences)
                   if source_sentence[:20] in s)
        f6  = 1.0 - (pos / len(sentences))
    except StopIteration:
        f6 = 0.5

    # f7: answer word coverage in source sentence
    f7 = len(src_words & ans_words) / (len(src_words) + 1)

    # f8: cosine similarity between question and answer word vectors
    vocab = q_words | ans_words
    if vocab:
        v1    = np.array([1.0 if w in q_words   else 0.0 for w in vocab])
        v2    = np.array([1.0 if w in ans_words  else 0.0 for w in vocab])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        f8    = float(np.dot(v1, v2) / (norm1 * norm2)) \
                if (norm1 > 0 and norm2 > 0) else 0.0
    else:
        f8 = 0.0

    return [f1, f2, f3, f4, f5, f6, f7, f8]

# ===========================================================================
# Build Training Dataset for the ML Ranker
#
# We use the RACE validation set to build training data for the ranker:
#   - For each question, the RACE original question is the POSITIVE example
#   - All generated candidate questions from templates are NEGATIVE examples
#     (label=0) unless they closely match the gold question (label=1)
#
# This gives the ranker signal to prefer questions that resemble gold ones.
# ===========================================================================
def build_ranker_dataset(df, n_samples=5000):
    """
    Build (features, label) pairs for ranker training.
    df must have columns: article, question, A, B, C, D, answer (gold label).
    Uses up to n_samples rows from df.
    """
    print(f"  Building ranker training data from {n_samples} samples...")
    X_rows, y_rows = [], []

    sample_df = df.sample(min(n_samples, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        article = str(row['article'])
        gold_q  = str(row['question'])
        gold_ans_key = str(row['answer']).strip().upper()
        if gold_ans_key not in ['A', 'B', 'C', 'D']:
            continue
        answer  = str(row[gold_ans_key])

        # Extract candidate sentences
        cand_sents = extract_candidate_sentences(article, answer, top_k=5)
        if not cand_sents:
            continue

        # Generate candidate questions
        candidates = generate_candidate_questions(cand_sents)
        if not candidates:
            continue

        gold_words = content_words(gold_q)

        for q, src_sent in candidates:
            feats = question_features(q, src_sent, article, answer)
            # Label 1 if generated question has high overlap with gold question
            q_words = content_words(q)
            overlap = (len(q_words & gold_words) / len(gold_words)
                       if gold_words else 0.0)
            label   = 1 if overlap >= 0.4 else 0
            X_rows.append(feats)
            y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"  Ranker dataset: {len(X)} samples | "
          f"positive: {y.sum()} | negative: {(y==0).sum()}")
    return X, y

# ===========================================================================
# Train ML Ranker (Section 4.2.3 — Step 3)
#
# We train both a Logistic Regression and a Random Forest ranker and keep
# the better one. The ranker scores each candidate question by fluency and
# relevance; the top-scored question is the final output.
# ===========================================================================
def train_ranker(X, y):
    """Train LR and RF rankers, return the better one by val F1."""
    if len(np.unique(y)) < 2:
        print("  WARNING: Only one class in ranker data. "
              "Using fallback Logistic Regression.")
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X, y)
        return lr, 'Logistic Regression (fallback)'

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression ranker
    print("  Training Logistic Regression ranker...")
    lr = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_tr, y_tr)
    lr_f1 = f1_score(y_v, lr.predict(X_v), average='macro', zero_division=0)
    print(f"  LR Ranker Val Macro F1: {lr_f1:.4f}")

    # Random Forest ranker
    print("  Training Random Forest ranker...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    rf_f1 = f1_score(y_v, rf.predict(X_v), average='macro', zero_division=0)
    print(f"  RF Ranker Val Macro F1: {rf_f1:.4f}")

    if rf_f1 >= lr_f1:
        print("  Selected: Random Forest ranker")
        return rf, 'Random Forest'
    else:
        print("  Selected: Logistic Regression ranker")
        return lr, 'Logistic Regression'

# ===========================================================================
# Answer Identification
#
# Given an article and a generated question, score all 4 candidate answer
# options using the trained answer verifier (LR from model_a_train.py).
# The option with the highest confidence score is the predicted answer.
# ===========================================================================
def identify_answer(article, question, options, verifier_model, vectorizer):
    """
    Score each option using the answer verifier and return the best option
    label (A/B/C/D) and its confidence score.

    options: dict {'A': text, 'B': text, 'C': text, 'D': text}
    """
    option_labels = ['A', 'B', 'C', 'D']
    texts = [
        clean_text(article) + ' ' + clean_text(question) + ' ' + clean_text(options[l])
        for l in option_labels
    ]
    X_ohe = vectorizer.transform(texts)
    dummy_lex = csr_matrix(np.zeros((X_ohe.shape[0], 5), dtype=np.float32))
    X = hstack([X_ohe, dummy_lex])

    if hasattr(verifier_model, 'predict_proba'):
        scores = verifier_model.predict_proba(X)[:, 1]
    else:
        scores = verifier_model.decision_function(X)

    best_idx   = int(np.argmax(scores))
    best_label = option_labels[best_idx]
    return best_label, float(scores[best_idx]), dict(zip(option_labels, scores.tolist()))

# ===========================================================================
# Full Pipeline — Generate Question + Identify Answer
#
# Given one article:
#   1. Extract top candidate sentences by overlap with the correct answer
#   2. Generate candidate questions via Wh-word templates
#   3. Rank candidates using the trained ML ranker
#   4. Use the answer verifier to identify the correct option
#   5. Return the best question + predicted answer label
# ===========================================================================
def generate_question_and_answer(article, options, ranker, verifier, vectorizer):
    """
    Full generation pipeline for one article.

    article : str  — the reading passage
    options : dict — {'A': text, 'B': text, 'C': text, 'D': text}
    ranker  : trained sklearn ranker model
    verifier: trained answer verifier (LR/SVM from model_a_train.py)
    vectorizer: fitted CountVectorizer from preprocessing.py

    Returns:
        question     : str   — best generated question
        pred_answer  : str   — predicted correct answer label (A/B/C/D)
        confidence   : float — verifier confidence for predicted answer
        all_scores   : dict  — verifier scores for all options
    """
    article_clean = clean_text(article)

    # Use all options as candidate answers to extract diverse sentences
    all_option_texts = ' '.join(options.values())
    
    # Extract sentences using RAW article, so punctuation isn't lost
    cand_sents = extract_candidate_sentences(article, all_option_texts, top_k=8)

    if not cand_sents:
        # Fallback: use first sentence of article
        cand_sents = split_sentences(article)[:3]

    # Generate candidate questions
    candidates = generate_candidate_questions(cand_sents)

    if not candidates:
        question = "What is the main idea of the passage?"
    else:
        # Score each candidate with the ranker
        feat_matrix = np.array([
            question_features(q, src, article, all_option_texts)
            for q, src in candidates
        ], dtype=np.float32)

        if hasattr(ranker, 'predict_proba'):
            scores = ranker.predict_proba(feat_matrix)[:, 1]
        else:
            scores = ranker.decision_function(feat_matrix)

        best_idx = int(np.argmax(scores))
        question = candidates[best_idx][0]

    # Identify the correct answer using the verifier
    pred_answer, confidence, all_scores = identify_answer(
        article_clean, question, options, verifier, vectorizer
    )

    return question, pred_answer, confidence, all_scores

# ===========================================================================
# Evaluate on RACE Val/Test Set
#
# Runs the full generation + answer identification pipeline on a sample of
# the dataset and reports answer identification accuracy (how often the
# verifier correctly picks the gold answer option).
# ===========================================================================
def evaluate_pipeline(df, ranker, verifier, vectorizer,
                      n_samples=500, split="Val"):
    """
    df must have columns: article, question (gold), A, B, C, D, answer (gold label)
    Evaluates answer identification accuracy across n_samples questions.
    """
    print(f"\n{'='*55}")
    print(f"PIPELINE EVALUATION [{split}] — {n_samples} samples")
    print(f"{'='*55}")

    sample_df = df.sample(min(n_samples, len(df)), random_state=42)

    correct_answer_id = 0
    generated_questions = []

    for _, row in sample_df.iterrows():
        article = str(row['article'])
        options = {
            'A': clean_text(str(row['A'])),
            'B': clean_text(str(row['B'])),
            'C': clean_text(str(row['C'])),
            'D': clean_text(str(row['D'])),
        }
        gold_answer = str(row['answer']).strip().upper()

        try:
            q, pred_ans, conf, all_scores = generate_question_and_answer(
                article, options, ranker, verifier, vectorizer
            )
        except Exception as e:
            print(f"  WARNING: generation failed for one sample: {e}")
            continue

        if pred_ans == gold_answer:
            correct_answer_id += 1

        generated_questions.append({
            'gold_question':  str(row['question']),
            'generated_q':    q,
            'gold_answer':    gold_answer,
            'pred_answer':    pred_ans,
            'confidence':     conf,
            'correct':        pred_ans == gold_answer
        })

    total = len(generated_questions)
    ans_acc = correct_answer_id / total if total > 0 else 0.0

    print(f"  Samples evaluated      : {total}")
    print(f"  Answer ID Accuracy     : {ans_acc:.4f}  "
          f"({correct_answer_id}/{total} correct)")

    # Show a few examples
    print(f"\n  --- Sample Generated Questions ---")
    for ex in generated_questions[:5]:
        print(f"\n  Gold Q  : {ex['gold_question'][:80]}")
        print(f"  Gen  Q  : {ex['generated_q'][:80]}")
        print(f"  Gold Ans: {ex['gold_answer']}  |  "
              f"Pred Ans: {ex['pred_answer']}  |  "
              f"Correct: {ex['correct']}")

    return generated_questions, ans_acc

# ===========================================================================
# Main
# ===========================================================================
def main():
    print("="*55)
    print("MODEL A — QUESTION GENERATION PIPELINE")
    print("="*55)

    # -----------------------------------------------------------------------
    # Load pre-trained verifier and vectorizer from model_a_train.py
    # -----------------------------------------------------------------------
    print("\nLoading pre-trained verifier and vectorizer...")
    verifier   = joblib.load(MODEL_DIR + 'logistic_regression.pkl')
    vectorizer = joblib.load(PROCESSED_DIR + 'vectorizer.pkl')
    print("  Loaded: logistic_regression.pkl + vectorizer.pkl")

    # -----------------------------------------------------------------------
    # Load raw data for Question Generation
    # We load the raw data so that punctuation is preserved for sentence splitting.
    # -----------------------------------------------------------------------
    print("\nLoading data...")
    train_df = pd.read_csv('data/raw/train.csv')
    val_df   = pd.read_csv('data/raw/val.csv')
    test_df  = pd.read_csv('data/raw/test.csv')

    # Note: We NO LONGER apply clean_text to the entire dataframes here!
    # Removing punctuation destroys the sentence boundaries needed for 
    # extract_candidate_sentences. The generation pipeline internally cleans
    # the text ONLY when scoring with the Answer Verifier.

    # -----------------------------------------------------------------------
    # Build ranker training dataset from raw training CSV
    # -----------------------------------------------------------------------
    print("\nBuilding ML ranker training dataset...")
    X_rank, y_rank = build_ranker_dataset(train_df, n_samples=5000)

    # -----------------------------------------------------------------------
    # Train ML Ranker
    # -----------------------------------------------------------------------
    print("\nTraining ML ranker (Step 3 — Section 4.2.3)...")
    ranker, ranker_name = train_ranker(X_rank, y_rank)
    joblib.dump(ranker, MODEL_DIR + 'question_ranker.pkl')
    print(f"  Saved -> models/model_a/traditional/question_ranker.pkl")
    print(f"  Ranker type: {ranker_name}")

    # -----------------------------------------------------------------------
    # Evaluate full pipeline on Val and Test sets
    # -----------------------------------------------------------------------
    val_results, val_acc = evaluate_pipeline(
        val_df, ranker, verifier, vectorizer,
        n_samples=500, split="Val"
    )
    test_results, test_acc = evaluate_pipeline(
        test_df, ranker, verifier, vectorizer,
        n_samples=500, split="Test"
    )

    # -----------------------------------------------------------------------
    # Save generated questions to CSV for report / human evaluation
    # -----------------------------------------------------------------------
    val_out  = pd.DataFrame(val_results)
    test_out = pd.DataFrame(test_results)

    val_out.to_csv(PROCESSED_DIR  + 'model_a_generated_val.csv',  index=False)
    test_out.to_csv(PROCESSED_DIR + 'model_a_generated_test.csv', index=False)
    print(f"\nGenerated questions saved:")
    print(f"  -> data/processed/model_a_generated_val.csv")
    print(f"  -> data/processed/model_a_generated_test.csv")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"GENERATION PIPELINE SUMMARY")
    print(f"{'='*55}")
    print(f"  Ranker model           : {ranker_name}")
    print(f"  Answer ID Accuracy Val : {val_acc:.4f}")
    print(f"  Answer ID Accuracy Test: {test_acc:.4f}")
    print(f"{'='*55}")
    print("\nQuestion generation complete.")
    print("Next step: run model_a_ensemble.py")

if __name__ == '__main__':
    main()