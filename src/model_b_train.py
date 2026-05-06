import joblib
import numpy as np
import pandas as pd
import os
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report,
                             r2_score, mean_squared_error)
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

# Paths 
PROCESSED_DIR = 'data/processed/'
MODEL_DIR     = 'models/model_b/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_DIR = 'data/raw/'

#  Stop Words (simple list, no NLTK needed) 
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

#  Load Raw Data 
def load_raw_data():
    print("Loading raw data...")
    train_df = pd.read_csv(RAW_DIR + 'train.csv')
    val_df   = pd.read_csv(RAW_DIR + 'val.csv')
    print(f"Train: {train_df.shape}, Val: {val_df.shape}")
    return train_df, val_df

#  Text Utilities 
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return [w for w in clean_text(text).split() if w not in STOP_WORDS and len(w) > 2]

def get_content_words(text):
    """Return high-frequency non-stop content words from text."""
    tokens = tokenize(text)
    freq   = Counter(tokens)
    return freq

def char_ngram_overlap(s1, s2, n=3):
    """Character n-gram overlap."""
    if len(s1) < n or len(s2) < n: return 0.0
    ngrams1 = set([s1[i:i+n] for i in range(len(s1)-n+1)])
    ngrams2 = set([s2[i:i+n] for i in range(len(s2)-n+1)])
    if not ngrams1 or not ngrams2: return 0.0
    return len(ngrams1 & ngrams2) / min(len(ngrams1), len(ngrams2))

def cosine_sim_bow(tokens1, tokens2):
    """Cosine similarity of bag-of-words (OHE)."""
    vocab = set(tokens1) | set(tokens2)
    if not vocab: return 0.0
    v1 = np.array([1 if w in tokens1 else 0 for w in vocab])
    v2 = np.array([1 if w in tokens2 else 0 for w in vocab])
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

#  Candidate Distractor Extraction 
# Strategy: extract all unique content word phrases from the article
# that are NOT in the correct answer, then rank by frequency

def extract_candidates(article, correct_answer, top_n=20):
    """
    Extract candidate distractor words/phrases from the article.
    Returns top_n candidates sorted by frequency.
    """
    article_clean  = clean_text(article)
    answer_clean   = clean_text(correct_answer)
    answer_tokens  = set(answer_clean.split())

    # Get all content words from article
    article_freq = get_content_words(article_clean)

    # Filter out words that appear in the correct answer
    candidates = {
        word: freq for word, freq in article_freq.items()
        if word not in answer_tokens and freq >= 2
    }

    # Sort by frequency descending
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_candidates[:top_n]]

#  Feature Engineering for Distractor Ranking 
# For each candidate we compute 5 features:
#   f1: overlap with correct answer tokens (low = good distractor)
#   f2: candidate frequency in article (normalized)
#   f3: candidate length in characters (normalized)
#   f4: overlap with question tokens (high = more relevant)
#   f5: position score — earlier in article = more prominent

def compute_distractor_features(candidate, article, question, correct_answer):
    cand_tokens   = set(tokenize(candidate))
    answer_tokens = set(tokenize(correct_answer))
    question_tokens = set(tokenize(question))
    article_tokens  = tokenize(article)
    article_set     = set(article_tokens)
    article_freq    = Counter(article_tokens)
    article_len     = max(len(article_tokens), 1)

    # f1: OHE cosine similarity with correct answer (lower = better distractor)
    f1 = cosine_sim_bow(cand_tokens, answer_tokens)

    # f2: frequency in article (normalized)
    f2 = sum(article_freq.get(w, 0) for w in cand_tokens) / article_len

    # f3: length of candidate (normalized)
    f3 = len(candidate) / 50.0

    # f4: overlap with question
    f4 = len(cand_tokens & question_tokens) / (len(question_tokens) + 1)

    # f5: position of first occurrence in article (earlier = higher score)
    words = article.lower().split()
    positions = [i for i, w in enumerate(words) if w == candidate]
    f5 = 1.0 - (positions[0] / len(words)) if positions else 0.0

    # f6: character-level match score (n-gram overlap)
    f6 = char_ngram_overlap(candidate.replace(' ', ''), correct_answer.replace(' ', ''), n=3)

    return [f1, f2, f3, f4, f5, f6]

#  Build Training Data for Distractor Ranker 
# For each row: extract candidates, label the actual wrong options as 1
# (good distractors) and random non-answer words as 0 (poor distractors)

def build_distractor_training_data(df, sample_size=10000):
    print(f"\nBuilding distractor training data (sample={sample_size})...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    X_rows  = []
    y_rows  = []

    for _, row in df_sample.iterrows():
        correct_col = row['answer']           # 'A', 'B', 'C', or 'D'
        correct_ans = clean_text(str(row[correct_col]))
        article     = str(row['article'])
        question    = str(row['question'])

        # The actual wrong options are ground-truth good distractors (label=1)
        wrong_options = [clean_text(str(row[opt]))
                         for opt in ['A', 'B', 'C', 'D'] if opt != correct_col]

        # Extract candidates from article
        candidates = extract_candidates(article, correct_ans, top_n=15)

        for cand in candidates:
            feats = compute_distractor_features(cand, article, question, correct_ans)
            # Label 1 if candidate closely matches a real wrong option, else 0
            is_good = int(any(cand in opt or opt in cand for opt in wrong_options))
            X_rows.append(feats)
            y_rows.append(is_good)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=int)
    print(f"Distractor dataset → X: {X.shape}, positives: {y.sum()}, negatives: {(y==0).sum()}")
    return X, y

#  Train Distractor Ranker 
def train_distractor_ranker(X_train, y_train):
    print("\nTraining Distractor Ranker (Logistic Regression)...")
    model = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR + 'distractor_ranker_lr.pkl')
    print("Saved → models/model_b/traditional/distractor_ranker_lr.pkl")
    return model

def train_distractor_ranker_rf(X_train, y_train):
    print("\nTraining Distractor Ranker (Random Forest)...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR + 'distractor_ranker_rf.pkl')
    print("Saved → models/model_b/traditional/distractor_ranker_rf.pkl")
    return model

#  Evaluate Distractor Ranker 
def evaluate_ranker(name, model, X_val, y_val):
    preds = model.predict(X_val)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds, average='macro', zero_division=0)
    prec  = precision_score(y_val, preds, average='macro', zero_division=0)
    rec   = recall_score(y_val, preds, average='macro', zero_division=0)
    cm    = confusion_matrix(y_val, preds)

    print(f"\n{'='*50}")
    print(f"Distractor Ranker: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_val, preds, zero_division=0)}")

    return {'name': name, 'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec, 'confusion_matrix': cm}

def evaluate_regressor(name, model, X_val, y_val):
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    
    print(f"\n{'='*50}")
    print(f"Hint Scorer (Regression): {name}")
    print(f"{'='*50}")
    print(f"  R2 Score  : {r2:.4f}")
    print(f"  MSE       : {mse:.4f}")
    
    return {'name': name, 'accuracy': np.nan, 'f1': np.nan,
            'precision': np.nan, 'recall': r2, 'confusion_matrix': None}

#  Generate Distractors for a Single Example 
def generate_distractors(article, question, correct_answer, ranker, n=3):
    """
    Given article, question, correct answer → return top-n distractors.
    """
    candidates = extract_candidates(article, correct_answer, top_n=20)
    if not candidates:
        return ["Option X", "Option Y", "Option Z"][:n]

    # Score each candidate
    features = np.array([
        compute_distractor_features(c, article, question, correct_answer)
        for c in candidates
    ], dtype=np.float32)

    scores = ranker.predict_proba(features)[:, 1]   # probability of being good distractor

    # Sort by score descending, apply diversity penalty
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Select top-n with diversity: skip candidates too similar to already selected
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

    # Pad if not enough candidates
    while len(selected) < n:
        selected.append("other option")

    return selected

#  Hint Generation 
# Extractive hint generation: score each sentence by keyword overlap
# with the question, return top-3 as graduated hints

def build_hint_training_data(df, sample_size=5000):
    print(f"\nBuilding hint training data (sample={sample_size})...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    X_rows = []
    y_rows = []

    for _, row in df_sample.iterrows():
        article      = str(row['article'])
        question     = str(row['question'])
        correct_col  = row['answer']
        correct_ans  = str(row[correct_col])

        sentences    = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 20]
        question_tok = set(tokenize(question))
        answer_tok   = set(tokenize(correct_ans))

        for i, sent in enumerate(sentences):
            sent_tok = set(tokenize(sent))
            sent_len = len(sent_tok)

            # Features for hint scoring
            f1 = cosine_sim_bow(sent_tok, question_tok)  # Extractive cosine sim
            f2 = len(sent_tok & answer_tok)   / (len(answer_tok) + 1)    # ans overlap
            f3 = 1.0 - (i / max(len(sentences), 1))                       # position (earlier = better)
            f4 = min(sent_len / 30.0, 1.0)                                # length score
            f5 = len(sent_tok) / (len(question_tok) + len(answer_tok) + 1) # coverage

            # Label: continuous score representing hint relevance regression target
            label = f2

            X_rows.append([f1, f3, f4, f5])  # Removed f2 to prevent data leakage
            y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    print(f"Hint dataset → X: {X.shape}, continuous targets: {y.mean():.4f} avg score")
    return X, y

def train_hint_scorer(X_train, y_train):
    print("\nTraining Hint Scorer (Ridge Regression)...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR + 'hint_scorer.pkl')
    print("Saved → models/model_b/traditional/hint_scorer.pkl")
    return model

def generate_hints(article, question, correct_answer, hint_scorer, n_hints=3):
    """
    Generate graduated hints from most general to most specific.
    Hint 1 → least revealing, Hint 3 → near-explicit.
    """
    sentences   = [s.strip() for s in re.split(r'[.!?]', article) if len(s.strip()) > 20]
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

        features = np.array([[f1, f3, f4, f5]], dtype=np.float32)
        score    = hint_scorer.predict(features)[0]
        scored.append((sent, score, f2))   # sentence, hint_score, answer_overlap

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Graduated hints:
    # Hint 1: high score but low answer overlap (general)
    # Hint 2: medium answer overlap
    # Hint 3: highest answer overlap (most explicit)

    general  = [s for s, sc, ao in scored if ao < 0.3]
    medium   = [s for s, sc, ao in scored if 0.3 <= ao < 0.6]
    specific = [s for s, sc, ao in scored if ao >= 0.6]

    hint1 = general[0]  if general  else scored[0][0]
    hint2 = medium[0]   if medium   else (scored[1][0] if len(scored) > 1 else hint1)
    hint3 = specific[0] if specific else (scored[2][0] if len(scored) > 2 else hint2)

    return [hint1, hint2, hint3][:n_hints]

#  Demo on Sample Examples 
def run_demo(val_df, distractor_ranker, hint_scorer, n_examples=3):
    print(f"\n{'='*60}")
    print("DEMO — Sample Distractor & Hint Generation")
    print(f"{'='*60}")

    samples = val_df.sample(n=n_examples, random_state=99)

    for i, (_, row) in enumerate(samples.iterrows()):
        correct_col = row['answer']
        correct_ans = str(row[correct_col])
        article     = str(row['article'])
        question    = str(row['question'])

        print(f"\n--- Example {i+1} ---")
        print(f"Question : {question[:100]}...")
        print(f"Correct  : {correct_ans}")

        distractors = generate_distractors(article, question, correct_ans, distractor_ranker)
        print(f"Distractors:")
        for j, d in enumerate(distractors):
            print(f"  {chr(65+j)}) {d}")

        hints = generate_hints(article, question, correct_ans, hint_scorer)
        print(f"Hints:")
        for j, h in enumerate(hints):
            print(f"  Hint {j+1}: {h[:120]}...")

#  Main 
def main():
    train_df, val_df = load_raw_data()

    #  Distractor Ranker 
    X_dist_train, y_dist_train = build_distractor_training_data(train_df, sample_size=10000)
    X_dist_val,   y_dist_val   = build_distractor_training_data(val_df,   sample_size=2000)

    lr_ranker = train_distractor_ranker(X_dist_train, y_dist_train)
    rf_ranker = train_distractor_ranker_rf(X_dist_train, y_dist_train)

    dist_results = []
    dist_results.append(evaluate_ranker("LR Distractor Ranker",  lr_ranker, X_dist_val, y_dist_val))
    dist_results.append(evaluate_ranker("RF Distractor Ranker",  rf_ranker, X_dist_val, y_dist_val))

    #  Hint Scorer 
    X_hint_train, y_hint_train = build_hint_training_data(train_df, sample_size=5000)
    X_hint_val,   y_hint_val   = build_hint_training_data(val_df,   sample_size=1000)

    hint_scorer = train_hint_scorer(X_hint_train, y_hint_train)
    dist_results.append(evaluate_regressor("Hint Scorer (Ridge)", hint_scorer, X_hint_val, y_hint_val))

    #  Comparison Table 
    print(f"\n{'='*60}")
    print(f"{'MODEL B RESULTS SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Accuracy':>10} {'Macro F1':>10} {'Precision':>10} {'Recall/R2':>10}")
    print(f"{'-'*60}")
    for r in dist_results:
        acc = f"{r['accuracy']:.4f}" if pd.notnull(r['accuracy']) else "N/A"
        f1 = f"{r['f1']:.4f}" if pd.notnull(r['f1']) else "N/A"
        prec = f"{r['precision']:.4f}" if pd.notnull(r['precision']) else "N/A"
        rec = f"{r['recall']:.4f}" if pd.notnull(r['recall']) else "N/A"
        print(f"{r['name']:<30} {acc:>10} {f1:>10} {prec:>10} {rec:>10}")
    print(f"{'='*60}")

    #  Demo 
    best_ranker = lr_ranker if dist_results[0]['f1'] >= dist_results[1]['f1'] else rf_ranker
    run_demo(val_df, best_ranker, hint_scorer)

    #  Save Results 
    summary = [{k: v for k, v in r.items() if k != 'confusion_matrix'}
               for r in dist_results]
    pd.DataFrame(summary).to_csv(
        PROCESSED_DIR + 'model_b_results.csv', index=False)
    print("\nResults saved → data/processed/model_b_results.csv")
    print("\nModel B training complete.")

if __name__ == '__main__':
    main()