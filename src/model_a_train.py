import joblib
import numpy as np
import pandas as pd
import os
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)

# ===========================================================================
# Paths
# ===========================================================================
PROCESSED_DIR = 'data/processed/'
MODEL_DIR     = 'models/model_a/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================================================================
# Load Features
# ===========================================================================
def load_features():
    print("Loading features...")
    X_train = joblib.load(PROCESSED_DIR + 'X_train.pkl')
    X_val   = joblib.load(PROCESSED_DIR + 'X_val.pkl')
    X_test  = joblib.load(PROCESSED_DIR + 'X_test.pkl')
    y_train = joblib.load(PROCESSED_DIR + 'y_train.pkl')
    y_val   = joblib.load(PROCESSED_DIR + 'y_val.pkl')
    y_test  = joblib.load(PROCESSED_DIR + 'y_test.pkl')

    # FIX: Load binary CSVs that now have separate article/question/option columns
    # (produced by the corrected preprocessing.py)
    train_df = pd.read_csv(PROCESSED_DIR + 'train_binary.csv')
    val_df   = pd.read_csv(PROCESSED_DIR + 'val_binary.csv')
    test_df  = pd.read_csv(PROCESSED_DIR + 'test_binary.csv')

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Label balance (train) — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    return X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df

# ===========================================================================
# Handcrafted Lexical Features
#
# FIX: Now uses the REAL article / question / option columns saved by the
# corrected preprocessing.py instead of the heuristic 80/10/10 split on the
# combined text string. This makes all 5 features accurate.
#
# Features:
#   f1 — article-option word overlap (fraction of option words in article)
#   f2 — question-option word overlap (fraction of option words in question)
#   f3 — option length ratio relative to article
#   f4 — article coverage (unique article words / total words)
#   f5 — cosine similarity between (article + question) and option
# ===========================================================================
def build_lexical_features(df):
    """
    df must have columns: article, question, option
    (produced by the corrected expand_to_binary in preprocessing.py)
    """
    # Validate columns — gives a clear error if old preprocessing output is used
    required = {'article', 'question', 'option'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Binary CSV is missing columns: {missing}. "
            "Re-run preprocessing.py to regenerate with separate columns."
        )

    features = []
    for _, row in df.iterrows():
        article_words  = set(str(row['article']).split())
        question_words = set(str(row['question']).split())
        option_words   = set(str(row['option']).split())

        opt_len   = len(option_words)
        total_len = len(article_words) + len(question_words) + opt_len

        # f1: fraction of option words found in article
        f1 = len(article_words & option_words) / (opt_len + 1)

        # f2: fraction of option words found in question
        f2 = len(question_words & option_words) / (opt_len + 1)

        # f3: option length ratio relative to total text length
        f3 = opt_len / (total_len + 1)

        # f4: article lexical coverage
        f4 = len(article_words) / (total_len + 1)

        # f5: cosine similarity between (article + question) and option
        vocab_aq  = article_words | question_words
        vocab_all = vocab_aq | option_words
        v1 = np.array([1 if w in vocab_aq    else 0 for w in vocab_all], dtype=np.float32)
        v2 = np.array([1 if w in option_words else 0 for w in vocab_all], dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        f5 = float(np.dot(v1, v2) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0

        features.append([f1, f2, f3, f4, f5])

    return csr_matrix(np.array(features, dtype=np.float32))

# ===========================================================================
# Combine One-Hot + Lexical Features
# ===========================================================================
def combine_features(X_ohe, X_lex):
    return hstack([X_ohe, X_lex])

# ===========================================================================
# Evaluate — Val AND Test
#
# FIX: Evaluation now runs on both val and test sets.
# FIX: Exact Match (EM) metric added as required by spec Section 4.5.
#
# Exact Match here means: for each original question (group of 4 option rows),
# the model's top-scored option matches the gold correct option.
# ===========================================================================
def exact_match_score(model, X, y_bin, n_options=4):
    """
    For each group of 4 consecutive rows (one question), check if the model's
    highest-confidence prediction is the row labeled 1 (correct answer).

    Works with predict_proba models. For LinearSVC uses decision_function.
    """
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)

    n_questions = len(scores) // n_options
    correct = 0
    for i in range(n_questions):
        chunk_scores = scores[i * n_options:(i + 1) * n_options]
        chunk_labels = y_bin[i * n_options:(i + 1) * n_options]
        predicted_best = np.argmax(chunk_scores)
        gold_best      = np.argmax(chunk_labels)   # should be exactly one 1
        if predicted_best == gold_best:
            correct += 1
    return correct / n_questions if n_questions > 0 else 0.0

def evaluate(name, model, X, y, label="Val", compute_em=True):
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average='macro')
    prec  = precision_score(y, preds, average='macro', zero_division=0)
    rec   = recall_score(y, preds, average='macro', zero_division=0)
    cm    = confusion_matrix(y, preds)
    em    = exact_match_score(model, X, y) if compute_em else None

    print(f"\n{'='*55}")
    print(f"Model: {name}  [{label}]")
    print(f"{'='*55}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {f1:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    if em is not None:
        print(f"  Exact Match  : {em:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y, preds, zero_division=0)}")

    result = {
        'name': name, 'split': label,
        'accuracy': acc, 'f1': f1,
        'precision': prec, 'recall': rec,
        'exact_match': em,
        'confusion_matrix': cm
    }
    return result

# ===========================================================================
# Train Models
# ===========================================================================
def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        solver='saga',
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR + 'logistic_regression.pkl')
    print("Saved → models/model_a/traditional/logistic_regression.pkl")
    return model

def train_svm(X_train, y_train):
    print("\nTraining Linear SVM...")
    model = LinearSVC(
        C=1.0,
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR + 'svm.pkl')
    print("Saved → models/model_a/traditional/svm.pkl")
    return model

def train_naive_bayes(X_train, y_train):
    print("\nTraining Naive Bayes...")
    # BernoulliNB doesn't support class_weight — use sample_weight instead
    weights = np.where(y_train == 1, 3.0, 1.0)
    model = BernoulliNB(alpha=1.0)
    model.fit(X_train, y_train, sample_weight=weights)
    joblib.dump(model, MODEL_DIR + 'naive_bayes.pkl')
    print("Saved → models/model_a/traditional/naive_bayes.pkl")
    return model

def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR + 'random_forest.pkl')
    print("Saved → models/model_a/traditional/random_forest.pkl")
    return model


# ===========================================================================
# Comparison Table
# ===========================================================================
def print_comparison(results, split="Val"):
    rows = [r for r in results if r['split'] == split]
    print(f"\n{'='*75}")
    print(f"{'MODEL COMPARISON TABLE — ' + split:^75}")
    print(f"{'='*75}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} "
          f"{'Precision':>10} {'Recall':>10} {'Exact Match':>12}")
    print(f"{'-'*75}")
    for r in rows:
        em_str = f"{r['exact_match']:.4f}" if r['exact_match'] is not None else "  N/A  "
        print(f"{r['name']:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f} {em_str:>12}")
    print(f"{'='*75}")

# ===========================================================================
# Main
# ===========================================================================
def main():
    X_train, X_val, X_test, y_train, y_val, y_test, \
        train_df, val_df, test_df = load_features()

    # Build lexical features using real article/question/option columns
    print("\nBuilding lexical features...")
    X_lex_train = build_lexical_features(train_df)
    X_lex_val   = build_lexical_features(val_df)
    X_lex_test  = build_lexical_features(test_df)
    print("Lexical features done.")

    # Combine One-Hot + lexical
    X_train_combined = combine_features(X_train, X_lex_train)
    X_val_combined   = combine_features(X_val,   X_lex_val)
    X_test_combined  = combine_features(X_test,  X_lex_test)
    print(f"Combined feature shape → Train: {X_train_combined.shape}")

    results = []

    # --- Logistic Regression ---
    lr = train_logistic_regression(X_train_combined, y_train)
    results.append(evaluate("Logistic Regression", lr, X_val_combined,  y_val,  "Val"))
    results.append(evaluate("Logistic Regression", lr, X_test_combined, y_test, "Test"))

    # --- Linear SVM ---
    svm = train_svm(X_train_combined, y_train)
    results.append(evaluate("Linear SVM", svm, X_val_combined,  y_val,  "Val"))
    results.append(evaluate("Linear SVM", svm, X_test_combined, y_test, "Test"))

    # --- Naive Bayes ---
    nb = train_naive_bayes(X_train_combined, y_train)
    results.append(evaluate("Naive Bayes", nb, X_val_combined,  y_val,  "Val"))
    results.append(evaluate("Naive Bayes", nb, X_test_combined, y_test, "Test"))

    # --- Random Forest ---
    rf = train_random_forest(X_train_combined, y_train)
    results.append(evaluate("Random Forest", rf, X_val_combined,  y_val,  "Val"))
    results.append(evaluate("Random Forest", rf, X_test_combined, y_test, "Test"))

    # Print comparison tables
    print_comparison(results, split="Val")
    print_comparison(results, split="Test")

    
    # Save combined feature matrices for ensemble use
    joblib.dump(X_train_combined, PROCESSED_DIR + 'X_train_combined.pkl')
    joblib.dump(X_val_combined,   PROCESSED_DIR + 'X_val_combined.pkl')
    joblib.dump(X_test_combined,  PROCESSED_DIR + 'X_test_combined.pkl')

    # Save results (exclude confusion matrix for CSV)
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'confusion_matrix'}
        for r in results
    ])
    summary_df.to_csv(PROCESSED_DIR + 'model_a_results.csv', index=False)
    print("\nResults saved → data/processed/model_a_results.csv")
    print("Next step: run model_a_unsupervised.py, then model_a_ensemble.py")

if __name__ == '__main__':
    main()