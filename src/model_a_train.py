import joblib
import numpy as np
import pandas as pd
import os
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier

# Paths 
PROCESSED_DIR = 'data/processed/'
MODEL_DIR     = 'models/model_a/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

#  Load Features & Raw Binary CSVs 
def load_features():
    print("Loading features...")
    X_train = joblib.load(PROCESSED_DIR + 'X_train.pkl')
    X_val   = joblib.load(PROCESSED_DIR + 'X_val.pkl')
    y_train = joblib.load(PROCESSED_DIR + 'y_train.pkl')
    y_val   = joblib.load(PROCESSED_DIR + 'y_val.pkl')

    train_df = pd.read_csv(PROCESSED_DIR + 'train_binary.csv')
    val_df   = pd.read_csv(PROCESSED_DIR + 'val_binary.csv')

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"Label balance (train) — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    return X_train, X_val, y_train, y_val, train_df, val_df

# Handcrafted Lexical Features 
# These give the model meaningful signal beyond raw word presence:
#   - keyword_overlap : how many words from the option appear in the article
#   - option_length   : length of the answer option (normalized)
#   - question_overlap: how many question words appear in the option

def keyword_overlap(text1, text2):
    """Fraction of words in text2 that appear in text1."""
    words1 = set(str(text1).split())
    words2 = set(str(text2).split())
    if not words2:
        return 0.0
    return len(words1 & words2) / len(words2)

def build_lexical_features(df):
    """
    df must have columns: text (article+question+option combined)
    We re-split it back since that's what we have saved.
    For simplicity we derive features from the combined text.
    """
    features = []
    for _, row in df.iterrows():
        parts = str(row['text']).split()
        total = len(parts)

        # Rough heuristic splits from combined text
        # article ~ first 80%, question ~ next 10%, option ~ last 10%
        article_end  = int(total * 0.80)
        question_end = int(total * 0.90)

        article_words  = set(parts[:article_end])
        question_words = set(parts[article_end:question_end])
        option_words   = set(parts[question_end:])

        opt_len = len(option_words)

        f1 = len(article_words  & option_words)  / (opt_len + 1)  # article-option overlap
        f2 = len(question_words & option_words)  / (opt_len + 1)  # question-option overlap
        f3 = opt_len / (total + 1)                                  # option length ratio
        f4 = len(article_words) / (total + 1)                       # article coverage

        # Cosine similarity between (article + question) and (option)
        vocab_aq = article_words | question_words
        vocab_all = vocab_aq | option_words
        v1 = np.array([1 if w in vocab_aq else 0 for w in vocab_all])
        v2 = np.array([1 if w in option_words else 0 for w in vocab_all])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        f5 = np.dot(v1, v2) / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

        features.append([f1, f2, f3, f4, f5])

    return csr_matrix(np.array(features, dtype=np.float32))

# Combine One-Hot  Lexical Features 
def combine_features(X_ohe, X_lex):
    return hstack([X_ohe, X_lex])

# Evaluate 
def evaluate(name, model, X_val, y_val):
    preds = model.predict(X_val)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds, average='macro')
    prec  = precision_score(y_val, preds, average='macro', zero_division=0)
    rec   = recall_score(y_val, preds, average='macro', zero_division=0)
    cm    = confusion_matrix(y_val, preds)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_val, preds, zero_division=0)}")

    return {'name': name, 'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec, 'confusion_matrix': cm}

# Train Models 
def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        solver='saga',
        C=1.0,
        class_weight='balanced',   # fixes imbalance
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
        class_weight='balanced',   # fixes imbalance
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR + 'svm.pkl')
    print("Saved → models/model_a/traditional/svm.pkl")
    return model

def train_naive_bayes(X_train, y_train):
    print("\nTraining Naive Bayes...")
    # BernoulliNB doesn't support class_weight
    # so we use sample_weight to replicate balanced weighting
    weights = np.where(y_train == 1, 3.0, 1.0)   # upweight positives 3x
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

# Comparison Table 
def print_comparison(results):
    print(f"\n{'='*60}")
    print(f"{'MODEL COMPARISON TABLE':^60}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['name']:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f}")
    print(f"{'='*60}")

# Main 
def main():
    X_train, X_val, y_train, y_val, train_df, val_df = load_features()

    # Build lexical features
    print("\nBuilding lexical features (this may take a few minutes)...")
    X_lex_train = build_lexical_features(train_df)
    X_lex_val   = build_lexical_features(val_df)
    print("Lexical features done.")

    # Combine with One-Hot features
    X_train_combined = combine_features(X_train, X_lex_train)
    X_val_combined   = combine_features(X_val,   X_lex_val)
    print(f"Combined feature shape → Train: {X_train_combined.shape}")

    results = []

    lr  = train_logistic_regression(X_train_combined, y_train)
    results.append(evaluate("Logistic Regression", lr, X_val_combined, y_val))

    svm = train_svm(X_train_combined, y_train)
    results.append(evaluate("Linear SVM", svm, X_val_combined, y_val))

    nb  = train_naive_bayes(X_train_combined, y_train)
    results.append(evaluate("Naive Bayes", nb, X_val_combined, y_val))

    rf = train_random_forest(X_train_combined, y_train)
    results.append(evaluate("Random Forest", rf, X_val_combined, y_val))

    print_comparison(results)

    # Save combined feature matrices for Model B and ensemble use
    joblib.dump(X_train_combined, PROCESSED_DIR + 'X_train_combined.pkl')
    joblib.dump(X_val_combined,   PROCESSED_DIR + 'X_val_combined.pkl')

    summary_df = pd.DataFrame([{k: v for k, v in r.items()
                                 if k != 'confusion_matrix'} for r in results])
    summary_df.to_csv(PROCESSED_DIR + 'model_a_results.csv', index=False)
    print("\nResults saved → data/processed/model_a_results.csv")

if __name__ == '__main__':
    main()