import joblib
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

#Paths
PROCESSED_DIR = 'data/processed/'
MODEL_DIR     = 'models/model_a/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

#Load Features & Models
def load_everything():
    print("Loading features...")
    X_train = joblib.load(PROCESSED_DIR + 'X_train_combined.pkl')
    X_val   = joblib.load(PROCESSED_DIR + 'X_val_combined.pkl')
    y_train = joblib.load(PROCESSED_DIR + 'y_train.pkl')
    y_val   = joblib.load(PROCESSED_DIR + 'y_val.pkl')

    print("Loading trained models...")
    lr  = joblib.load(MODEL_DIR + 'logistic_regression.pkl')
    svm = joblib.load(MODEL_DIR + 'svm.pkl')
    nb  = joblib.load(MODEL_DIR + 'naive_bayes.pkl')
    rf  = joblib.load(MODEL_DIR + 'random_forest.pkl')

    print("All loaded.")
    return X_train, X_val, y_train, y_val, lr, svm, nb, rf

#Evaluate Helper
def evaluate(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    return {'name': name, 'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec, 'confusion_matrix': cm}

# Get Probability Estimates
# SVM doesn't output probabilities natively so we use decision_function
# and normalize it to [0,1] range to simulate soft probabilities

def get_probabilities(model, X, model_name):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)          # shape (n, 2)
    else:
        # LinearSVC — use decision function and convert to pseudo-probabilities
        scores = model.decision_function(X)    # shape (n,)
        # Normalize to [0, 1] using sigmoid
        pos_prob = 1 / (1 + np.exp(-scores))
        neg_prob = 1 - pos_prob
        return np.column_stack([neg_prob, pos_prob])

#Hard Voting
# Each model casts a vote (0 or 1), majority wins

def hard_voting(models, X, model_names):
    print("\nRunning Hard Voting...")
    votes = np.stack([m.predict(X) for m in models], axis=1)  # (n, 3)
    # Majority vote — if sum >= 2 out of 3, predict 1
    final_preds = (votes.sum(axis=1) >= 2).astype(int)
    return final_preds

#Soft Voting 
# Average probability outputs across all models, pick argmax

def soft_voting(models, X, model_names):
    print("\nRunning Soft Voting...")
    probs = np.stack(
        [get_probabilities(m, X, n) for m, n in zip(models, model_names)],
        axis=0
    )  # shape (n_models, n_samples, 2)
    avg_probs   = probs.mean(axis=0)          # (n_samples, 2)
    final_preds = avg_probs.argmax(axis=1)    # (n_samples,)
    return final_preds

#Stacking 
# Train a meta-classifier (Logistic Regression) on the outputs of base models
# We use val predictions as meta-features to avoid overfitting

def stacking(models, model_names, X_train, y_train, X_val, y_val):
    print("\nBuilding Stacking Ensemble...")

    # We must use out-of-fold predictions for the train set to avoid data leakage
    print("  Subsampling training data to 30,000 rows to speed up cross-validation...")
    
    # Subsample X_train to make CV fast (meta-classifier only needs a small subset to learn)
    subsample_size = min(30000, X_train.shape[0])
    idx = np.random.choice(X_train.shape[0], subsample_size, replace=False)
    X_train_sub = X_train[idx]
    y_train_sub = y_train[idx]

    print("  Generating train meta-features via cross-validation...")
    
    train_meta_list = []
    train_probs_list = []
    
    for m, name in zip(models, model_names):
        print(f"    -> Cross-validating {name}...")
        cloned_m = clone(m)
        
        # Get out-of-fold hard predictions
        oof_preds = cross_val_predict(cloned_m, X_train_sub, y_train_sub, cv=3, n_jobs=-1)
        train_meta_list.append(oof_preds)
        
        # Get out-of-fold probability estimates
        if hasattr(cloned_m, 'predict_proba'):
            oof_probs = cross_val_predict(cloned_m, X_train_sub, y_train_sub, cv=3, method='predict_proba', n_jobs=-1)[:, 1]
        else:
            oof_scores = cross_val_predict(cloned_m, X_train_sub, y_train_sub, cv=3, method='decision_function', n_jobs=-1)
            oof_probs = 1 / (1 + np.exp(-oof_scores))
        
        train_probs_list.append(oof_probs)

    train_meta = np.column_stack(train_meta_list)
    train_probs = np.column_stack(train_probs_list)
    train_meta_full = np.hstack([train_meta, train_probs])

    print("  Generating val meta-features (using fully trained models)...")
    val_meta = np.column_stack([m.predict(X_val) for m in models])
    val_probs = np.hstack(
        [get_probabilities(m, X_val, n)[:, 1].reshape(-1, 1)
         for m, n in zip(models, model_names)]
    )
    val_meta_full = np.hstack([val_meta, val_probs])
    
    print(f"  Meta-feature shape: {train_meta_full.shape}")

    # Train meta-classifier
    print("  Training meta-classifier (Logistic Regression)...")
    meta_clf = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        random_state=42
    )
    meta_clf.fit(train_meta_full, y_train_sub)

    val_preds = meta_clf.predict(val_meta_full)

    joblib.dump(meta_clf, MODEL_DIR + 'stacking_meta_clf.pkl')
    print("  Saved → models/model_a/traditional/stacking_meta_clf.pkl")

    return val_preds

#Final Comparison Table
def print_final_comparison(results):
    # Load individual model results for reference
    try:
        individual = pd.read_csv(PROCESSED_DIR + 'model_a_results.csv')
        ref_rows = individual[['name', 'accuracy', 'f1']].to_dict('records')
    except Exception:
        ref_rows = []

    print(f"\n{'='*70}")
    print(f"{'ENSEMBLE vs INDIVIDUAL MODEL COMPARISON':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10} {'Type':>12}")
    print(f"{'-'*70}")

    for r in ref_rows:
        print(f"{r['name']:<35} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {'Individual':>12}")

    print(f"{'-'*70}")
    for r in results:
        print(f"{r['name']:<35} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {'Ensemble':>12}")

    print(f"{'='*70}")

#Main
def main():
    X_train, X_val, y_train, y_val, lr, svm, nb, rf = load_everything()

    models       = [lr, svm, nb, rf]
    model_names  = ['Logistic Regression', 'Linear SVM', 'Naive Bayes', 'Random Forest']

    results = []

    # Hard Voting
    hard_preds = hard_voting(models, X_val, model_names)
    results.append(evaluate("Hard Voting Ensemble", y_val, hard_preds))

    # Soft Voting
    soft_preds = soft_voting(models, X_val, model_names)
    results.append(evaluate("Soft Voting Ensemble", y_val, soft_preds))

    # Stacking
    stack_preds = stacking(models, model_names, X_train, y_train, X_val, y_val)
    results.append(evaluate("Stacking Ensemble", y_val, stack_preds))

    # Final comparison
    print_final_comparison(results)

    # Save results
    summary = [{k: v for k, v in r.items() if k != 'confusion_matrix'}
               for r in results]
    pd.DataFrame(summary).to_csv(
        PROCESSED_DIR + 'model_a_ensemble_results.csv', index=False)
    print("\nResults saved → data/processed/model_a_ensemble_results.csv")
    print("\nEnsemble training complete.")

if __name__ == '__main__':
    main()