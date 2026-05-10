import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Ensure the root and src directories are in the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_a_train import exact_match_score
from src.model_a_ensemble import hard_voting, soft_voting, get_probabilities

PROCESSED_DIR = 'data/processed/'
MODEL_DIR = 'models/model_a/traditional/'

def evaluate_model_a(name, model, X, y, y_bin=None, compute_em=False):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro', zero_division=0)
    prec = precision_score(y, preds, average='macro', zero_division=0)
    rec = recall_score(y, preds, average='macro', zero_division=0)
    cm = confusion_matrix(y, preds)
    em = exact_match_score(model, X, y_bin) if compute_em and y_bin is not None else None
    
    return {
        'Category': 'Model A (Answer Verification)',
        'Model': name,
        'Split': 'Test',
        'Accuracy': acc,
        'Macro F1': f1,
        'Precision': prec,
        'Recall': rec,
        'Exact Match': em,
        'Confusion Matrix': str(cm.tolist())
    }

def main():
    print("="*100)
    print(f"{'FULL PROJECT EVALUATION SCRIPT':^100}")
    print("="*100)
    
    # 1. Load test data
    print("\n[1/4] Loading test data and models...")
    X_test = joblib.load(PROCESSED_DIR + 'X_test_combined.pkl')
    y_test = joblib.load(PROCESSED_DIR + 'y_test.pkl')
    
    # 2. Load Models
    lr = joblib.load(MODEL_DIR + 'logistic_regression.pkl')
    svm = joblib.load(MODEL_DIR + 'svm.pkl')
    nb = joblib.load(MODEL_DIR + 'naive_bayes.pkl')
    rf = joblib.load(MODEL_DIR + 'random_forest.pkl')
    meta_clf = joblib.load(MODEL_DIR + 'stacking_meta_clf.pkl')
    
    print("Models loaded successfully.")
    
    results = []
    
    # 3. Model A Individual Classifiers Evaluation
    print("\n[2/4] Evaluating Model A Base Classifiers on Test Set...")
    results.append(evaluate_model_a('Logistic Regression', lr, X_test, y_test, y_test, True))
    results.append(evaluate_model_a('Linear SVM', svm, X_test, y_test, y_test, True))
    results.append(evaluate_model_a('Naive Bayes', nb, X_test, y_test, y_test, True))
    results.append(evaluate_model_a('Random Forest', rf, X_test, y_test, y_test, True))
    
    # 4. Model A Ensembles Evaluation
    print("Evaluating Model A Ensembles on Test Set...")
    models = [lr, svm, rf]
    model_names = ['Logistic Regression', 'Linear SVM', 'Random Forest']
    
    # Hard Voting
    hv_preds = hard_voting(models, X_test, model_names)
    results.append({
        'Category': 'Model A (Ensemble)',
        'Model': 'Hard Voting',
        'Split': 'Test',
        'Accuracy': accuracy_score(y_test, hv_preds),
        'Macro F1': f1_score(y_test, hv_preds, average='macro', zero_division=0),
        'Precision': precision_score(y_test, hv_preds, average='macro', zero_division=0),
        'Recall': recall_score(y_test, hv_preds, average='macro', zero_division=0),
        'Exact Match': None,
        'Confusion Matrix': str(confusion_matrix(y_test, hv_preds).tolist())
    })
    
    # Soft Voting
    sv_preds = soft_voting(models, X_test, model_names)
    results.append({
        'Category': 'Model A (Ensemble)',
        'Model': 'Soft Voting',
        'Split': 'Test',
        'Accuracy': accuracy_score(y_test, sv_preds),
        'Macro F1': f1_score(y_test, sv_preds, average='macro', zero_division=0),
        'Precision': precision_score(y_test, sv_preds, average='macro', zero_division=0),
        'Recall': recall_score(y_test, sv_preds, average='macro', zero_division=0),
        'Exact Match': None,
        'Confusion Matrix': str(confusion_matrix(y_test, sv_preds).tolist())
    })
    
    # Stacking
    test_meta = np.column_stack([m.predict(X_test) for m in models])
    test_probs = np.hstack([get_probabilities(m, X_test, n)[:, 1].reshape(-1, 1) for m, n in zip(models, model_names)])
    test_meta_full = np.hstack([test_meta, test_probs])
    stack_preds = meta_clf.predict(test_meta_full)
    results.append({
        'Category': 'Model A (Ensemble)',
        'Model': 'Stacking',
        'Split': 'Test',
        'Accuracy': accuracy_score(y_test, stack_preds),
        'Macro F1': f1_score(y_test, stack_preds, average='macro', zero_division=0),
        'Precision': precision_score(y_test, stack_preds, average='macro', zero_division=0),
        'Recall': recall_score(y_test, stack_preds, average='macro', zero_division=0),
        'Exact Match': None,
        'Confusion Matrix': str(confusion_matrix(y_test, stack_preds).tolist())
    })

    # 5. Extract Model B Metrics from model_b_results.csv
    print("\n[3/4] Extracting Model B Metrics from saved CSVs...")
    try:
        mb_df = pd.read_csv(PROCESSED_DIR + 'model_b_results.csv')
        mb_test = mb_df[mb_df['split'] == 'Test']
        if not mb_test.empty:
            ranker = mb_test[mb_test['type'] == 'ranker']
            gen = mb_test[mb_test['type'] == 'distractor_generation']
            hint = mb_test[mb_test['type'] == 'hint_generation']
            
            if not gen.empty:
                results.append({
                    'Category': 'Model B (Distractor & Hint)',
                    'Model': 'Distractor Generation',
                    'Split': 'Test',
                    'Accuracy': None,
                    'Macro F1': float(gen['f1'].iloc[0]) if 'f1' in gen.columns else None,
                    'Precision': float(gen['precision'].iloc[0]) if 'precision' in gen.columns else None,
                    'Recall': float(gen['recall'].iloc[0]) if 'recall' in gen.columns else None,
                    'Exact Match': None,
                    'Confusion Matrix': None
                })
                
            if not ranker.empty:
                results.append({
                    'Category': 'Model B (Distractor & Hint)',
                    'Model': 'Distractor Ranker',
                    'Split': 'Test',
                    'Accuracy': float(ranker['accuracy'].iloc[0]) if 'accuracy' in ranker.columns else None,
                    'Macro F1': float(ranker['f1'].iloc[0]) if 'f1' in ranker.columns else None,
                    'Precision': float(ranker['precision'].iloc[0]) if 'precision' in ranker.columns else None,
                    'Recall': float(ranker['recall'].iloc[0]) if 'recall' in ranker.columns else None,
                    'Exact Match': None,
                    'Confusion Matrix': None
                })
                
            if not hint.empty:
                results.append({
                    'Category': 'Model B (Distractor & Hint)',
                    'Model': 'Hint Generation',
                    'Split': 'Test',
                    'Accuracy': None,
                    'Macro F1': None,
                    'Precision': float(hint['hint_precision'].iloc[0]) if 'hint_precision' in hint.columns else None,
                    'Recall': None,
                    'Exact Match': None,
                    'Confusion Matrix': None
                })
    except Exception as e:
        print(f"Warning: Could not fully extract Model B metrics. {e}")

    # 6. Load ALL CSV files for full reporting
    print("\n[4/4] Loading and displaying all historical CSV results...")
    csv_files = [
        'model_a_results.csv', 
        'model_a_ensemble_results.csv', 
        'model_a_unsupervised_results.csv', 
        'model_b_results.csv'
    ]
    for file in csv_files:
        path = PROCESSED_DIR + file
        if os.path.exists(path):
            print(f"\n--- {file} ---")
            df = pd.read_csv(path)
            # Truncate string columns to prevent massive console output
            with pd.option_context('display.max_columns', None, 'display.width', 150):
                print(df.to_string(index=False))
        else:
            print(f"\n--- {file} (Not Found) ---")

    # Print Combined Summary Table
    final_df = pd.DataFrame(results)
    print("\n" + "="*120)
    print(f"{'FINAL COMBINED SUMMARY TABLE (TEST SET)':^120}")
    print("="*120)
    
    # Format for display
    display_df = final_df.copy()
    display_df.drop(columns=['Confusion Matrix'], inplace=True, errors='ignore')
    # Use rounded formatting for readability
    for col in ['Accuracy', 'Macro F1', 'Precision', 'Recall', 'Exact Match']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
    print(display_df.to_string(index=False))
    print("="*120)
    
    # Save to CSV
    export_path = PROCESSED_DIR + 'full_evaluation_report.csv'
    final_df.to_csv(export_path, index=False)
    print(f"\nSuccessfully exported full evaluation report to: {export_path}")

if __name__ == '__main__':
    main()
