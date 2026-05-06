import joblib
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report,
                             silhouette_score)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

#Paths 
PROCESSED_DIR = 'data/processed/'
MODEL_DIR     = 'models/model_a/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Features 
def load_features():
    print("Loading features...")
    X_train = joblib.load(PROCESSED_DIR + 'X_train_combined.pkl')
    X_val   = joblib.load(PROCESSED_DIR + 'X_val_combined.pkl')
    y_train = joblib.load(PROCESSED_DIR + 'y_train.pkl')
    y_val   = joblib.load(PROCESSED_DIR + 'y_val.pkl')
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val

#  Dimensionality Reduction 
# 15004 features is too large for clustering and label propagation
# TruncatedSVD (LSA) reduces to 100 dense dimensions while preserving structure
# This is standard practice for sparse text features before clustering

def reduce_dimensions(X_train, X_val, n_components=100):
    print(f"\nReducing dimensions to {n_components} with TruncatedSVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train)
    X_val_reduced   = svd.transform(X_val)

    # Normalize after SVD — important for cosine-based clustering
    X_train_reduced = normalize(X_train_reduced)
    X_val_reduced   = normalize(X_val_reduced)

    explained = svd.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained:.4f}")
    print(f"Reduced shapes → Train: {X_train_reduced.shape}, Val: {X_val_reduced.shape}")

    joblib.dump(svd, MODEL_DIR + 'svd_reducer.pkl')
    return X_train_reduced, X_val_reduced, svd

#  K-Means Clustering 
# We use k=2 since we have a binary task (correct vs incorrect answer)
# After clustering we assign labels by majority vote within each cluster

def clustering_purity(y_true, cluster_labels):
    """Purity: for each cluster, fraction of majority class."""
    clusters = np.unique(cluster_labels)
    total = len(y_true)
    purity = 0
    for c in clusters:
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        majority = np.bincount(y_true[mask]).max()
        purity += majority
    return purity / total

def assign_cluster_labels(y_train, cluster_labels_train):
    """Map each cluster id to the majority true label in that cluster."""
    cluster_to_label = {}
    for c in np.unique(cluster_labels_train):
        mask = cluster_labels_train == c
        majority = np.bincount(y_train[mask]).argmax()
        cluster_to_label[c] = majority
    return cluster_to_label

def run_kmeans(X_train_reduced, X_val_reduced, y_train, y_val, sample_size=50000):
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING (k=2)")
    print("="*50)

    # Use a sample for silhouette score — computing on 351k rows is too slow
    print(f"Training KMeans on full train set...")

    # MiniBatchKMeans is faster on large datasets
    kmeans = MiniBatchKMeans(
        n_clusters=2,
        random_state=42,
        batch_size=10000,
        n_init=10
    )
    kmeans.fit(X_train_reduced)

    # Get cluster assignments
    train_clusters = kmeans.labels_
    val_clusters   = kmeans.predict(X_val_reduced)

    # Purity
    train_purity = clustering_purity(y_train, train_clusters)
    val_purity   = clustering_purity(y_val,   val_clusters)
    print(f"  Cluster Purity (Train): {train_purity:.4f}")
    print(f"  Cluster Purity (Val)  : {val_purity:.4f}")

    # Silhouette score on a sample (too expensive on full set)
    print(f"  Computing silhouette score on {sample_size} samples...")
    idx = np.random.choice(len(X_train_reduced), sample_size, replace=False)
    sil = silhouette_score(X_train_reduced[idx], train_clusters[idx], metric='cosine')
    print(f"  Silhouette Score (sample): {sil:.4f}")

    # Map clusters to labels and evaluate
    cluster_to_label = assign_cluster_labels(y_train, train_clusters)
    print(f"  Cluster mapping: {cluster_to_label}")

    val_preds = np.array([cluster_to_label[c] for c in val_clusters])

    acc  = accuracy_score(y_val, val_preds)
    f1   = f1_score(y_val, val_preds, average='macro')
    prec = precision_score(y_val, val_preds, average='macro', zero_division=0)
    rec  = recall_score(y_val, val_preds, average='macro', zero_division=0)
    cm   = confusion_matrix(y_val, val_preds)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_val, val_preds, zero_division=0)}")

    joblib.dump(kmeans, MODEL_DIR + 'kmeans.pkl')
    print("Saved → models/model_a/traditional/kmeans.pkl")

    return {
        'name': 'K-Means (k=2)',
        'accuracy': acc, 'f1': f1,
        'precision': prec, 'recall': rec,
        'purity': val_purity, 'silhouette': sil,
        'confusion_matrix': cm
    }

#  Label Propagation (Semi-Supervised) 
# We simulate semi-supervised learning:
#   - Take a small labeled subset (labeled_ratio of train)
#   - Mark the rest as unlabeled (-1)
#   - Let Label Spreading propagate labels through the feature graph

def run_label_propagation(X_train_reduced, X_val_reduced,
                          y_train, y_val,
                          labeled_ratio=0.05,
                          max_samples=20000):
    print("\n" + "="*50)
    print("LABEL SPREADING (Semi-Supervised)")
    print("="*50)

    # Work on a manageable subset — LabelSpreading is O(n²) in memory
    total = min(len(X_train_reduced), max_samples)
    idx   = np.random.choice(len(X_train_reduced), total, replace=False)
    X_sub = X_train_reduced[idx]
    y_sub = y_train[idx]

    # Mark only labeled_ratio of samples as labeled
    n_labeled = int(total * labeled_ratio)
    y_semi    = np.full(total, -1)          # -1 means unlabeled
    labeled_idx = np.random.choice(total, n_labeled, replace=False)
    y_semi[labeled_idx] = y_sub[labeled_idx]

    n_labeled_pos = (y_sub[labeled_idx] == 1).sum()
    n_labeled_neg = (y_sub[labeled_idx] == 0).sum()
    print(f"  Total samples : {total}")
    print(f"  Labeled       : {n_labeled} ({labeled_ratio*100:.0f}%) "
          f"→ pos: {n_labeled_pos}, neg: {n_labeled_neg}")
    print(f"  Unlabeled     : {total - n_labeled}")

    print("  Training LabelSpreading...")
    model = LabelSpreading(
        kernel='knn',
        n_neighbors=7,
        alpha=0.2,
        max_iter=30,
        n_jobs=-1
    )
    model.fit(X_sub, y_semi)

    # Evaluate on validation set
    val_preds = model.predict(X_val_reduced)

    acc  = accuracy_score(y_val, val_preds)
    f1   = f1_score(y_val, val_preds, average='macro')
    prec = precision_score(y_val, val_preds, average='macro', zero_division=0)
    rec  = recall_score(y_val, val_preds, average='macro', zero_division=0)
    cm   = confusion_matrix(y_val, val_preds)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_val, val_preds, zero_division=0)}")

    joblib.dump(model, MODEL_DIR + 'label_spreading.pkl')
    print("Saved → models/model_a/traditional/label_spreading.pkl")

    return {
        'name': f'Label Spreading ({labeled_ratio*100:.0f}% labeled)',
        'accuracy': acc, 'f1': f1,
        'precision': prec, 'recall': rec,
        'confusion_matrix': cm
    }

#  Comparison Table 
def print_comparison(unsupervised_results, supervised_f1_reference=0.4965):
    print(f"\n{'='*70}")
    print(f"{'UNSUPERVISED vs SUPERVISED COMPARISON':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10} {'vs Supervised':>12}")
    print(f"{'-'*70}")

    for r in unsupervised_results:
        diff = r['f1'] - supervised_f1_reference
        sign = '+' if diff >= 0 else ''
        print(f"{r['name']:<35} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{sign}{diff:>11.4f}")

    print(f"{'Supervised (SVM baseline)':<35} {'0.5323':>10} {supervised_f1_reference:>10.4f} "
          f"{'(reference)':>12}")
    print(f"{'='*70}")


def main():
    X_train, X_val, y_train, y_val = load_features()

    # Reduce dimensions first — required for both clustering and label prop
    X_train_r, X_val_r, svd = reduce_dimensions(X_train, X_val, n_components=100)

    results = []

    # K-Means
    km_result = run_kmeans(X_train_r, X_val_r, y_train, y_val)
    results.append(km_result)

    # Label Spreading
    lp_result = run_label_propagation(X_train_r, X_val_r, y_train, y_val,
                                      labeled_ratio=0.05,
                                      max_samples=20000)
    results.append(lp_result)

    # Comparison table
    print_comparison(results)

    # Save summary
    summary = [{k: v for k, v in r.items() if k != 'confusion_matrix'}
               for r in results]
    pd.DataFrame(summary).to_csv(
        PROCESSED_DIR + 'model_a_unsupervised_results.csv', index=False)
    print("\nResults saved → data/processed/model_a_unsupervised_results.csv")
    print("\nUnsupervised/Semi-Supervised training complete.")

if __name__ == '__main__':
    main()