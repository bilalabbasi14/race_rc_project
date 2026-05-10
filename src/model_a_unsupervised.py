import joblib
import numpy as np
import pandas as pd
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix, classification_report,
                             silhouette_score)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ===========================================================================
# Paths
# ===========================================================================
PROCESSED_DIR = 'data/processed/'
MODEL_DIR     = 'models/model_a/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================================================================
# Load Features
#
# FIX: Now loads BOTH binary labels (y_train/y_val/y_test — 0/1 correctness)
# AND multiclass labels (y_train_multiclass etc — A/B/C/D → 0/1/2/3).
#
# Binary labels     → used for Label Spreading (semi-supervised verification)
# Multiclass labels → used for K-Means and GMM clustering, since the spec
#                     says clustering should discover latent ANSWER PATTERNS
#                     across the 4 options, which maps naturally to k=4.
# FIX: Also loads X_test and y_test for final test-set evaluation.
# ===========================================================================
def load_features():
    print("Loading features...")
    X_train = joblib.load(PROCESSED_DIR + 'X_train_combined.pkl')
    X_val   = joblib.load(PROCESSED_DIR + 'X_val_combined.pkl')
    X_test  = joblib.load(PROCESSED_DIR + 'X_test_combined.pkl')

    y_train = joblib.load(PROCESSED_DIR + 'y_train.pkl')
    y_val   = joblib.load(PROCESSED_DIR + 'y_val.pkl')
    y_test  = joblib.load(PROCESSED_DIR + 'y_test.pkl')

    # FIX: Multiclass labels for clustering (one per question, not per option row)
    y_train_mc = joblib.load(PROCESSED_DIR + 'y_train_multiclass.pkl')
    y_val_mc   = joblib.load(PROCESSED_DIR + 'y_val_multiclass.pkl')
    y_test_mc  = joblib.load(PROCESSED_DIR + 'y_test_multiclass.pkl')

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Binary label balance (train) — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    print(f"Multiclass label balance (train) — "
          f"A: {(y_train_mc==0).sum()} B: {(y_train_mc==1).sum()} "
          f"C: {(y_train_mc==2).sum()} D: {(y_train_mc==3).sum()}")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            y_train_mc, y_val_mc, y_test_mc)

# ===========================================================================
# Load Supervised Baseline Results Dynamically
#
# FIX: Reference F1 and accuracy are loaded from model_a_results.csv instead
# of being hardcoded. Falls back to None if the file doesn't exist yet.
# ===========================================================================
def load_supervised_reference():
    path = PROCESSED_DIR + 'model_a_results.csv'
    if not os.path.exists(path):
        print("  WARNING: model_a_results.csv not found. "
              "Run model_a_train.py first for a valid comparison.")
        return None, None

    df = pd.read_csv(path)
    # Use the best val-split supervised model as reference
    val_rows = df[df['split'] == 'Val'] if 'split' in df.columns else df
    if val_rows.empty:
        val_rows = df

    best_row = val_rows.loc[val_rows['f1'].idxmax()]
    ref_name = best_row['name']
    ref_f1   = float(best_row['f1'])
    ref_acc  = float(best_row['accuracy'])
    print(f"  Supervised reference → {ref_name}: "
          f"Acc={ref_acc:.4f}, F1={ref_f1:.4f}")
    return ref_acc, ref_f1

# ===========================================================================
# Dimensionality Reduction
#
# TruncatedSVD (LSA) reduces sparse high-dimensional One-Hot features to 100
# dense dimensions while preserving structure. Standard practice before
# clustering on text features. L2 normalisation after SVD ensures cosine-
# based distance metrics work correctly.
#
# FIX: Now also reduces X_test for test-set evaluation.
# ===========================================================================
def reduce_dimensions(X_train, X_val, X_test, n_components=100):
    print(f"\nReducing dimensions to {n_components} with TruncatedSVD (LSA)...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    X_train_r = normalize(svd.fit_transform(X_train))
    X_val_r   = normalize(svd.transform(X_val))
    X_test_r  = normalize(svd.transform(X_test))

    explained = svd.explained_variance_ratio_.sum()
    print(f"  Explained variance : {explained:.4f}")
    print(f"  Reduced shapes → Train: {X_train_r.shape}, "
          f"Val: {X_val_r.shape}, Test: {X_test_r.shape}")

    joblib.dump(svd, MODEL_DIR + 'svd_reducer.pkl')
    print("  Saved → models/model_a/traditional/svd_reducer.pkl")
    return X_train_r, X_val_r, X_test_r, svd

# ===========================================================================
# Clustering Utilities
# ===========================================================================
def clustering_purity(y_true, cluster_labels):
    """Purity: weighted fraction of majority class across all clusters."""
    total  = len(y_true)
    purity = 0
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        majority = np.bincount(y_true[mask]).max()
        purity  += majority
    return purity / total

def assign_cluster_labels(y_true, cluster_labels):
    """Map each cluster id → majority true label in that cluster."""
    mapping = {}
    for c in np.unique(cluster_labels):
        mask     = cluster_labels == c
        majority = np.bincount(y_true[mask]).argmax()
        mapping[c] = int(majority)
    return mapping

def evaluate_cluster_predictions(name, y_true, cluster_labels,
                                 cluster_to_label, split="Val"):
    """Convert cluster assignments to label predictions and evaluate."""
    preds = np.array([cluster_to_label[c] for c in cluster_labels])
    acc   = accuracy_score(y_true, preds)
    f1    = f1_score(y_true, preds, average='macro')
    prec  = precision_score(y_true, preds, average='macro', zero_division=0)
    rec   = recall_score(y_true, preds, average='macro', zero_division=0)
    cm    = confusion_matrix(y_true, preds)

    print(f"\n  [{split}] Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | "
          f"Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_true, preds, zero_division=0)}")

    return {'name': name, 'split': split,
            'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec,
            'confusion_matrix': cm}

# ===========================================================================
# K-Means Clustering
#
# FIX: Uses k=4 with MULTICLASS labels (A/B/C/D → 0/1/2/3) to discover
# latent answer-position patterns across the 4 options, as described in
# the spec ("group question-answer pairs by feature similarity to discover
# latent answer patterns without labels").
#
# Also runs k=2 with binary labels for direct comparison with supervised
# answer verification models.
#
# FIX: Evaluates on val AND test sets.
# ===========================================================================
def run_kmeans(X_train_r, X_val_r, X_test_r,
               y_train_bin, y_val_bin, y_test_bin,
               y_train_mc,  y_val_mc,  y_test_mc,
               sample_size=50000):
    print("\n" + "="*55)
    print("K-MEANS CLUSTERING")
    print("="*55)

    results = []

    for k, y_tr, y_v, y_te, label_type in [
        (2, y_train_bin, y_val_bin, y_test_bin, "binary (k=2)"),
        (4, y_train_mc,  y_val_mc,  y_test_mc,  "multiclass (k=4)"),
    ]:
        print(f"\n--- K-Means {label_type} ---")

        # FIX: y_tr may have different length than X_train_r when using
        # multiclass labels (one per question vs one per option row).
        # We subsample X_train_r to match the multiclass label count.
        if len(y_tr) != len(X_train_r):
            # Multiclass: take every 4th row (the correct-option row per question)
            step = len(X_train_r) // len(y_tr)
            X_tr_use = X_train_r[::step][:len(y_tr)]
            X_v_use  = X_val_r[::step][:len(y_v)]
            X_te_use = X_test_r[::step][:len(y_te)]
        else:
            X_tr_use, X_v_use, X_te_use = X_train_r, X_val_r, X_test_r

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=10000,
            n_init=10
        )
        kmeans.fit(X_tr_use)

        train_clusters = kmeans.labels_
        val_clusters   = kmeans.predict(X_v_use)
        test_clusters  = kmeans.predict(X_te_use)

        # Purity
        train_purity = clustering_purity(y_tr, train_clusters)
        val_purity   = clustering_purity(y_v,  val_clusters)
        print(f"  Cluster Purity → Train: {train_purity:.4f} | Val: {val_purity:.4f}")

        # Silhouette on a sample
        sil_idx = np.random.choice(len(X_tr_use), min(sample_size, len(X_tr_use)),
                                   replace=False)
        sil = silhouette_score(X_tr_use[sil_idx], train_clusters[sil_idx],
                               metric='cosine')
        print(f"  Silhouette Score (sample n={len(sil_idx)}): {sil:.4f}")

        # Map clusters to labels
        cluster_to_label = assign_cluster_labels(y_tr, train_clusters)
        print(f"  Cluster → label mapping: {cluster_to_label}")

        name = f"K-Means {label_type}"
        r_val  = evaluate_cluster_predictions(name, y_v,  val_clusters,
                                              cluster_to_label, "Val")
        r_test = evaluate_cluster_predictions(name, y_te, test_clusters,
                                              cluster_to_label, "Test")

        r_val['purity']     = val_purity
        r_val['silhouette'] = sil
        r_test['purity']    = clustering_purity(y_te, test_clusters)
        r_test['silhouette'] = sil

        results.extend([r_val, r_test])

        joblib.dump(kmeans, MODEL_DIR + f'kmeans_k{k}.pkl')
        print(f"  Saved → models/model_a/traditional/kmeans_k{k}.pkl")

    return results

# ===========================================================================
# Gaussian Mixture Models (GMM)
#
# FIX: Added — GMM is explicitly listed in the spec (Section 4.2.2).
# GMM assigns SOFT membership probabilities to clusters, making it more
# expressive than hard K-Means. Useful for identifying question type patterns
# where boundaries between answer clusters are fuzzy.
#
# Uses k=4 with multiclass labels and k=2 with binary labels.
# Evaluates on val AND test sets.
# ===========================================================================
def run_gmm(X_train_r, X_val_r, X_test_r,
            y_train_bin, y_val_bin, y_test_bin,
            y_train_mc,  y_val_mc,  y_test_mc):
    print("\n" + "="*55)
    print("GAUSSIAN MIXTURE MODEL (GMM)")
    print("="*55)

    results = []

    for k, y_tr, y_v, y_te, label_type in [
        (2, y_train_bin, y_val_bin, y_test_bin, "binary (k=2)"),
        (4, y_train_mc,  y_val_mc,  y_test_mc,  "multiclass (k=4)"),
    ]:
        print(f"\n--- GMM {label_type} ---")

        # Align feature rows with label count (same logic as K-Means)
        if len(y_tr) != len(X_train_r):
            step = len(X_train_r) // len(y_tr)
            X_tr_use = X_train_r[::step][:len(y_tr)]
            X_v_use  = X_val_r[::step][:len(y_v)]
            X_te_use = X_test_r[::step][:len(y_te)]
        else:
            X_tr_use, X_v_use, X_te_use = X_train_r, X_val_r, X_test_r

        # Subsample training data — GMM is expensive on 350k rows
        sub_size  = min(30000, len(X_tr_use))
        sub_idx   = np.random.choice(len(X_tr_use), sub_size, replace=False)
        X_tr_sub  = X_tr_use[sub_idx]
        y_tr_sub  = y_tr[sub_idx]

        print(f"  Fitting GMM on {sub_size} training samples...")
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='diag',   # diag is much faster than full for high-dim
            random_state=42,
            max_iter=100,
            n_init=3
        )
        gmm.fit(X_tr_sub)

        train_clusters = gmm.predict(X_tr_sub)
        val_clusters   = gmm.predict(X_v_use)
        test_clusters  = gmm.predict(X_te_use)

        # Purity on training subsample
        train_purity = clustering_purity(y_tr_sub, train_clusters)
        val_purity   = clustering_purity(y_v, val_clusters)
        print(f"  Cluster Purity → Train (sub): {train_purity:.4f} | Val: {val_purity:.4f}")

        # Silhouette on training subsample
        sil = silhouette_score(X_tr_sub, train_clusters, metric='cosine')
        print(f"  Silhouette Score: {sil:.4f}")

        # Map clusters to labels using training subsample
        cluster_to_label = assign_cluster_labels(y_tr_sub, train_clusters)
        print(f"  Cluster → label mapping: {cluster_to_label}")

        name   = f"GMM {label_type}"
        r_val  = evaluate_cluster_predictions(name, y_v,  val_clusters,
                                              cluster_to_label, "Val")
        r_test = evaluate_cluster_predictions(name, y_te, test_clusters,
                                              cluster_to_label, "Test")

        r_val['purity']      = val_purity
        r_val['silhouette']  = sil
        r_test['purity']     = clustering_purity(y_te, test_clusters)
        r_test['silhouette'] = sil

        results.extend([r_val, r_test])

        joblib.dump(gmm, MODEL_DIR + f'gmm_k{k}.pkl')
        print(f"  Saved → models/model_a/traditional/gmm_k{k}.pkl")

    return results

# ===========================================================================
# Label Spreading (Semi-Supervised)
#
# Simulates a realistic semi-supervised setting:
#   - Small labeled subset (labeled_ratio of train)
#   - Rest marked as unlabeled (-1)
#   - LabelSpreading propagates labels through the KNN graph
#
# FIX: Evaluates on val AND test sets.
# FIX: Uses binary labels (correct/incorrect) — appropriate for the answer
#      verification task that Label Spreading is solving here.
# ===========================================================================
def run_label_propagation(X_train_r, X_val_r, X_test_r,
                          y_train, y_val, y_test,
                          labeled_ratio=0.05,
                          max_samples=20000):
    print("\n" + "="*55)
    print("LABEL SPREADING (Semi-Supervised)")
    print("="*55)

    # Subsample — LabelSpreading is O(n²) in memory
    total   = min(len(X_train_r), max_samples)
    idx     = np.random.choice(len(X_train_r), total, replace=False)
    X_sub   = X_train_r[idx]
    y_sub   = y_train[idx]

    # Mark only labeled_ratio as labeled, rest as -1 (unlabeled)
    n_labeled   = int(total * labeled_ratio)
    y_semi      = np.full(total, -1)
    labeled_idx = np.random.choice(total, n_labeled, replace=False)
    y_semi[labeled_idx] = y_sub[labeled_idx]

    n_pos = (y_sub[labeled_idx] == 1).sum()
    n_neg = (y_sub[labeled_idx] == 0).sum()
    print(f"  Total samples  : {total}")
    print(f"  Labeled        : {n_labeled} ({labeled_ratio*100:.0f}%) "
          f"→ correct: {n_pos}, incorrect: {n_neg}")
    print(f"  Unlabeled      : {total - n_labeled}")

    print("  Training LabelSpreading (kernel=knn, n_neighbors=7)...")
    model = LabelSpreading(
        kernel='knn',
        n_neighbors=7,
        alpha=0.2,
        max_iter=30,
        n_jobs=-1
    )
    model.fit(X_sub, y_semi)

    results = []
    for X_eval, y_eval, split in [
        (X_val_r,  y_val,  "Val"),
        (X_test_r, y_test, "Test"),
    ]:
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average='macro')
        prec  = precision_score(y_eval, preds, average='macro', zero_division=0)
        rec   = recall_score(y_eval, preds, average='macro', zero_division=0)
        cm    = confusion_matrix(y_eval, preds)

        print(f"\n  [{split}] Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | "
              f"Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print(f"\n{classification_report(y_eval, preds, zero_division=0)}")

        results.append({
            'name': f'Label Spreading ({labeled_ratio*100:.0f}% labeled)',
            'split': split,
            'accuracy': acc, 'f1': f1,
            'precision': prec, 'recall': rec,
            'confusion_matrix': cm
        })

    joblib.dump(model, MODEL_DIR + 'label_spreading.pkl')
    print("  Saved → models/model_a/traditional/label_spreading.pkl")

    return results

# ===========================================================================
# Comparison Table
#
# FIX: Reference scores loaded dynamically from model_a_results.csv instead
# of being hardcoded. Shows val and test splits separately.
# ===========================================================================
def print_comparison(all_results, ref_acc, ref_f1):
    for split in ["Val", "Test"]:
        rows = [r for r in all_results if r['split'] == split]
        if not rows:
            continue

        print(f"\n{'='*75}")
        print(f"{'UNSUPERVISED vs SUPERVISED — ' + split:^75}")
        print(f"{'='*75}")
        print(f"{'Model':<38} {'Accuracy':>10} {'Macro F1':>10} {'vs Supervised F1':>14}")
        print(f"{'-'*75}")

        for r in rows:
            if ref_f1 is not None:
                diff = r['f1'] - ref_f1
                sign = '+' if diff >= 0 else ''
                diff_str = f"{sign}{diff:.4f}"
            else:
                diff_str = "N/A"
            purity_str = f"  (purity={r['purity']:.4f})" if 'purity' in r else ""
            print(f"{r['name']:<38} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
                  f"{diff_str:>14}{purity_str}")

        if ref_f1 is not None:
            print(f"{'-'*75}")
            print(f"{'Best Supervised (reference)':<38} {ref_acc:>10.4f} "
                  f"{ref_f1:>10.4f} {'(reference)':>14}")

        print(f"{'='*75}")

# ===========================================================================
# Main
# ===========================================================================
def main():
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_train_mc, y_val_mc, y_test_mc) = load_features()

    # Load supervised reference scores dynamically
    ref_acc, ref_f1 = load_supervised_reference()

    # Reduce dimensions (required for clustering and label propagation)
    X_train_r, X_val_r, X_test_r, _ = reduce_dimensions(
        X_train, X_val, X_test, n_components=100
    )

    all_results = []

    # K-Means (binary k=2 + multiclass k=4)
    km_results = run_kmeans(
        X_train_r, X_val_r, X_test_r,
        y_train, y_val, y_test,
        y_train_mc, y_val_mc, y_test_mc
    )
    all_results.extend(km_results)

    # GMM (binary k=2 + multiclass k=4)
    gmm_results = run_gmm(
        X_train_r, X_val_r, X_test_r,
        y_train, y_val, y_test,
        y_train_mc, y_val_mc, y_test_mc
    )
    all_results.extend(gmm_results)

    # Label Spreading (semi-supervised, binary labels)
    lp_results = run_label_propagation(
        X_train_r, X_val_r, X_test_r,
        y_train, y_val, y_test,
        labeled_ratio=0.05,
        max_samples=20000
    )
    all_results.extend(lp_results)

    # Comparison table
    print_comparison(all_results, ref_acc, ref_f1)

    # Save summary CSV
    summary = [{k: v for k, v in r.items() if k != 'confusion_matrix'}
               for r in all_results]
    pd.DataFrame(summary).to_csv(
        PROCESSED_DIR + 'model_a_unsupervised_results.csv', index=False
    )
    print("\nResults saved → data/processed/model_a_unsupervised_results.csv")
    print("Next step: run model_a_ensemble.py")

if __name__ == '__main__':
    main()