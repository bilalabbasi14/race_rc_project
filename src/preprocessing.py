import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Paths
RAW_DIR       = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ===========================================================================
# Load Data
# ===========================================================================
def load_data():
    train_df = pd.read_csv(RAW_DIR + 'train.csv')
    val_df   = pd.read_csv(RAW_DIR + 'val.csv')
    test_df  = pd.read_csv(RAW_DIR + 'test.csv')
    print(f"Loaded → Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df

# ===========================================================================
# Clean Text
# ===========================================================================
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace
    return text

def clean_dataframe(df):
    df = df.copy()
    df['article']  = df['article'].apply(clean_text)
    df['question'] = df['question'].apply(clean_text)
    df['A']        = df['A'].apply(clean_text)
    df['B']        = df['B'].apply(clean_text)
    df['C']        = df['C'].apply(clean_text)
    df['D']        = df['D'].apply(clean_text)
    return df

# ===========================================================================
# Expand to Binary Format
#
# FIX: Now saves article, question, and option as SEPARATE columns alongside
# the combined text column. This allows model_a_train.py to compute accurate
# lexical features using real boundaries instead of heuristic splits.
#
# Each question → 4 rows (one per option A/B/C/D).
# label = 1 if that option is the correct answer, else 0.
# ===========================================================================
def expand_to_binary(df):
    rows = []
    for _, row in df.iterrows():
        correct = row['answer']   # 'A', 'B', 'C', or 'D'
        for opt in ['A', 'B', 'C', 'D']:
            combined = row['article'] + ' ' + row['question'] + ' ' + row[opt]
            label    = 1 if opt == correct else 0
            rows.append({
                'text':     combined,       # kept for CountVectorizer
                'article':  row['article'], # FIX: separate column
                'question': row['question'],# FIX: separate column
                'option':   row[opt],       # FIX: separate column
                'label':    label
            })
    return pd.DataFrame(rows)

# ===========================================================================
# One-Hot Encoding via CountVectorizer (binary=True)
#
# CountVectorizer with binary=True is equivalent to One-Hot / Bag-of-Words
# encoding. Vocabulary is capped at top 15,000 words to keep memory
# manageable. Fitted on training text only; val/test are transformed only.
# ===========================================================================
def build_features(train_texts, val_texts, test_texts, max_features=15000):
    vectorizer = CountVectorizer(
        binary=True,           # One-Hot style: 1 if word present, 0 if not
        max_features=max_features,
        stop_words='english'
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)
    print(f"Feature matrix → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, vectorizer

# ===========================================================================
# Encode Answer Labels
#
# FIX: Now produces TWO label arrays and saves BOTH:
#
#   y_*_multiclass  →  A/B/C/D mapped to 0/1/2/3, one value per QUESTION.
#                      Needed for unsupervised clustering (K-Means, GMM) and
#                      semi-supervised methods (Label Propagation) in Model A.
#
#   y_*_bin         →  0/1 correctness label, one value per OPTION ROW in the
#                      binary-expanded dataframe. Used for supervised training
#                      of all classifiers in model_a_train.py.
# ===========================================================================
def encode_labels_multiclass(df):
    """One label per question row: A→0, B→1, C→2, D→3."""
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return df['answer'].map(mapping).values

# ===========================================================================
# Cosine Similarity Matrix (for Model B distractor ranking)
#
# FIX: Added — the spec's data-flow diagram lists a cosine similarity matrix
# as a preprocessing output. We compute it on a vocabulary built from the
# training binary-expanded text, then save a (vocab_size × vocab_size) sparse
# dot-product matrix so downstream code doesn't have to recompute it.
#
# Because the full matrix is huge we compute it lazily as a normalised term
# co-occurrence (each word vector is its One-Hot row normalised to unit L2).
# ===========================================================================
def build_cosine_sim_matrix(vectorizer, X_train):
    """
    Build a word-level cosine-similarity proxy matrix.

    Approach: normalise each column (word) of X_train so that dot(col_i, col_j)
    equals the cosine similarity between the two word occurrence vectors across
    all training documents.  Saves as a dense float32 array of shape
    (vocab_size, vocab_size).

    NOTE: With vocab_size=15000 the matrix is 15000×15000 ≈ 900 MB float32.
    We therefore save only the top-K (K=50) most similar neighbours for each
    word as a dict instead of the full matrix — this is the practical format
    used by Model B.
    """
    from sklearn.preprocessing import normalize
    from scipy.sparse import csr_matrix

    print("Building cosine similarity neighbour map (top-50 per word)...")
    vocab_size = len(vectorizer.vocabulary_)

    # Column-normalised: each column = occurrence vector of one word
    # Shape: (n_docs, vocab_size) — already sparse
    X_norm = normalize(X_train, norm='l2', axis=0)   # normalise per word

    # Dot product between all pairs of word vectors: (vocab_size, vocab_size)
    # We do this in chunks to avoid OOM
    CHUNK = 500
    id2word = {v: k for k, v in vectorizer.vocabulary_.items()}
    top_k_neighbours = {}   # word → [(similar_word, score), ...]

    for start in range(0, vocab_size, CHUNK):
        end = min(start + CHUNK, vocab_size)
        # cols for this chunk: shape (n_docs, chunk_size) → transpose → (chunk, n_docs)
        chunk_T = X_norm[:, start:end].T              # (chunk, n_docs)
        # dot with all word vectors: (chunk, vocab_size)
        sims = chunk_T.dot(X_norm).toarray() if hasattr(chunk_T.dot(X_norm), 'toarray') \
               else chunk_T.dot(X_norm)

        for local_i, global_i in enumerate(range(start, end)):
            row = sims[local_i]
            row[global_i] = -1.0                      # exclude self
            top_k_idx = np.argpartition(row, -50)[-50:]
            top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]
            top_k_neighbours[id2word[global_i]] = [
                (id2word[j], float(row[j])) for j in top_k_idx if row[j] > 0
            ]

        if start % 5000 == 0:
            print(f"  Cosine sim: processed {start}/{vocab_size} words...")

    print("  Cosine similarity map done.")
    return top_k_neighbours

# ===========================================================================
# Save Everything
# ===========================================================================
def save_artifacts(X_train, X_val, X_test,
                   y_train_bin, y_val_bin, y_test_bin,
                   y_train_multiclass, y_val_multiclass, y_test_multiclass,
                   vectorizer,
                   train_binary, val_binary, test_binary,
                   cosine_sim_map):

    # One-Hot feature matrices (binary expanded — one row per option)
    joblib.dump(X_train,    PROCESSED_DIR + 'X_train.pkl')
    joblib.dump(X_val,      PROCESSED_DIR + 'X_val.pkl')
    joblib.dump(X_test,     PROCESSED_DIR + 'X_test.pkl')

    # Binary labels (0/1 correctness — aligned with X_train/val/test rows)
    joblib.dump(y_train_bin, PROCESSED_DIR + 'y_train.pkl')
    joblib.dump(y_val_bin,   PROCESSED_DIR + 'y_val.pkl')
    joblib.dump(y_test_bin,  PROCESSED_DIR + 'y_test.pkl')

    # FIX: Multiclass labels (0-3 for A-D — one per original question row)
    # Required by the unsupervised / semi-supervised component of Model A
    joblib.dump(y_train_multiclass, PROCESSED_DIR + 'y_train_multiclass.pkl')
    joblib.dump(y_val_multiclass,   PROCESSED_DIR + 'y_val_multiclass.pkl')
    joblib.dump(y_test_multiclass,  PROCESSED_DIR + 'y_test_multiclass.pkl')

    # Fitted vectorizer (needed for inference)
    joblib.dump(vectorizer, PROCESSED_DIR + 'vectorizer.pkl')

    # FIX: Binary-expanded dataframes now include separate article/question/option
    # columns — model_a_train.py uses these for accurate lexical feature engineering
    train_binary.to_csv(PROCESSED_DIR + 'train_binary.csv', index=False)
    val_binary.to_csv(  PROCESSED_DIR + 'val_binary.csv',   index=False)
    test_binary.to_csv( PROCESSED_DIR + 'test_binary.csv',  index=False)

    # FIX: Cosine similarity neighbour map (used by Model B distractor ranker)
    joblib.dump(cosine_sim_map, PROCESSED_DIR + 'cosine_sim_map.pkl')

    print("All artifacts saved to data/processed/")
    print("  X_train / X_val / X_test        → sparse One-Hot matrices")
    print("  y_train / y_val / y_test         → binary 0/1 labels")
    print("  y_train_multiclass / ...         → A/B/C/D → 0/1/2/3 labels")
    print("  vectorizer.pkl                   → fitted CountVectorizer")
    print("  train_binary / val / test .csv   → expanded rows w/ separate columns")
    print("  cosine_sim_map.pkl               → top-50 word neighbours by cosine sim")

# ===========================================================================
# Main Pipeline
# ===========================================================================
def main():
    # 1. Load
    train_df, val_df, test_df = load_data()

    # 2. Clean text in all columns
    print("Cleaning text...")
    train_df = clean_dataframe(train_df)
    val_df   = clean_dataframe(val_df)
    test_df  = clean_dataframe(test_df)

    # 3. Encode multiclass answer labels (per question) — saved separately
    #    FIX: these are now actually saved, not silently discarded
    y_train_multiclass = encode_labels_multiclass(train_df)
    y_val_multiclass   = encode_labels_multiclass(val_df)
    y_test_multiclass  = encode_labels_multiclass(test_df)
    print(f"Multiclass label balance (train): "
          f"A={( y_train_multiclass==0).sum()} "
          f"B={(y_train_multiclass==1).sum()} "
          f"C={(y_train_multiclass==2).sum()} "
          f"D={(y_train_multiclass==3).sum()}")

    # 4. Expand each question into 4 binary rows (one per option)
    #    FIX: binary CSVs now include article / question / option as separate columns
    print("Expanding to binary format (4 rows per question)...")
    train_binary = expand_to_binary(train_df)
    val_binary   = expand_to_binary(val_df)
    test_binary  = expand_to_binary(test_df)
    print(f"Binary rows → Train: {len(train_binary)}, "
          f"Val: {len(val_binary)}, Test: {len(test_binary)}")
    print(f"Binary columns: {list(train_binary.columns)}")

    # 5. Build One-Hot feature matrices from the combined 'text' column
    print("Building One-Hot feature matrices (CountVectorizer, binary=True)...")
    X_train, X_val, X_test, vectorizer = build_features(
        train_binary['text'],
        val_binary['text'],
        test_binary['text']
    )

    # Binary correctness labels aligned with the expanded rows
    y_train_bin = train_binary['label'].values
    y_val_bin   = val_binary['label'].values
    y_test_bin  = test_binary['label'].values

    print(f"Label balance (train binary) — "
          f"correct: {y_train_bin.sum()}, incorrect: {(y_train_bin==0).sum()}")

    # 6. FIX: Build cosine similarity neighbour map (spec requirement, used by Model B)
    cosine_sim_map = build_cosine_sim_matrix(vectorizer, X_train)

    # 7. Save all artifacts
    save_artifacts(
        X_train, X_val, X_test,
        y_train_bin, y_val_bin, y_test_bin,
        y_train_multiclass, y_val_multiclass, y_test_multiclass,
        vectorizer,
        train_binary, val_binary, test_binary,
        cosine_sim_map
    )

    print("\nPreprocessing complete.")
    print("Next step: run model_a_train.py")

if __name__ == '__main__':
    main()