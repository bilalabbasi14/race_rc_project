import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Paths 
RAW_DIR       = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
os.makedirs(PROCESSED_DIR, exist_ok=True)

#Load Data 
def load_data():
    train_df = pd.read_csv(RAW_DIR + 'train.csv')
    val_df   = pd.read_csv(RAW_DIR + 'val.csv')
    test_df  = pd.read_csv(RAW_DIR + 'test.csv')
    print(f"Loaded → Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    return train_df, val_df, test_df

#Clean Text 
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

#Build Combined Text Column 
#For each row we concatenate article + question + one option
#This produces one row per (article, question, option) combination
#label = 1 if that option is the correct answer, else 0

def expand_to_binary(df):
    rows = []
    for _, row in df.iterrows():
        correct = row['answer']  # 'A', 'B', 'C', or 'D'
        for opt in ['A', 'B', 'C', 'D']:
            combined = row['article'] + ' ' + row['question'] + ' ' + row[opt]
            label = 1 if opt == correct else 0
            rows.append({'text': combined, 'label': label})
    return pd.DataFrame(rows)

#One-Hot Encoding via CountVectorizer (binary=True)     
#CountVectorizer with binary=True is equivalent to One-Hot Encoding
#We limit vocab to top 15000 words to keep memory manageable

def build_features(train_texts, val_texts, test_texts, max_features=15000):
    vectorizer = CountVectorizer(
        binary=True,          # One-Hot style: 1 if word present, 0 if not
        max_features=max_features,
        stop_words='english'
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)
    print(f"Feature matrix → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, vectorizer

#Encode Answer Labels (A/B/C/D → 0/1/2/3)
def encode_labels(df):
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return df['answer'].map(mapping).values

#Save Everything 
def save_artifacts(X_train, X_val, X_test,
                   y_train, y_val, y_test,
                   vectorizer,
                   train_binary, val_binary, test_binary):

    joblib.dump(X_train,    PROCESSED_DIR + 'X_train.pkl')
    joblib.dump(X_val,      PROCESSED_DIR + 'X_val.pkl')
    joblib.dump(X_test,     PROCESSED_DIR + 'X_test.pkl')
    joblib.dump(y_train,    PROCESSED_DIR + 'y_train.pkl')
    joblib.dump(y_val,      PROCESSED_DIR + 'y_val.pkl')
    joblib.dump(y_test,     PROCESSED_DIR + 'y_test.pkl')
    joblib.dump(vectorizer, PROCESSED_DIR + 'vectorizer.pkl')

    # Also save the binary-expanded dataframes (needed for Model A training)
    train_binary.to_csv(PROCESSED_DIR + 'train_binary.csv', index=False)
    val_binary.to_csv(  PROCESSED_DIR + 'val_binary.csv',   index=False)
    test_binary.to_csv( PROCESSED_DIR + 'test_binary.csv',  index=False)

    print("All artifacts saved to data/processed/")

#Main Pipeline 
def main():
    # Load
    train_df, val_df, test_df = load_data()

    # Clean
    print("Cleaning text...")
    train_df = clean_dataframe(train_df)
    val_df   = clean_dataframe(val_df)
    test_df  = clean_dataframe(test_df)

    # Encode answer labels (for multiclass use: 0,1,2,3)
    y_train = encode_labels(train_df)
    y_val   = encode_labels(val_df)
    y_test  = encode_labels(test_df)

    # Expand each row into 4 binary rows (one per option)
    print("Expanding to binary format...")
    train_binary = expand_to_binary(train_df)
    val_binary   = expand_to_binary(val_df)
    test_binary  = expand_to_binary(test_df)

    # Build One-Hot features on binary-expanded text
    print("Building One-Hot feature matrices...")
    X_train, X_val, X_test, vectorizer = build_features(
        train_binary['text'],
        val_binary['text'],
        test_binary['text']
    )

    y_train_bin = train_binary['label'].values
    y_val_bin   = val_binary['label'].values
    y_test_bin  = test_binary['label'].values

    # Save
    save_artifacts(X_train, X_val, X_test,
                   y_train_bin, y_val_bin, y_test_bin,
                   vectorizer,
                   train_binary, val_binary, test_binary)

    print("Preprocessing complete.")

if __name__ == '__main__':
    main()