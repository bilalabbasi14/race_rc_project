import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer

# Paths
PROCESSED_DIR = 'data/processed/'
MODEL_DIR = 'models/model_a/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

class QuestionGenerator:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        self.ranker = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        
    def extract_sentences(self, article):
        """Rule-based sentence splitter."""
        sentences = re.split(r'(?<=[.!?])\s+', str(article))
        return [s.strip() for s in sentences if len(s.strip()) > 10]
        
    def get_candidate_sentences(self, article, answer_text, top_n=3):
        """Extract candidate sentences using keyword overlap with correct answer."""
        sentences = self.extract_sentences(article)
        if not sentences:
            return []
            
        ans_words = set(answer_text.lower().split())
        if not ans_words:
            return sentences[:top_n]
            
        scores = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(ans_words & s_words)
            scores.append(overlap)
            
        # Get top indices
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [sentences[i] for i in top_indices]

    def apply_wh_templates(self, sentence):
        """Apply simple Wh-word templates to form questions."""
        # A simple classical ML constraint template approach
        lower_s = sentence.lower()
        questions = []
        
        # Heuristics for Wh- word
        if any(w in lower_s for w in ['in', 'at', 'on', 'where']):
            questions.append(f"Where {sentence}?")
        if any(w in lower_s for w in ['when', 'year', 'day', 'time', 'after', 'before']):
            questions.append(f"When {sentence}?")
        if any(w in lower_s for w in ['who', 'he', 'she', 'they', 'person', 'man', 'woman']):
            questions.append(f"Who {sentence}?")
        
        # Default fallbacks
        questions.append(f"What {sentence}?")
        questions.append(f"Why {sentence}?")
        
        # Clean up double punctuation
        questions = [q.replace('?.', '?').replace('?!', '?') for q in questions]
        return list(set(questions)) # unique

    def _extract_features(self, questions, answers):
        """Extract features for the ranking model."""
        # Bag of words
        X_bow = self.vectorizer.transform(questions)
        
        # Lexical features
        lex_features = []
        for q, a in zip(questions, answers):
            q_words = set(str(q).lower().split())
            a_words = set(str(a).lower().split())
            overlap = len(q_words & a_words)
            q_len = len(q_words)
            lex_features.append([overlap, q_len])
            
        X_lex = np.array(lex_features)
        return hstack([X_bow, X_lex])

    def train_ranker(self, train_df):
        """Train the question ranker to distinguish real vs fake questions."""
        print("Preparing training data for Question Ranker...")
        
        # Real questions from dataset
        real_q = train_df['question'].tolist()
        # Correct answer option
        real_ans = []
        for _, row in train_df.iterrows():
            correct_opt = row['answer']
            real_ans.append(row[correct_opt])
            
        y_real = [1] * len(real_q)
        
        # Generate fake/bad questions for negative sampling
        fake_q = []
        fake_ans = []
        print("Generating negative samples...")
        for i, row in train_df.head(len(train_df)).iterrows(): # sample same amount
            correct_opt = row['answer']
            ans_text = str(row[correct_opt])
            
            # extract candidate sentences
            cands = self.get_candidate_sentences(row['article'], ans_text, top_n=1)
            if cands:
                templates = self.apply_wh_templates(cands[0])
                fake_q.append(templates[0]) # pick first
            else:
                fake_q.append("What is this?")
            fake_ans.append(ans_text)
            
        y_fake = [0] * len(fake_q)
        
        X_text = real_q + fake_q
        ans_text = real_ans + fake_ans
        y = y_real + y_fake
        
        print("Fitting vectorizer...")
        self.vectorizer.fit(X_text)
        
        print("Extracting features...")
        X = self._extract_features(X_text, ans_text)
        
        print("Training Random Forest ranker...")
        self.ranker.fit(X, y)
        print("Ranker trained.")
        
    def save(self):
        joblib.dump(self.vectorizer, MODEL_DIR + 'qgen_vectorizer.pkl')
        joblib.dump(self.ranker, MODEL_DIR + 'qgen_ranker.pkl')
        print(f"Saved generator artifacts to {MODEL_DIR}")
        
    def load(self):
        self.vectorizer = joblib.load(MODEL_DIR + 'qgen_vectorizer.pkl')
        self.ranker = joblib.load(MODEL_DIR + 'qgen_ranker.pkl')
        print("Loaded generator artifacts.")
        
    def generate_and_rank(self, article, answer_text):
        """Generate questions and return the highest ranked one."""
        candidates = self.get_candidate_sentences(article, answer_text, top_n=2)
        all_questions = []
        for cand in candidates:
            all_questions.extend(self.apply_wh_templates(cand))
            
        if not all_questions:
            return "Could not generate a question."
            
        # Rank them
        answers = [answer_text] * len(all_questions)
        X = self._extract_features(all_questions, answers)
        
        scores = self.ranker.predict_proba(X)[:, 1] # Probability of being a 'real' question
        
        best_idx = np.argmax(scores)
        return all_questions[best_idx], scores[best_idx]


def main():
    print("Loading raw data for generation task...")
    # Load raw data to get correct answers easily
    train_df = pd.read_csv('data/raw/train.csv').dropna().sample(5000, random_state=42) # limit for speed
    
    gen = QuestionGenerator()
    gen.train_ranker(train_df)
    gen.save()
    
    # Test generation
    print("\n--- Testing Question Generation ---")
    sample = train_df.iloc[0]
    ans_text = sample[sample['answer']]
    print(f"Article snippet: {sample['article'][:200]}...")
    print(f"Answer: {ans_text}")
    print(f"Original Question: {sample['question']}")
    
    gen_q, score = gen.generate_and_rank(sample['article'], ans_text)
    print(f"Generated Question: {gen_q} (Score: {score:.4f})")

if __name__ == '__main__':
    main()
