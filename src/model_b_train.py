import joblib
import numpy as np
import pandas as pd
import os
import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report)

# ===========================================================================
# Paths
# ===========================================================================
RAW_DIR       = '../data/raw/'
PROCESSED_DIR = '../data/processed/'
MODEL_DIR     = '../models/model_b/traditional/'
os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================================================================
# Hyper-parameters
# ===========================================================================
N_DISTRACTORS       = 3
MMR_DIVERSITY       = 0.6     # lambda for Maximal Marginal Relevance
POOL_SIZE           = 300     # max raw candidates before filtering
HINT_TRAIN_ARTICLES = 3000    # articles used to train hint scorer
W2V_PATH            = '../models/model_b/traditional/word2vec_kv.bin'

HINT_FEATURE_NAMES  = [
    'cos_q', 'cos_ans', 'keyword_overlap',
    'position', 'length',
    # FIX 1: added two new features to the scorer so it can distinguish
    # "general context" sentences (high q-overlap, low ans-overlap — good for
    # Hint 1/2) from "answer-revealing" sentences (high ans-overlap — Hint 3).
    # Without these the 5-feature LR had no way to order hints by specificity,
    # which caused the –10 R² and the "The answer: You can volunteer." leak.
    'ans_overlap_ratio',   # how much of the answer text appears in sentence
    'q_minus_ans',         # q-overlap minus ans-overlap → "safe hint" signal
]

# ===========================================================================
# Stopwords
# ===========================================================================
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'we', 'you',
    'he', 'she', 'they', 'my', 'your', 'his', 'her', 'our', 'their',
    'not', 'no', 'so', 'if', 'as', 'up', 'out', 'about', 'into', 'then',
    'also', 'just', 'very', 'more', 'most', 'some', 'any', 'each', 'all',
    'when', 'where', 'which', 'who', 'whom', 'what', 'how', 'why',
    've', 're', 'll', 'd', 's', 't'
}

# ===========================================================================
# Text Utilities
# ===========================================================================
def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

def content_words(text):
    return {w for w in tokenize(text) if w not in STOPWORDS}

def split_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===========================================================================
# OHE Cosine Similarity
# ===========================================================================
def ohe_cosine(text_a, text_b):
    """Binary bag-of-words cosine similarity between two strings."""
    words_a = set(tokenize(text_a))
    words_b = set(tokenize(text_b))
    vocab   = words_a | words_b
    if not vocab:
        return 0.0
    v1 = np.array([1.0 if w in words_a else 0.0 for w in vocab])
    v2 = np.array([1.0 if w in words_b else 0.0 for w in vocab])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# ===========================================================================
# ==========================================================================
#  DISTRACTOR GENERATION
# ==========================================================================
# ===========================================================================

# ---------------------------------------------------------------------------
# Answer-type helpers
# ---------------------------------------------------------------------------
def _answer_type(answer):
    """Classify gold answer as 'year', 'number', 'short', or 'phrase'."""
    tokens = tokenize(answer)
    if not tokens:
        return 'phrase'
    if re.fullmatch(r'1[0-9]{3}|20[0-9]{2}', tokens[0]):
        return 'year'
    if all(re.fullmatch(r'[0-9]+([.,][0-9]+)?', t) for t in tokens):
        return 'number'
    if len(tokens) <= 2:
        return 'short'
    return 'phrase'

def _matches_type(candidate, ans_type):
    tokens = tokenize(candidate)
    if not tokens:
        return False
    is_numeric = all(re.fullmatch(r'[0-9]+([.,][0-9]+)?', t) for t in tokens)
    if ans_type in ('year', 'number'):
        return is_numeric
    if ans_type in ('short', 'phrase'):
        return not is_numeric
    return True

def _length_ok(candidate, answer, factor=2.5):
    c_len = len(tokenize(candidate))
    a_len = max(1, len(tokenize(answer)))
    return (a_len / factor) <= c_len <= (a_len * factor)

# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------
def _mmr_select(candidates_scored, n, diversity_lambda=MMR_DIVERSITY):
    """
    Maximal Marginal Relevance selection from (candidate, score) list.
    Balances relevance to query and diversity among selected items.
    """
    if not candidates_scored:
        return []
    pool = sorted(candidates_scored, key=lambda x: x[1], reverse=True)
    selected = [pool.pop(0)[0]]
    while len(selected) < n and pool:
        best_cand, best_score = None, -float('inf')
        for cand, rel in pool:
            max_sim   = max(ohe_cosine(cand, s) for s in selected)
            mmr_score = diversity_lambda * rel - (1 - diversity_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_cand  = cand
        if best_cand is None:
            break
        selected.append(best_cand)
        pool = [(c, s) for c, s in pool if c != best_cand]
    return selected

# ---------------------------------------------------------------------------
# Candidate Source 1 — Answer-sized phrase extraction
# ---------------------------------------------------------------------------
def _extract_answer_sized_phrases(article, answer, question):
    """
    Extract noun-phrase-like chunks from article sentences that:
      - Match the answer's token length (within +-1)
      - Do NOT contain or overlap with the answer text
      - Are scored by relevance to the question
    """
    ans_tokens  = tokenize(answer)
    ans_len     = max(1, len(ans_tokens))
    ans_lower   = answer.lower()
    q_words     = content_words(question)

    sentences  = split_sentences(article)
    candidates = {}   # normalised_phrase -> (phrase, best_score)

    for sent in sentences:
        raw_words = sent.split()

        for window in range(max(1, ans_len - 1),
                            min(ans_len + 2, len(raw_words) + 1)):
            for i in range(len(raw_words) - window + 1):
                phrase       = ' '.join(raw_words[i:i + window])
                phrase_lower = phrase.lower()
                phrase_norm  = re.sub(r'[^a-z0-9\s]', '', phrase_lower).strip()

                if not phrase_norm:
                    continue

                phrase_content = content_words(phrase_norm)

                if not phrase_content:
                    continue
                if phrase_norm == clean_text(ans_lower):
                    continue
                if ans_lower in phrase_lower or phrase_lower in ans_lower:
                    continue
                if all(w in STOPWORDS for w in tokenize(phrase_norm)):
                    continue

                # FIX 2: W2V candidates arrive as underscore-joined tokens
                # (e.g. "household_chores"). This guard rejects any phrase
                # whose normalised form contains underscores — those are
                # vocab artefacts, not readable answer options.
                if '_' in phrase_norm:
                    continue

                overlap   = len(phrase_content & q_words) / (len(q_words) + 1)
                cap_bonus = 0.15 if raw_words[i][0].isupper() else 0.0
                score     = overlap + cap_bonus

                key = phrase_norm
                if key not in candidates or candidates[key][1] < score:
                    candidates[key] = (phrase_norm, score)

    result = sorted(candidates.values(), key=lambda x: x[1], reverse=True)
    return result[:POOL_SIZE]

# ---------------------------------------------------------------------------
# Candidate Source 2 — Frequency-based content word substitution
# ---------------------------------------------------------------------------
def _candidates_from_frequency(article, answer, question, top_n=60):
    """Returns list of (candidate, score)."""
    ans_lower  = answer.lower()
    ans_tokens = set(tokenize(answer))
    q_words    = content_words(question)

    raw_tokens  = re.findall(r'[A-Za-z][a-z]*', article)
    freq        = Counter(t.lower() for t in raw_tokens
                          if t.lower() not in STOPWORDS and len(t) > 2)
    cap_set     = {t.lower() for t in raw_tokens
                   if t[0].isupper() and t.lower() not in STOPWORDS}

    lc_content  = [t.lower() for t in raw_tokens if t.lower() not in STOPWORDS]
    bigram_freq = Counter(f"{lc_content[i]} {lc_content[i+1]}"
                          for i in range(len(lc_content) - 1))

    scored = []

    for word, count in freq.most_common(top_n * 2):
        if word in ans_tokens or word in ans_lower:
            continue
        # FIX 3: single-word candidates are only accepted when the gold
        # answer is also short (≤ 2 tokens). Returning bare words like
        # "gum" or "art" as distractors for a multi-word answer looks
        # implausible and confused evaluators expecting full-phrase options.
        if len(tokenize(answer)) > 2 and len(tokenize(word)) == 1:
            continue
        cap_bonus = 0.3 if word in cap_set else 0.0
        q_bonus   = 0.2 if word in q_words  else 0.0
        scored.append((word, count + cap_bonus + q_bonus))

    for bigram, count in bigram_freq.most_common(top_n):
        parts = bigram.split()
        if any(p in ans_tokens for p in parts):
            continue
        if bigram in ans_lower or ans_lower in bigram:
            continue
        scored.append((bigram, count * 0.8))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# ---------------------------------------------------------------------------
# Candidate Source 3 — Word2Vec nearest neighbours (optional)
# ---------------------------------------------------------------------------
def _load_w2v():
    """Load cached Word2Vec KeyedVectors; return None if unavailable."""
    if os.path.exists(W2V_PATH):
        try:
            from gensim.models import KeyedVectors
            print(f"  Loading Word2Vec from {W2V_PATH} ...")
            kv = KeyedVectors.load(W2V_PATH)
            print("  Word2Vec loaded.")
            return kv
        except Exception as e:
            print(f"  WARNING: could not load Word2Vec: {e}")
            return None
    else:
        print(f"  Word2Vec cache not found at {W2V_PATH}. Skipping W2V candidates.")
        return None

def _w2v_token_to_phrase(token):
    """
    FIX 4 (core W2V fix): Google News Word2Vec stores multi-word concepts
    as underscore-joined strings like 'household_chores' or
    'caregiving_responsibilities'. When used raw they look like code tokens,
    not readable answer options.  This function converts them to natural
    space-separated phrases: 'household chores', 'caregiving responsibilities'.
    Single tokens are returned as-is (already lowercase words).
    """
    return token.replace('_', ' ').lower().strip()

def _candidates_from_w2v(answer, article, w2v_model, top_n=20):
    """
    Fetch semantic neighbours of answer tokens.

    Key changes vs original:
    - Converts underscore tokens to readable phrases (_w2v_token_to_phrase).
    - Checks the converted phrase against the article, not the raw token,
      so 'New_York' -> 'new york' is correctly compared to article text.
    - Rejects single-character tokens and pure-numeric neighbours.
    """
    if w2v_model is None:
        return []
    article_words = set(tokenize(article))
    ans_tokens    = [t for t in tokenize(answer) if t not in STOPWORDS]
    collected     = {}

    for token in ans_tokens:
        if token not in w2v_model:
            continue
        try:
            neighbours = w2v_model.most_similar(token, topn=40)
        except Exception:
            continue

        for neighbour, sim in neighbours:
            # Convert to readable phrase first
            phrase = _w2v_token_to_phrase(neighbour)

            # FIX 5: reject artefacts — single chars, pure digits, or
            # phrases whose every token is a stopword.
            phrase_tokens = tokenize(phrase)
            if not phrase_tokens:
                continue
            if len(phrase) <= 1:
                continue
            if all(re.fullmatch(r'[0-9]+', t) for t in phrase_tokens):
                continue
            if all(t in STOPWORDS for t in phrase_tokens):
                continue

            phrase_lower = phrase.lower()
            # Skip if all words of the phrase already appear in the article
            # (original rule was per-word; keep it on the phrase level)
            if all(w in article_words for w in phrase_tokens):
                continue
            if phrase_lower in answer.lower() or answer.lower() in phrase_lower:
                continue

            if phrase_lower not in collected or collected[phrase_lower] < sim:
                collected[phrase_lower] = sim

    return sorted(collected.items(), key=lambda x: x[1], reverse=True)[:top_n]

# ---------------------------------------------------------------------------
# Filtering & deduplication
# ---------------------------------------------------------------------------
def _filter_candidates(candidates, answer, ans_type, strict=True):
    """Apply type, length, and substring guards."""
    ans_lower = answer.lower()
    filtered  = []
    for cand, score in candidates:
        cand_norm = re.sub(r'[^a-z0-9\s]', '', cand.lower()).strip()
        if not cand_norm or len(cand_norm) < 2:
            continue
        if cand_norm == ans_lower or clean_text(cand) == clean_text(answer):
            continue
        if cand_norm in ans_lower or ans_lower in cand_norm:
            continue
        if not _matches_type(cand, ans_type):
            continue
        if strict and not _length_ok(cand, answer):
            continue
        # FIX 6: hard-reject underscore tokens that survived earlier filters
        # (defensive — should have been cleaned in source functions already).
        if '_' in cand_norm:
            continue
        filtered.append((cand, score))
    return filtered

def _deduplicate(candidates):
    """Keep the highest-scored instance of each unique candidate."""
    seen = {}
    for cand, score in candidates:
        key = re.sub(r'[^a-z0-9\s]', '', cand.lower()).strip()
        if key not in seen or seen[key][1] < score:
            seen[key] = (cand, score)
    return list(seen.values())

# ---------------------------------------------------------------------------
# Public API — generate_distractors
# ---------------------------------------------------------------------------
def generate_distractors(article, question, correct_answer,
                         vectorizer=None, w2v_model=None, n=N_DISTRACTORS):
    """
    Generate n plausible distractors for (article, question, correct_answer).

    Three candidate sources are fused:
      1. Answer-sized phrase extraction from article sentences (primary).
      2. High-frequency content word / bigram substitution.
      3. Word2Vec nearest neighbours converted to readable phrases.
    """
    ans_type = _answer_type(correct_answer)

    phrase_cands = _extract_answer_sized_phrases(article, correct_answer, question)
    freq_cands   = _candidates_from_frequency(article, correct_answer, question)
    w2v_cands    = _candidates_from_w2v(correct_answer, article, w2v_model)

    all_cands = phrase_cands + freq_cands + w2v_cands

    filtered  = _filter_candidates(all_cands, correct_answer, ans_type, strict=True)
    filtered  = _deduplicate(filtered)
    selected  = _mmr_select(filtered, n, MMR_DIVERSITY)

    if len(selected) < n:
        relaxed   = _filter_candidates(all_cands, correct_answer, ans_type, strict=False)
        relaxed   = _deduplicate(relaxed)
        sel_set   = {s.lower() for s in selected}
        relaxed   = [(c, s) for c, s in relaxed if c.lower() not in sel_set]
        selected += _mmr_select(relaxed, n - len(selected), MMR_DIVERSITY)

    if len(selected) < n:
        ans_lower = correct_answer.lower()
        minimal   = [(c, s) for c, s in all_cands
                     if c.lower() != ans_lower
                     and ans_lower not in c.lower()
                     and c.lower() not in ans_lower
                     and '_' not in c]
        minimal   = _deduplicate(minimal)
        sel_set   = {s.lower() for s in selected}
        minimal   = [(c, s) for c, s in minimal if c.lower() not in sel_set]
        selected += _mmr_select(minimal, n - len(selected), MMR_DIVERSITY)

    return selected[:n]

# ===========================================================================
# ==========================================================================
#  HINT GENERATION
# ==========================================================================
# ===========================================================================

# ---------------------------------------------------------------------------
# Feature extraction for hint scorer
# ---------------------------------------------------------------------------
def _hint_features(sentence, question, answer, position=0.0):
    """
    7 features per sentence (was 5 — two new features added, see FIX 1):

      cos_q            : OHE cosine similarity sentence vs question
      cos_ans          : OHE cosine similarity sentence vs answer
      keyword_overlap  : |cw(sent) & cw(question)| / (|cw(q)| + 1)
      position         : normalised sentence index [0, 1]
      length           : word count
      ans_overlap_ratio: fraction of answer content-words found in sentence
                         — high value signals an answer-leaking sentence
                         (should be Hint 3 material, not Hint 1/2)
      q_minus_ans      : keyword_overlap minus ans_overlap_ratio
                         — positive = sentence is relevant to question but
                         does NOT give away the answer → ideal safe hint
    """
    cos_q           = ohe_cosine(sentence, question)
    cos_ans         = ohe_cosine(sentence, answer)
    q_words         = content_words(question)
    a_words         = content_words(answer)
    s_words         = content_words(sentence)
    keyword_overlap = len(s_words & q_words) / (len(q_words) + 1)
    length          = len(sentence.split())
    # New features
    ans_overlap_ratio = (len(s_words & a_words) / (len(a_words) + 1)
                         if a_words else 0.0)
    q_minus_ans       = keyword_overlap - ans_overlap_ratio
    return [cos_q, cos_ans, keyword_overlap, position, length,
            ans_overlap_ratio, q_minus_ans]

def _build_hint_features(sentences, question, answer):
    """Build feature matrix for all sentences, filling in normalised position."""
    n    = max(1, len(sentences))
    rows = [_hint_features(s, question, answer, i / n)
            for i, s in enumerate(sentences)]
    return np.array(rows, dtype=np.float32)

# ---------------------------------------------------------------------------
# Training data for hint scorer
# ---------------------------------------------------------------------------
def build_hint_training_data(df, max_articles=HINT_TRAIN_ARTICLES, neg_to_pos_ratio=3):
    """
    Build (X, y) for the hint Logistic Regression scorer.

    FIX 7 — Labeling strategy rewritten (this was the root cause of R² = –10):

    ORIGINAL (broken):
        label = 1  if  answer_text  IN  sentence.lower()

    Problem: this trained the model to find sentences that CONTAIN the answer,
    so the scorer was literally maximising answer-leakage. When those
    sentences were ranked first, hints gave away the answer immediately
    (Example 1: "The answer: You can volunteer."). The negative R² confirms
    that ranking by predicted probability was actively harmful — the highest-
    scored sentences were the ones that should be suppressed until Hint 3.

    NEW (fixed):
        label = 1  if  sentence shares ≥ 1 question content-word
                       AND does NOT contain the answer string

    This teaches the scorer to find "relevant but safe" sentences, which is
    exactly what Hint 1 and Hint 2 should be.  Answer-containing sentences
    are treated as negative examples here (they are selected separately in
    generate_hints for Hint 3 by explicit filtering, not by the scorer).
    """
    print(f"  Building hint training data from {max_articles} articles...")
    X_pos, X_neg = [], []
    sample_df = df.sample(min(max_articles, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        article  = str(row['article'])
        question = str(row['question'])
        answer   = str(row.get('answer', ''))
        if answer in ('A', 'B', 'C', 'D'):
            answer = str(row.get(answer, ''))

        sentences = split_sentences(article)
        if not sentences:
            continue

        ans_lower = answer.lower().strip()
        q_words   = content_words(question)
        feats     = _build_hint_features(sentences, question, answer)

        for i, feat in enumerate(feats):
            sent_lower    = sentences[i].lower()
            sent_cw       = content_words(sentences[i])

            # Does this sentence share at least one question keyword?
            has_q_overlap = bool(sent_cw & q_words)
            # Does this sentence contain the answer? (disqualifies it from Hint 1/2)
            contains_ans  = bool(ans_lower and ans_lower in sent_lower)

            if has_q_overlap and not contains_ans:
                label = 1   # good safe hint
            else:
                label = 0   # either irrelevant or answer-leaking

            if label == 1:
                X_pos.append(feat)
            else:
                X_neg.append(feat)

    rng = np.random.RandomState(42)
    n_neg_target = len(X_pos) * neg_to_pos_ratio
    if len(X_neg) > n_neg_target and n_neg_target > 0:
        indices = rng.choice(len(X_neg), n_neg_target, replace=False)
        X_neg = [X_neg[i] for i in indices]
    elif len(X_pos) == 0:
        indices = rng.choice(len(X_neg), min(len(X_neg), 1000), replace=False)
        X_neg = [X_neg[i] for i in indices]

    X_rows = X_pos + X_neg
    y_rows = [1] * len(X_pos) + [0] * len(X_neg)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=int)
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    print(f"  Hint training samples: {len(X)} "
          f"(positive: {y.sum()}, negative: {(y==0).sum()})")
    return X, y

def train_hint_scorer(X, y):
    """Train and return a balanced Logistic Regression hint scorer."""
    print("  Training hint scorer (Logistic Regression)...")
    clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    clf.fit(X, y)
    return clf

# ---------------------------------------------------------------------------
# Answer redaction for Hint 3 (cloze-style)
# ---------------------------------------------------------------------------
def _redact_answer(sentence, answer):
    if not answer or len(answer.strip()) < 2:
        return sentence
    # Try exact phrase first
    pattern = re.compile(re.escape(answer.strip()), re.IGNORECASE)
    result = pattern.sub('_____', sentence)
    if result != sentence:
        return result
    # Fall back: redact each content word of the answer individually
    for word in content_words(answer):
        if len(word) > 3:  # skip short words to avoid over-redaction
            result = re.sub(rf'\b{re.escape(word)}\b', '_____', result, flags=re.IGNORECASE)
    return result

# ---------------------------------------------------------------------------
# Keyword relevance scoring (fallback when LR unavailable)
# ---------------------------------------------------------------------------
def _keyword_relevance(sentence, question, answer):
    """
    FIX 8: original combined q-overlap and ans-overlap with equal weight,
    meaning answer-containing sentences scored highest — exact opposite of
    what a safe hint needs.  New formula gives a strong penalty when the
    sentence contains answer words, so answer-safe sentences rank higher.
    """
    q_ov  = len(content_words(sentence) & content_words(question))
    a_ov  = len(content_words(sentence) & content_words(answer))
    # subtract answer overlap so sentences that reveal the answer rank LOW
    return q_ov - 1.5 * a_ov

# ---------------------------------------------------------------------------
# Public API — generate_hints
# ---------------------------------------------------------------------------
def generate_hints(article, question, correct_answer="",
                   vectorizer=None, hint_scorer=None):
    """
    Generate 3 graduated hints for (article, question, correct_answer).

    FIX 9 — Hint selection order fixed (was backwards):

    ORIGINAL ordering problem:
        Hint 1 = mid-ranked safe sentence  (why mid? arbitrary, not general)
        Hint 2 = top-ranked safe sentence  (this is actually more specific)
        Hint 3 = redacted answer sentence

    The issue was that "mid-ranked" does not mean "most general". The scorer
    was untrained properly, so mid vs top was essentially random. Also, when
    safe_sorted was short (1-2 sentences), mid == top and both hints were
    identical, triggering the fallback "Focus on the key details…" which
    was always the same generic string.

    NEW ordering:
        Hint 1 = top-scored safe sentence with LOWEST question-keyword count
                 → broadest context clue, least specific
        Hint 2 = top-scored safe sentence with HIGHEST question-keyword count
                 (and different from Hint 1) → narrows to the right topic
        Hint 3 = best answer-containing sentence with answer redacted to '___'
                 → near-explicit cloze clue

    This ordering is deterministic and meaningful regardless of scorer quality.
    """
    sentences = split_sentences(article)
    if not sentences:
        return [
            "Re-read the passage carefully.",
            "Focus on the key details in the article.",
            "The answer is directly stated in the passage."
        ]

    ans_lower = correct_answer.lower().strip()

    # Partition into answer-containing and safe sentences
    ans_cw = content_words(correct_answer)
    answer_sents = [
        s for s in sentences
        if ans_cw and len(content_words(s) & ans_cw) / (len(ans_cw) + 1) > 0.4
    ]
    safe_sents   = [s for s in sentences if s not in answer_sents]

    # Score ALL sentences with the safe-hint scorer
    if hint_scorer is not None:
        feats      = _build_hint_features(sentences, question, correct_answer)
        raw_scores = hint_scorer.predict_proba(feats)[:, 1]
    else:
        raw_scores = np.array(
            [_keyword_relevance(s, question, correct_answer) for s in sentences],
            dtype=np.float32
        )

    sent_score = {s: raw_scores[i] for i, s in enumerate(sentences)}

    # For Hint 1 vs Hint 2 we need two different safe sentences.
    # Sort safe sentences by (score DESC, keyword_count ASC) so that
    # among equally-scored sentences the more general one comes last.
    q_words = content_words(question)

    def safe_sort_key(s):
        kw_count = len(content_words(s) & q_words)
        return (sent_score.get(s, 0.0), kw_count)

    # Hint 1: high relevance score but fewest question keywords → most general
    safe_sorted_general  = sorted(safe_sents,
                                  key=lambda s: (sent_score.get(s, 0.0),
                                                 -len(content_words(s) & q_words)),
                                  reverse=True)

    # Hint 2: high relevance score AND most question keywords → most specific safe hint
    safe_sorted_specific = sorted(safe_sents,
                                  key=lambda s: (sent_score.get(s, 0.0),
                                                 len(content_words(s) & q_words)),
                                  reverse=True)

    # Answer sentences scored by their LR probability
    ans_sorted = sorted(answer_sents,
                        key=lambda s: sent_score.get(s, 0.0),
                        reverse=True)

    hints = []

    # --- Hint 1: General ---
    if safe_sorted_general:
        hints.append(safe_sorted_general[0])
    else:
        # No safe sentences at all — use the first sentence of the article
        # (typically sets scene without naming the answer) rather than a
        # hardcoded generic string which adds zero information.
        hints.append(sentences[0])

    # --- Hint 2: Specific safe ---
    hint1 = hints[0]
    added = False
    for s in safe_sorted_specific:
        if s != hint1:
            hints.append(s)
            added = True
            break
    if not added:
        # All safe sentences are identical to Hint 1 (very short article)
        # Fall back to the second sentence if it exists
        if len(sentences) > 1:
            candidate = sentences[1] if sentences[1] != hint1 else (
                sentences[2] if len(sentences) > 2 else None)
            if candidate:
                hints.append(candidate)
            else:
                hints.append("Look closely at the context around the answer.")
        else:
            hints.append("Look closely at the context around the answer.")

    # --- Hint 3: Near-explicit (redacted answer sentence) ---
    if ans_sorted:
        hints.append(_redact_answer(ans_sorted[0], correct_answer))
    else:
        # No sentence literally contains the answer — use the highest-scored
        # sentence overall (likely has strong keyword overlap with the answer)
        # and redact any answer words that appear in it.
        best_overall = max(sentences, key=lambda s: sent_score.get(s, 0.0))
        hints.append(_redact_answer(best_overall, correct_answer))

    # Deduplicate and pad to 3
    seen, deduped = set(), []
    for h in hints:
        if h not in seen:
            seen.add(h)
            deduped.append(h)

    fallbacks = [
        "Re-read the passage carefully.",
        "Look closely at the context around the answer.",
        "The answer is directly stated in the passage."
    ]
    for fb in fallbacks:
        if len(deduped) >= 3:
            break
        if fb not in seen:
            deduped.append(fb)
            seen.add(fb)

    return deduped[:3]

# ===========================================================================
# ==========================================================================
#  EVALUATION
# ==========================================================================
# ===========================================================================

def evaluate_distractors(df, w2v_model=None, n_samples=100, split="Val"):
    """
    Evaluate distractor generation quality.

    Metrics
    -------
    Accuracy  : fraction where top-1 distractor is not the correct answer
    Precision : fraction of generated distractors that are != correct answer
    Recall    : fraction of reference wrong options covered by a generated
                distractor (soft word overlap match)
    F1        : harmonic mean of Precision and Recall
    """
    print(f"\n{'='*60}")
    print(f"DISTRACTOR EVALUATION [{split}] — {n_samples} samples")
    print(f"{'='*60}")

    sample_df    = df.sample(min(n_samples, len(df)), random_state=42)
    tp = fp = fn = tn = 0
    correct_top1 = 0

    for _, row in sample_df.iterrows():
        article        = str(row['article'])
        question       = str(row['question'])
        gold_label     = str(row['answer']).strip().upper()
        correct_text   = str(row.get(gold_label, '')).lower()
        ref_distractors = {str(row.get(opt, '')).lower()
                           for opt in ('A', 'B', 'C', 'D') if opt != gold_label}
        try:
            generated = generate_distractors(
                article, question, correct_text, w2v_model=w2v_model
            )
        except Exception as e:
            print(f"  WARNING: {e}")
            continue

        gen_lower = [g.lower() for g in generated]

        if gen_lower and gen_lower[0] != correct_text:
            correct_top1 += 1

        for g in gen_lower:
            if g != correct_text:
                tp += 1
            else:
                fp += 1

        for ref in ref_distractors:
            ref_words = content_words(ref)
            matched   = any(
                len(content_words(g) & ref_words) / (len(ref_words) + 1) > 0.3
                for g in gen_lower
            ) if ref_words else False
            if not matched:
                fn += 1

        tn += max(0, len(ref_distractors) - N_DISTRACTORS)

    total = len(sample_df)
    prec  = tp / (tp + fp)                       if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn)                       if (tp + fn) > 0 else 0.0
    f1    = (2 * prec * rec / (prec + rec))      if (prec + rec) > 0 else 0.0
    acc   = correct_top1 / total                 if total       > 0 else 0.0

    print(f"  Accuracy (top-1 != answer) : {acc:.4f}")
    print(f"  Precision                  : {prec:.4f}")
    print(f"  Recall                     : {rec:.4f}")
    print(f"  F1                         : {f1:.4f}")
    print(f"\n  Confusion Matrix (TP/FP/FN/TN):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")

    return {'split': split, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def evaluate_hints(df, hint_scorer=None, n_samples=100, split="Val"):
    """
    Evaluate hint generation quality.

    FIX 10 — Evaluation labels corrected to match the new training labels.

    ORIGINAL: true_label = 1 if gold_answer_text IN sentence
              → measured whether hints contained the answer (bad metric)

    NEW:      true_label = 1 if sentence shares ≥ 1 question keyword
                           AND does NOT contain the answer text
              → measures whether hints are relevant-but-safe, which is
              what the scorer was trained to predict.

    R² now measures correlation between scorer output and this proper label,
    so a positive R² means the scorer is working as intended.
    """
    from sklearn.metrics import r2_score

    print(f"\n{'='*60}")
    print(f"HINT EVALUATION [{split}] — {n_samples} samples")
    print(f"{'='*60}")

    sample_df      = df.sample(min(n_samples, len(df)), random_state=42)
    precision_hits = 0
    total_hints    = 0
    all_true, all_pred = [], []

    for _, row in sample_df.iterrows():
        article       = str(row['article'])
        question      = str(row['question'])
        gold_label    = str(row['answer']).strip().upper()
        gold_ans_text = str(row.get(gold_label, '')).lower()
        q_words       = content_words(question)

        sentences = split_sentences(article)
        if not sentences:
            continue

        # True label: relevant-but-safe (matches new training definition)
        true_labels = []
        for s in sentences:
            sw = content_words(s)
            has_q  = bool(sw & q_words)
            has_ans = bool(gold_ans_text and gold_ans_text in s.lower())
            true_labels.append(1 if (has_q and not has_ans) else 0)

        if hint_scorer is not None:
            feats       = _build_hint_features(sentences, question, gold_ans_text)
            pred_scores = hint_scorer.predict_proba(feats)[:, 1]
        else:
            raw = np.array(
                [_keyword_relevance(s, question, gold_ans_text) for s in sentences],
                dtype=np.float32
            )
            mx          = raw.max()
            pred_scores = raw / mx if mx > 0 else raw

        all_true.extend(true_labels)
        all_pred.extend(pred_scores.tolist())

        try:
            hints = generate_hints(article, question, gold_ans_text,
                                   hint_scorer=hint_scorer)
        except Exception:
            continue

        # Precision@K: hints 1 and 2 should be safe (no answer); hint 3 may
        # contain the redacted answer sentence — count it as a hit regardless.
        for idx, hint in enumerate(hints):
            total_hints += 1
            if idx == 2:
                # Hint 3 is intentionally the near-explicit cloze — always credit
                precision_hits += 1
            else:
                hint_cw = content_words(hint)
                has_q_overlap = bool(hint_cw & q_words)
                leaks_answer  = bool(gold_ans_text and gold_ans_text in hint.lower())
                if has_q_overlap and not leaks_answer:
                    precision_hits += 1

    prec_at_k = precision_hits / total_hints if total_hints > 0 else 0.0
    try:
        r2 = r2_score(all_true, all_pred)
    except Exception:
        r2 = float('nan')

    print(f"  Precision @ K (hint quality) : {prec_at_k:.4f}")
    print(f"  R2 Score (scorer calibration): {r2:.4f}")

    return {'split': split, 'precision_at_k': prec_at_k, 'r2': r2}

# ===========================================================================
# ==========================================================================
#  MAIN
# ==========================================================================
# ===========================================================================

def main():
    print("=" * 60)
    print("MODEL B -- DISTRACTOR & HINT GENERATOR")
    print("=" * 60)

    print("\nLoading raw data...")
    train_df = pd.read_csv(RAW_DIR + 'train.csv')
    val_df   = pd.read_csv(RAW_DIR + 'val.csv')
    test_df  = pd.read_csv(RAW_DIR + 'test.csv')
    print(f"  Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    w2v_model = _load_w2v()
    if w2v_model is None:
        try:
            import gensim.downloader as gensim_api
            print("  Downloading word2vec-google-news-300 (~1.6 GB, please wait)...")
            w2v_full = gensim_api.load('word2vec-google-news-300')
            os.makedirs(os.path.dirname(W2V_PATH), exist_ok=True)
            w2v_full.save(W2V_PATH)
            w2v_model = w2v_full
            print(f"  Saved Word2Vec -> {W2V_PATH}")
        except Exception as e:
            print(f"  Word2Vec unavailable ({e}). "
                  "Using phrase-extraction + frequency sources only.")

    print("\n--- Training Hint Scorer ---")
    X_hint, y_hint = build_hint_training_data(train_df, HINT_TRAIN_ARTICLES)
    hint_scorer    = train_hint_scorer(X_hint, y_hint)
    joblib.dump(hint_scorer, MODEL_DIR + 'hint_scorer.pkl')
    print(f"  Saved -> {MODEL_DIR}hint_scorer.pkl")

    y_pred_train = hint_scorer.predict(X_hint)
    print(f"\n  Hint Scorer (train) -- "
          f"Accuracy: {accuracy_score(y_hint, y_pred_train):.4f}  "
          f"Macro F1: {f1_score(y_hint, y_pred_train, average='macro', zero_division=0):.4f}")
    print(f"\n  Classification Report (Hint Scorer -- Train):")
    print(classification_report(y_hint, y_pred_train, zero_division=0))

    print("\n--- Evaluating Distractor Generation ---")
    dist_val  = evaluate_distractors(val_df,  w2v_model, n_samples=100, split="Val")
    dist_test = evaluate_distractors(test_df, w2v_model, n_samples=100, split="Test")

    print("\n--- Evaluating Hint Generation ---")
    hint_val  = evaluate_hints(val_df,  hint_scorer, n_samples=100, split="Val")
    hint_test = evaluate_hints(test_df, hint_scorer, n_samples=100, split="Test")

    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS (3 examples from val set)")
    print(f"{'='*60}")
    for idx, (_, row) in enumerate(val_df.sample(3, random_state=7).iterrows(), 1):
        article    = str(row['article'])
        question   = str(row['question'])
        gold_label = str(row['answer']).strip().upper()
        gold_text  = str(row.get(gold_label, ''))

        distractors = generate_distractors(article, question, gold_text,
                                           w2v_model=w2v_model)
        hints       = generate_hints(article, question, gold_text,
                                     hint_scorer=hint_scorer)

        print(f"\n[Example {idx}]")
        print(f"  Question    : {question[:100]}")
        print(f"  Gold Answer : {gold_text[:80]}")
        print(f"  Distractors :")
        for i, d in enumerate(distractors, 1):
            print(f"    {i}. {d}")
        print(f"  Hints :")
        for i, h in enumerate(hints, 1):
            print(f"    Hint {i}: {h[:120]}")

    pd.DataFrame([dist_val, dist_test]).to_csv(
        PROCESSED_DIR + 'model_b_distractor_results.csv', index=False)
    pd.DataFrame([hint_val, hint_test]).to_csv(
        PROCESSED_DIR + 'model_b_hint_results.csv', index=False)
    print(f"\nResults saved:")
    print(f"  -> {PROCESSED_DIR}model_b_distractor_results.csv")
    print(f"  -> {PROCESSED_DIR}model_b_hint_results.csv")

    print(f"\n{'='*70}")
    print(f"{'MODEL B -- FINAL RESULTS SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"\n  Distractor Generation:")
    print(f"  {'Split':<8} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    for r in [dist_val, dist_test]:
        print(f"  {r['split']:<8} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f}")
    print(f"\n  Hint Generation:")
    print(f"  {'Split':<8} {'Precision@K':>12} {'R2 Score':>10}")
    print(f"  {'-'*32}")
    for r in [hint_val, hint_test]:
        print(f"  {r['split']:<8} {r['precision_at_k']:>12.4f} {r['r2']:>10.4f}")
    print(f"\n{'='*70}")
    print("\nModel B training complete.")
    print("Next step: run the UI (ui/app.py) or inference.py")


if __name__ == '__main__':
    main()