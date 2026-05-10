# AI-Powered Reading Comprehension and Quiz Generation System

## Table of Contents
- [Introduction](#introduction)
- [Architecture and Models](#architecture-and-models)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Evaluation](#evaluation)
- [Testing](#testing)

## Introduction
This project is an end-to-end Reading Comprehension and Quiz Generation system built using classical Machine Learning techniques (adhering strictly to non-deep-learning constraints). Given a reading passage, the system is capable of verifying the correctness of multiple-choice answers, generating plausible distractors (incorrect options), and providing intelligent hints to guide users toward the correct answer.

The project features a full data preprocessing pipeline, multiple trained classifier layers, ensemble methods, and an interactive Streamlit user interface equipped with an analytics dashboard to track session metrics.

## Architecture and Models
The system is divided into two primary logical layers:

### Model A: Answer Verification and Scoring
Model A acts as the core reading comprehension engine. It evaluates a given article, question, and candidate answer, outputting a probability score representing the correctness of the answer.
- **Features:** Utilizes a combination of One-Hot Bag-of-Words encoding and handcrafted lexical features (word overlap, length ratios, article coverage, cosine similarity).
- **Classifiers:** Trained using Logistic Regression, Linear SVM, Naive Bayes, and Random Forest.
- **Ensembles:** Combines base classifiers using Hard Voting, Soft Voting, and Stacking (with a Logistic Regression meta-classifier).
- **Unsupervised Methods:** Implements K-Means clustering and Label Propagation for unsupervised and semi-supervised evaluations.

### Model B: Distractor and Hint Generation
Model B is responsible for creating the pedagogical components of the quiz.
- **Distractor Generation:** Generates incorrect but highly plausible multiple-choice options by utilizing a Cosine Similarity Word Map built from the training corpus, followed by a Ranker model that ensures the distractors are grammatically and semantically sound.
- **Hint Generation:** Produces contextual hints (e.g., cloze-style sentences) by identifying sentences in the source article that possess high lexical overlap with the correct answer, safely omitting the direct answer to avoid leaking the solution.

## Installation and Setup

1. **Clone the repository and navigate to the project root:**
   ```bash
   cd race_rc_project
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   The repository includes a `.env` file that sets `PYTHONPATH=src` to ensure local module imports resolve correctly.

## Usage

### 1. Data Preprocessing
Before training, process the raw RACE dataset (train, val, test CSVs) into binary expanded feature matrices:
```bash
python src/preprocessing.py
```

### 2. Model Training
Train the classifiers in sequence:
```bash
python src/model_a_train.py
python src/model_a_unsupervised.py
python src/model_a_ensemble.py
python src/model_b_train.py
```

### 3. Running the Streamlit UI
To launch the interactive quiz interface and analytics dashboard:
```bash
streamlit run ui/app.py
```
This application allows you to paste reading passages, automatically evaluate user answers, generate hints on demand, and view live pipeline latency and performance metrics.

## Code Structure

- `/data/raw/`: Original RACE dataset files (train, val, test).
- `/data/processed/`: Artifacts generated during preprocessing (feature matrices, labels, similarity maps) and evaluation reports.
- `/models/`: Serialized model artifacts (.pkl files) for fast inference.
- `/src/`: Core Python pipeline scripts.
  - `preprocessing.py`: Text cleaning, One-Hot encoding, lexical feature engineering.
  - `model_a_train.py` / `model_a_ensemble.py` / `model_a_unsupervised.py`: Training routines for Model A.
  - `model_b_train.py`: Training and generation routines for Model B distractors and hints.
  - `inference.py`: Centralized API orchestrator that loads the `ModelBundle` and handles live UI requests.
  - `evaluate.py`: Standalone script for computing test-set metrics.
- `/ui/app.py`: Streamlit frontend application.
- `/tests/test.py`: Unit tests validating the inference API.

## Evaluation
To run a comprehensive evaluation of all trained models on the held-out test set:
```bash
python src/evaluate.py
```
This script computes Accuracy, Macro F1, Precision, Recall, and Exact Match for Model A, alongside Distractor/Hint Precision for Model B. The final aggregated summary is exported to `data/processed/full_evaluation_report.csv`.

## Testing
The project includes a suite of unit tests to verify the integrity of the unified inference API (loading artifacts, parsing answers, and formatting outputs).

Run the tests using Python's built-in `unittest` framework:
```bash
python tests/test.py -v
```
All tests should pass, ensuring that the `ModelBundle` successfully manages state and latency logging without exceptions.
