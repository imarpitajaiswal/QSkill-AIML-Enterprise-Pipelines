# ==============================================================================
# TASK 2: SPAM MAIL DETECTOR (NLP PIPELINE)
# AUTHOR: Arpita Jaiswal
# DOMAIN: Artificial Intelligence & Machine Learning
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ------------------------------------------------------------------------------
# STEP 1: Data Ingestion & Verification
# ------------------------------------------------------------------------------
print("=== Step 1: Loading Text Dataset ===")
# Utilizing a reliable public raw mirror of the SMS Spam Collection dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(f"Dataset Dimensions: {df.shape}")
print("\nClass Distribution:")
print(df['label'].value_counts(normalize=True) * 100)

# Convert labels to binary format: ham = 0, spam = 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ------------------------------------------------------------------------------
# STEP 2: Train-Test Split & Feature Extraction (TF-IDF Vectorization)
# ------------------------------------------------------------------------------
print("\n=== Step 2: Text Preprocessing & Vectorization ===")
X = df['message']
y = df['label_num']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Convert raw text into numeric features using TF-IDF while filtering English stopwords
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_transformed = tfidf.fit_transform(X_train_raw)
X_test_transformed = tfidf.transform(X_test_raw)

print(f"Vocabulary Size (Unique tokens extracted): {len(tfidf.vocabulary_)}")
print(f"Sparse Matrix Shape (Train): {X_train_transformed.shape}")

# ------------------------------------------------------------------------------
# STEP 3: Model Training (Multinomial Naive Bayes)
# ------------------------------------------------------------------------------
print("\n=== Step 3: Training Multinomial Naive Bayes Classifier ===")
spam_detector = MultinomialNB()
spam_detector.fit(X_train_transformed, y_train)
print("[INFO] NLP Model training finalized.")

# ------------------------------------------------------------------------------
# STEP 4: Corporate-Grade Metrics Evaluation
# ------------------------------------------------------------------------------
print("\n=== Step 4: System Performance Evaluation ===")
y_pred = spam_detector.predict(X_test_transformed)

# High-ticket corporate models focus heavily on Precision for Spam (minimizing False Positives)
print(f"Accuracy Score : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision Score: {precision_score(y_test, y_pred) * 100:.2f}% (High precision prevents marking valid emails as spam)")
print(f"Recall Score   : {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Optimal Score: {f1_score(y_test, y_pred) * 100:.2f}%")

print("\nDetailed Execution Breakdown:")
print(classification_report(y_test, y_pred, target_names=['Ham (Valid)', 'Spam']))