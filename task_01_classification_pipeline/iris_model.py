# ==============================================================================
# TASK 1: IRIS FLOWER CLASSIFICATION PIPELINE
# AUTHOR: Arpita Jaiswal
# DOMAIN: Artificial Intelligence & Machine Learning
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------------------
# STEP 1: Data Ingestion & Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
print("=== Step 1: Loading and Exploring Dataset ===")
from sklearn.datasets import load_iris
iris_data = load_iris()

# Convert to DataFrame for enterprise data handling
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 Rows of the Dataset:")
print(df.head())

# Visualizing distributions to verify feature variance (Saves plot locally)
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue='species_name', palette='Set2')
plt.savefig('iris_pairplot.png')
print("\n[INFO] Pairplot visualization saved successfully as 'iris_pairplot.png'.")

# ------------------------------------------------------------------------------
# STEP 2: Data Preprocessing & Train-Test Split
# ------------------------------------------------------------------------------
print("\n=== Step 2: Preprocessing and Splitting Data ===")
X = df[iris_data.feature_names]
y = df['species']

# 80-20 Train-Test split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize features (Mean=0, Volatility=1) for faster convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Matrix Shape: {X_train_scaled.shape}")
print(f"Testing Matrix Shape: {X_test_scaled.shape}")

# ------------------------------------------------------------------------------
# STEP 3: Model Training (Logistic Regression)
# ------------------------------------------------------------------------------
print("\n=== Step 3: Training Logistic Regression Classifier ===")
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)
print("[INFO] Model optimization complete.")

# ------------------------------------------------------------------------------
# STEP 4: Performance Evaluation
# ------------------------------------------------------------------------------
print("\n=== Step 4: Model Evaluation Metrics ===")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris_data.target_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)