# Credit Card Fraud Detection Project

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset
df = pd.read_csv("creditcard.csv")

# Remove Extra Spaces from Column Names
df.columns = df.columns.str.strip()

# Display First 5 Rows
print("\nFirst 5 Rows: - creditcard_fraud_detection.py:25")
print(df.head())

# Dataset Information
print("\nDataset Information: - creditcard_fraud_detection.py:29")
print(df.info())

# Missing Values
print("\nMissing Values: - creditcard_fraud_detection.py:33")
print(df.isnull().sum())

# Print Column Names
print("\nColumn Names: - creditcard_fraud_detection.py:37")
print(df.columns.tolist())

# Check if 'Class' column exists
if 'Class' not in df.columns:
    print("\nERROR: 'Class' column not found in dataset. - creditcard_fraud_detection.py:42")
    print("Please check your dataset column names. - creditcard_fraud_detection.py:43")
    exit()

# Class Distribution
print("\nClass Distribution: - creditcard_fraud_detection.py:47")
print(df['Class - creditcard_fraud_detection.py:48'].value_counts())

# Visualization
plt.figure(figsize=(6, 4))

sns.countplot(x='Class', data=df)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()

# Features and Target
X = df.drop('Class', axis=1)
y = df['Class']

# Normalize Amount and Time Columns
scaler = StandardScaler()

X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score: - creditcard_fraud_detection.py:92")
print(accuracy)

# Classification Report
print("\nClassification Report: - creditcard_fraud_detection.py:96")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix: - creditcard_fraud_detection.py:102")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()