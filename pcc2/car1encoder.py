import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load dataset
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(r"C:\Users\pc21\Desktop\pcc\car.data", names=columns)

print("First 5 rows:\n", df.head())
print("\nMissing values check:\n", df.isnull().sum())

# OneHotEncode features (excluding target column 'class')
X = df.drop("class", axis=1)
y = df["class"]

# Use pandas get_dummies (simpler than OneHotEncoder)
X_encoded = pd.get_dummies(X, drop_first=False)  

print("\nEncoded feature columns:\n", X_encoded.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
print("\nTraining accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
