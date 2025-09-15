import numpy as np 
import pandas as pd
import matplotlib.pyplot as pl
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load dataset
columns = ['buying', 'maint','doors','persons','lug_boot','safety','class']
df = pd.read_csv(r"C:\Users\pc21\Downloads\car.data",names=columns)

#print("First 5 rows of dataset:")
print(df.head())
#print("\nMissing values check:")
print(df.isnull().sum())

# Encode categorical values into numbers

le=LabelEncoder()
for col in df.columns:
    df[col]  =le.fit_transform(df[col])

# -----------------------------linear reg ------------------------

# label_encoders = {}
# for col in df.columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# print("\nEncoded Dataset (first 5 rows):")
#print(df.head())

# Features (X) and Target (y)
# X = df.drop("class", axis=1)
# y = df["class"]

# Split into training & testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# Predictions
# y_pred = model.predict(X_test)

# Evaluate
# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# print("R² Score:", r2_score(y_test, y_pred))

# ---------------------------linear reg ends ---------------

# Split features and target
X = df.drop("class", axis=1)
y = df["class"]
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Train-test split
print("training accuracy:", model.score(X, y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Test accuracy
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

print("Training accuracy:", model.score(X_train, y_train))
print("test accuracy:", model.score(X_test, y_test))