import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# 1. Load Data
# =====================
#import os
file_path = "/Users/apple/Desktop/python/Rainfall_data.csv"

df = pd.read_csv(file_path)


print("Data loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# =====================
# 2. Preprocessing
# =====================
# Define binary target: Did it rain significantly? (e.g., >1mm)
df["RainfallClass"] = (df["Precipitation"] > 1.0).astype(int)

X = df.drop(columns=["Precipitation", "RainfallClass"])
y = df["RainfallClass"]

# =====================
# 3. Train/Validation Split
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# 4. Model Training
# =====================
model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# =====================
# 5. Evaluation
# =====================
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Cross-Validation Accuracy:", cv_scores.mean(), "±", cv_scores.std())

# =====================
# 6. Predictions on Full Dataset
# =====================
df["PredictedRainClass"] = model.predict(X)

output = df[["Year", "Month", "Day", "PredictedRainClass"]]
output_file = r"C:\Users\user\Desktop\project\rainfall_predictions.csv"
output.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")

# =====================
# 7. Save Model
# =====================
model_file = r"C:\Users\user\Desktop\project\rainfall_rf_model.pkl"
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")

# =====================
# 8. KMeans Clustering (Elbow Method)
# =====================
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("The Elbow Method (Rainfall Data)")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
