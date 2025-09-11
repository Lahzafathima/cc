import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# data

# load dataset
data =  load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

df_small =df[["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "target"]]

#split dataset
x = df_small.drop(columns=["target"])
y = df_small["target"]

x_train, x_test, y_train,y_test=train_test_split(
    x, y,test_size=0.2, random_state=42, stratify=y
)

# model creation
#3 train logistic regression
model = LogisticRegression()
model.fit(x_train, y_train)

# evalution
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nclassification Report:\n", classification_report(y_test, y_pred))

# evaluation

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, xticklabels=["Malignant", "Benign"], yticklabels=["Malignant","Benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# prediction

# user input prediction
print("\n -- Breast Cancer Prediction --")

mean_radius = float(input("Enter mean radius (e.g. 14.5):"))
mean_texture = float(input("Enter mean texture (e.g. 20.0):"))
mean_perimeter = float(input("Enter mean perimeter (e.g. 100.0):"))
mean_area = float(input("Enter mean area (e.g. 900.0):"))
mean_smoothness = float(input("Enter mean smoothness (e.g. 0.1):"))

# create dataframe for user input

user_data ={
    "mean radius":mean_radius,"mean texture": mean_texture, "mean perimeter": mean_perimeter, "mean area": mean_area, "mean smoothness": mean_smoothness
}
user_df= pd.DataFrame([user_data])

#predict
prediction = model.predict(user_df)[0]

if prediction == 1:
    print(f"\n prediction: Benign (not cancerous)")
else:
    print(f"\nPrediction: Malignant (cancerous)")