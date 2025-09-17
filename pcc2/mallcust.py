# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as pl
# from sklearn.model_selection import train_test_split


# # Load dataset
# columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# df = pd.read_csv(r"C:\Users\pc21\Downloads\Mall_Customers.csv", names=columns)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.cluster import KMeans

# 1. Load dataset
customer_data = pd.read_csv('Mall_Customers.csv')
print(customer_data.head())

X= customer_data.iloc[:,[3,4]].values
print(X)

#WCC ->within Clusters sum of square
wcss=[]
for i in range(1,11):
    kMeans = KMeans(n_clusters=i, init='k-means++',random_state=42)
    kMeans.fit(X)
    wcss.append(kMeans.inertia_)


# Elbow Method using Seaborn
plt.figure(figsize=(8,5))
sns.lineplot(x=range(1, 11), y=wcss, marker="o", markersize=8, linewidth=2)
plt.title("Elbow Method - Optimal K", fontsize=14)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("WCSS (Within Cluster Sum of Squares)", fontsize=12)
plt.show()

clusters=5

kmeans=KMeans(n_clusters=5, init='k-means++', random_state=0)
y = kmeans.fit_predict(X)
print(y)

clusters =0,1,2,3,4

plt.figure(figsize=(8,8))
plt.scatter(X[y==0,0],X[y==0,1],s=50,c='blue',label='cluster 1')
plt.scatter(X[y==1,0],X[y==1,1],s=50,c='green',label='cluster 2')
plt.scatter(X[y==2,0],X[y==2,1],s=50,c='pink',label='cluster 3')
plt.scatter(X[y==3,0],X[y==3,1],s=50,c='black',label='cluster 4')
plt.scatter(X[y==4,0],X[y==4,1],s=50,c='gray',label='cluster 5')

#centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='red',label='centroids')
plt.title('customer group')
plt.xlabel('annual income')
plt.show()
































# # 2. Handle missing values
# data.fillna(data.mean(), inplace=True)  # numerical imputation

# # 3. Encode categorical variables
# le = LabelEncoder()
# data['Category'] = le.fit_transform(data['Category'])  # label encoding

# # One-hot encoding (if needed)
# data = pd.get_dummies(data, columns=['City'])  

# # 4. Feature scaling
# scaler = StandardScaler()
# X = data.drop("Target", axis=1)
# y = data["Target"]
# X_scaled = scaler.fit_transform(X)

# # 5. Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
