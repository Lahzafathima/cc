import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    "category": ['A','B', 'C', 'D'], "value": [4,7,2,9]
})
sns.barplot(x="category", y="value",data=data)
plt.title("normal bar chart (seaborn)")
plt.show()