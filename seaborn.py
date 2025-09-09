import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("tips")
sns.histplot(df["total_bill"], bins=20, KDE= True, color="sky blue")
plt.title("histogram + KDE")
plt.show()