import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df= sns.load_dataset("tips")

sns.boxplot(x="day", y="total_bill", data= df, palette="pastel")
plt.title("box plot")
plt.show()