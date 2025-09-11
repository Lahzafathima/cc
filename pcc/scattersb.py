import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df= sns.load_dataset("tips")

sns.scatterplot(x="total_bill", y="tip", data=df, hue="sex", style="time")
plt.title("scatter plot")
plt.show()