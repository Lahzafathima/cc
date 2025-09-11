import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df= sns.load_dataset("tips")

sns.countplot(x="day", data= df, palette="Set2")
plt.title("count plot")
plt.show()