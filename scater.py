import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x = np.random.rand(500)
y = np.random.rand(500)
plt.scatter(x, y, color='gray', marker= 'x')
plt.title("scatter plot")
plt.show()