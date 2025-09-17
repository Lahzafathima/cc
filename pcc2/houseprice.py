# import numpy as np
# # Example: square feet values (X)
# X = np.array([500, 750, 1000, 1250, 1500, 1750, 2000])
# # Example: price values (Y) - could be directly proportional to square feet
# Y = np.array([250000, 375000, 500000, 625000, 750000, 875000, 1000000])
# print("X (Square Feet):", X)
# print("Y (Price):", Y)


# import pandas as pd
# # Create dataset as dictionary
# data = {
#     "SquareFeet": [500, 750, 1000, 1250, 1500, 1750, 2000],
#     "Price": [250000, 375000, 500000, 625000, 750000, 875000, 1000000]
# }
# # Convert into DataFrame
# df = pd.DataFrame(data)
# print(df)


# import numpy as np
# import pandas as pd
# # Generate random square feet values
# X = np.random.randint(500, 2000, 50)  # 50 samples between 500 and 2000 sq ft
# # Assume price is 500 * sq ft + some noise
# Y = 500 * X + np.random.randint(-20000, 20000, 50)
# # Put into DataFrame
# df = pd.DataFrame({"SquareFeet": X, "Price": Y})
# print(df.head())

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#dataset house size
x= np.array([500,800,1000,1200,1500]).reshape(-1,1) 
#sqft
y= np.array([150000,200000,230000,260000,300000])
#price in $
model = LinearRegression()
model.fit(x,y)
# model parmeters
print("Intercept (base price):", model.intercept_)
print("Slop (price per sqft):", model.coef_[0])
size =1100
