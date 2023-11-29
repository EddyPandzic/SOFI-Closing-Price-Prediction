# SOFIpriceprediction.py
# Author: Eddy Pandzic
# SOFI CAPM Analsis with linear regression, and closing price prediction with simple machine learning model


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df2 = pd.read_csv("SOFI.csv")

# Prints out first 5 rows of data from the csv
df2.head()

# Print summary statistics of SOFI stock
df2.describe()

# Plotting a time series of closing price
plt.figure(figsize=(10,4))
plt.title("SOFI Stock Price")
plt.xlabel("Date (in Days)")
plt.ylabel("Closing Price")
plt.plot(df2["Close"])

# Print out correlation chart, and correlation heatmap
print(df2.corr(numeric_only = True))
sns.heatmap(df2.corr(numeric_only = True))
plt.show()

# Applying data to basic machine learning model
x = df2[["Open", "High", "Low"]]
y = df2[["Close"]]
x = x.to_numpy()
y = y.to_numpy()
#Variable x will be the attributes for which the decision tree regression will find the SDR for
#Variable x will be the target variable, which will be taken as the average returned from an array of data sorted by the attributes
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Decision Tree Regression Algorithm
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted Closing Price": ypred})
print(data.head())

# Plotting Returns
plt.plot(data)
plt.xlabel("Time(Days)")
plt.ylabel("Stock Price")
plt.title("Stock Prediction for Social Finance (SOFI)")