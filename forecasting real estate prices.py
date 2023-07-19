import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
 
data=pd.read_csv("multilinearregression.csv",sep=";")
plt.title("house prediction", color="orange")
plt.scatter(data["alan"], data["fiyat"], color="blue")
plt.xlabel("Alan", color="red")
plt.ylabel("fiyat", color="green")

x=data.iloc[:,0:3]
y=data.iloc[:,3:]

x_test=[230,4,10]

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x, y)
y_pred=lr.predict(x_test)
    