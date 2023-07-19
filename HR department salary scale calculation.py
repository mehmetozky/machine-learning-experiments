import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


"""
Region manager ile Country manager arasındaki yeni (4.5)Managerlik açarsak onun maaşı 
ne olur şimdi bunu bulmaya çalışalım

"""

veriler=pd.read_csv("salary scale.csv",sep=";")
x=veriler.iloc[:,0:1].values
y=veriler.iloc[:,1:]

plt.title("MAAS SKALASI", color="black")
plt.scatter(x,y,color="red")
plt.xlabel("DENEYIM", color="purple")
plt.ylabel("MAAS", color="purple")




poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)

newManager=poly_reg.fit_transform([[4.5]])

lr=LinearRegression()
lr.fit(x_poly, y)




lr2=LinearRegression()
lr2.fit(x, y)




plt.plot(x, lr2.predict(x),color="navy") # Linear Regression
plt.plot(x, lr.predict(x_poly),color="black") # Polynomial regression 
New_manager=lr.predict(newManager)












