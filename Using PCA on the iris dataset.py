import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

data=pd.read_csv("Iris.csv")

x=data.iloc[:,1:5]
y=data.iloc[:,5:]

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X=sc.fit_transform(x)


# 4D to 2D compression using PCA

from sklearn.decomposition import PCA

pca=PCA(n_components=2)

principalComponents=pca.fit_transform(X)



indirgenmişData=pd.DataFrame(data=principalComponents,columns=["new1","new2"])

newData=pd.concat([indirgenmişData,y],axis=1)

df_setosa=newData[newData["Species"]=="Iris-setosa"]
df_versicolor=newData[newData["Species"]=="Iris-versicolor"]
df_virginica=newData[newData["Species"]=="Iris-virginica"]

plt.xlabel("new1")
plt.ylabel("new2")
plt.scatter(df_setosa["new1"], df_setosa["new2"], color="green")
plt.scatter(df_virginica["new1"], df_virginica["new2"], color="red")
plt.scatter(df_versicolor["new1"], df_versicolor["new2"], color="blue")

# varyans=%95

varyans=pca.explained_variance_ratio_.sum()





 

