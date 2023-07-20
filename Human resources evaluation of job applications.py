import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

df=pd.read_csv("CV.csv")

düzenliveri=df[["Deneyim Yili","Eski Calistigi Firmalar"]]
numeric=df[["SuanCalisiyor?","Top10 Universite?","StajBizdeYaptimi?"]]
egitim_seviye=df[["Egitim Seviyesi"]]
y=df.iloc[:,6:]

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

ohe=OneHotEncoder()
lb=LabelEncoder()
# Katagoric--> Numeric

egitim_seviyesiNC=ohe.fit_transform(egitim_seviye).toarray()
numericLB=lb.fit_transform(df["SuanCalisiyor?"])
numericLB2=lb.fit_transform(df["Top10 Universite?"])
numericLB3=lb.fit_transform(df["StajBizdeYaptimi?"])

newData=pd.DataFrame(data=egitim_seviyesiNC,columns=["BS","PhD","MS"])
newData2=pd.DataFrame(data=düzenliveri,columns=["Deneyim Yili","Eski Calistigi Firmalar"])
newData3=pd.DataFrame(data=numericLB,columns=["SuanCalisiyor?"])
newData4=pd.DataFrame(data=numericLB2,columns=["Top10 Universite?"])
newData5=pd.DataFrame(data=numericLB3,columns=["StajBizdeYaptimi?"])    
newData6=pd.DataFrame(data=y,columns=["IseAlindi"])

tumData=pd.concat([newData,newData2,newData3,newData4,newData5,newData6],axis=1)
x=pd.concat([newData,newData2,newData3,newData4,newData5],axis=1)
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(x, newData6)
Ypred=dtc.predict( [ [1,0,0,11,4,1,0,0] ] ) # TRUE












