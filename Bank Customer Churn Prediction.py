import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("Bank Customer Churn Prediction.csv")

credit_score=veriler.iloc[:,1:2]

düzenliveriseti=veriler.iloc[:,4:11]
countryset=veriler.iloc[:,2:3]
genderset=veriler.iloc[:,3:4]
churn= veriler.iloc[:,11]


from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

one=OneHotEncoder()

genderveriseti=one.fit_transform(genderset).toarray()
countryveriseti=one.fit_transform(countryset).toarray()

kolon1=pd.DataFrame(data=düzenliveriseti,index=range(10000),columns=["age","tenure","balance","products_number","credit_card","active_member","estimated_salary"])
kolon2=pd.DataFrame(data=genderveriseti,index=range(10000),columns=["Female","Male"])
kolon3=pd.DataFrame(data=countryveriseti,index=range(10000),columns=["France","Spain","Germany"])
kolon4=pd.DataFrame(data=credit_score,index=range(10000),columns=["credit_score"])

X=pd.concat([kolon1,kolon2,kolon3,kolon4],axis=1)
Y=pd.DataFrame(data=churn,index=range(10000),columns=["churn"])

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

Knn=KNeighborsClassifier(n_neighbors=10,metric="minkowski")

Knn.fit(X_train, Y_train)

Y_pred=Knn.predict(X_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test, Y_pred)















