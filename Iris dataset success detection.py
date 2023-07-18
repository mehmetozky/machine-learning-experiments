import pandas as pd 
import numpy as np 
#                                       DESİCİON TREE CLASSİFİER
veriler= pd.read_csv("Iris.csv")

x=veriler.iloc[:,1:5]
y=veriler.iloc[:,5:]

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="entropy")

dtc.fit(x_train, y_train)
y_pred=dtc.predict(x_test)

from sklearn.metrics import confusion_matrix
cmDT=confusion_matrix(y_test, y_pred)
# Accuracy rate %96

#                                          K-NN CLASSİFİER
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3,metric="minkowski")
knn.fit(x_train, y_train)
y_pred2=knn.predict(x_test)

cmKNN=confusion_matrix(y_test, y_pred2)

#                                               Lojistik Regreesion 
from sklearn.linear_model import LogisticRegression

log_re=LogisticRegression()

log_re.fit(x_train, y_train)
y_pred3=log_re.predict(x_test)

cmLOGR=confusion_matrix(y_test, y_pred3)
#                                              Naif Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

nb.fit(x_train, y_train)
y_pred4=nb.predict(x_test)

cmNB=confusion_matrix(y_test, y_pred4)

#                                        Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=5,criterion="entropy")
rf.fit(x_train, y_train)
y_pred5=rf.predict(x_test)
cmRF=confusion_matrix(y_test, y_pred5)

#                                       SVC ALGORİTMA
from sklearn.svm import SVC

svc=SVC(kernel="linear")
svc.fit(x_train, y_train)
y_pred6=svc.predict(x_test)

cmSVC=confusion_matrix(y_test, y_pred6)














