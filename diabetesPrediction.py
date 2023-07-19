import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
veriler=pd.read_csv("diabetes.csv")
a=veriler.head()
x=veriler.iloc[:,0:8]
y=veriler.iloc[:,8:]

saglıksiz=veriler[veriler["Outcome"]==1]
saglıklı=veriler[ veriler["Outcome"]==0]

plt.title("Healty", color="black")
plt.scatter(saglıklı["Age"], saglıklı["Glucose"], color="green",alpha=0.4,label="saglıklı")
plt.xlabel("Age", color="blue")
plt.ylabel("Glucose", color="blue")

plt.scatter(saglıksiz["Age"], saglıksiz["Glucose"], color="red",alpha=0.5,label="saglıksız")
plt.legend()
# verileri normalize etmeliyiz

x_normal=(x-np.min(x)) / (np.max(x)-np.min(x))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_normal,y,test_size=0.20,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)

accuracy=knn.score(x_test,y_test)

# accuracy=%77









