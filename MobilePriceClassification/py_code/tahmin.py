import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('train.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

######################## KNN TAHMİN  ############################
knn = KNeighborsClassifier()
KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

data_test=pd.read_csv('test.csv')
data_test.head()

data_test=data_test.drop('id',axis=1)
data_test.head()    ##ValueError: X ve Y matrisleri için uyumsuz boyut: X.shape [1] == 21 iken Y.shape [1] == 20

predicted_price=knn.predict(data_test)
predicted_price 

data_test['price_range']=predicted_price
data_test

######################## DNN TAHMİN  ############################
from tensorflow.keras import models, Sequential
from tensorflow.keras.layers import Dense
import numpy as np
data = pd.read_csv('train.csv')
X = data.iloc[:,:20].values
y = data.iloc[:,20:21].values
#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=20, validation_split=0.001)
##tahmin
data_test=pd.read_csv('test.csv')
data_test.head()

data_test=data_test.drop('id',axis=1)
data_test.head() 

predicted_price=model.predict(data_test)
predicted_price

inverted = list()
for i in range(len(predicted_price)):
    inverted.append(np.argmax(predicted_price[i]))

data_test['price_range']=inverted
data_test

