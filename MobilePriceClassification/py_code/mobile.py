import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('train.csv')
X = data.iloc[:,:20].values
y = data.iloc[:,20:21].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

data.head()

def model(model):
    global X,y,X_train, X_test, y_train, y_test
    pred = model.predict(X_test)
    acs = accuracy_score(y_test,pred) 
    print("Accuracy Score             :",acs*100)
    
    plot_confusion_matrix(model,X_test,y_test,cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show() 
def class_report(y_test, y_preds): 
    print(classification_report(y_test, y_preds))

####################        KNN          ############################
print("##########    KNN       ##########")

knn = KNeighborsClassifier()
KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

a = knn.score(X_test, y_test) #knn değeri
knn_score = a*100
print(knn_score)

cm = confusion_matrix(y_test, y_pred)
print(cm)

model(knn)
class_report(y_test,y_pred)

####################        DNN          ############################
print("##########    DNN        ##########")
from tensorflow.keras import models, Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv('train.csv')
X = data.iloc[:,:20].values
y = data.iloc[:,20:21].values

#Standardizasyon 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print('Normalized data:')
print(X[0])
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print('One hot encoded array:')
print(y[0:5])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#Model
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=20, validation_split=0.001)

y_pred = model.predict(X_test)

#tahminleri etikete dönüştürme
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#etikete dönüştürme
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
dnn_score = a*100
print('Accuracy:', a*100) #dnn değeri

cm = confusion_matrix(pred,test)
print(cm)
print(classification_report(pred, test))

import seaborn as sns
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()

######################     HTML    ################################
from flask import Flask, render_template
app = Flask(__name__) 
@app.route("/index")
def index():
    return render_template("index.html",knn_deger=knn_score, dnn_deger = dnn_score)
if __name__ == "__main__":
    app.run(debug=True)

