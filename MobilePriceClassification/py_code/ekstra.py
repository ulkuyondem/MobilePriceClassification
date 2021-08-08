import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix,accuracy_score

data = pd.read_csv('train.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

def model(model):
    global X,y,X_train, X_test, y_train, y_test
    print(type(model).__name__)   
    plot_confusion_matrix(model,X_test,y_test,cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show() 
    
#################################################   RF
print("##########    RF        ##########")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier().fit(X_train,y_train)
pred = rfc.predict(X_test)
acs = accuracy_score(y_test,pred)
rf_deger =  acs*100
print("Accuracy Score             :",rf_deger)
model(rfc) 

################################################   BAYES
print("##########    BAYES        ##########")
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_acc_score = accuracy_score(y_test, nbpred)
bayes_deger =  nb_acc_score*100
print("Accuracy Score             :",bayes_deger)
model(nb)

###############################################    SVC
print("##########    SVC        ##########")
from sklearn.svm import SVC
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
svc_deger =  svc_acc_score*100
print("Accuracy Score             :",svc_deger)
model(svc)

