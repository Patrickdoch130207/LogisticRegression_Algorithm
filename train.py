import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt
from LogReg import LogisticRegression

bc = datasets.load_breast_cancer()

X,y = bc.data, bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

df = pd.DataFrame(X,columns=bc.feature_names)
df['target'] = y

print(df.head())

print(bc.target_names)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clasifier = LogisticRegression(lr=0.01)
clasifier.fit(X_train,y_train)
y_pred = clasifier.predict(X_test)

def accuracy(y_pred, y_test):
  return np.sum((y_pred == y_test)/len(y_test))

print(accuracy(y_pred,y_test))
