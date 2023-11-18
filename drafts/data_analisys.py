import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import XGBRegressor


data = pd.read_csv("path")
print(data.shape)
print(data.head())
print(data.desctibe())

# preprocess
print(data.isnull().sum())
data = data.dropna()
data.replace({"Loan_Status": {"N": 0, "Y": 1}}, inplace=True)
print(data["Dependents"].value_counts())
data.replace({"Dependents":{"3+":4}}, inplace = True)
print(data["Dependents"].value_counts())

# visualization
corr = data.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws= {'size':8})

sns.countplot(x = 'Education', hue = 'Loan_Status', data = data)
data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':0,'Female':1},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# data prep
X = data.drop(columns = ["Loan_Status", "Loan_ID"], axis = 1)
y = data["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify = y, random_state = 42)

# model


model = RandomForestClassifier()
model.fit(X_train, y_train)

classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)
test_pred = classifier.predict(X_test)


print("Score for the training data:", accuracy_score(train_pred, y_train))
print("Score for the testing data:", accuracy_score(test_pred, y_test))
