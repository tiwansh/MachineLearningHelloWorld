import pandas as pd
import numpy as np

import sklearn

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 


dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

dataset = pd.read_csv(dataset_url, sep=';')

#print dataset.head()
#print dataset

Y = dataset.quality
X = dataset.drop('quality', 1)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 7)

#print len(X_train)
#print len(Y_test)


val = model_selection.KFold(n_splits = 100, random_state = 7)
result = model_selection.cross_val_score(LogisticRegression(), X_train, Y_train, cv = val, scoring = 'accuracy')

#print result

lr= LogisticRegression()
lr.fit(X_train, Y_train)

#predictions = lr.predict(X_test)
#print "Using Logistic regression : "
#print (accuracy_score(Y_test, predictions))
#print classification_report(Y_test, predictions)

#Train the model
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
 

#these are uncontrolled parameters
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth' : [None, 5, 3, 1]}

#Grid search CV performs cross validation across entire grid of hyperparmeters
clf = GridSearchCV(pipeline, hyperparameters, cv= 10)

#to fit and tune model, train it bey
clf.fit(X_train, Y_train)

#clf best trainingmodel yaad rakhega and jis tarah se training model ko scale and normalise kiya tha same test par bhi karega


#now predcit using clf
pred = clf.predict(X_test)

print "Using model_selection "
print r2_score(Y_test, pred)
print mean_squared_error(Y_test, pred)



