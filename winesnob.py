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


dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

dataset = pd.read_csv(dataset_url, sep=';')

print dataset.head()
print dataset

Y = dataset.quality
X = dataset.drop('quality', 1)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 7)

print len(X_train)
print len(Y_test)


#val = model_selection.KFold(n_splits = 100, random_state = 7)
#result = model_selection.cross_val_score(LogisticRegression(), X_train, Y_train, cv = val, scoring = 'accuracy')

#print result

lr= LogisticRegression()
lr.fit(X_train, Y_train)

predictions = lr.predict(X_test)
print (accuracy_score(Y_test, predictions))
print classification_report(Y_test, predictions)
