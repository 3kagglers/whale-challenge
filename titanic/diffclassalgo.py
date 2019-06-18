"""File to test different algorithms on the problem."""

from warnings import simplefilter   # Warning filter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import preprocessor

# Ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

prepr = preprocessor.Preprocessor()
prepr.process_training_dataset('train.csv')
features, survived = prepr.get_train_datasets()
prepr.process_test_dataset('test.csv')
x_test = prepr.get_test_dataset()
df = pd.read_csv('gender_submission.csv')
y_test = df.iloc[:, 1]
# Get number of columns in training data
n_cols = features.shape[1]

# prepare models
models = []
models.append(('RF', RandomForestClassifier(
    n_estimators=100, random_state=10)))
models.append(('LR', LogisticRegression(solver='lbfgs', random_state=10)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='scale', random_state=10)))

# evaluate each model in turn
names = []
results = []
seed = 10
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=100, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, features, survived, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


def fit_predict_analyse(classifier):
    """
    Receives a classifier, calls fit method, predict, invokes a confusion
    matrix and plots accuracy from it.
    :clf: Any : Will have methods clf.fit(input,output) and
                output = clf.predict(input) called.
    """
    classifier.fit(features, survived)
    y_pred = classifier.predict(x_test)
    return confusion_matrix(y_test, y_pred)


print("LR")
params = [0.01, 0.1, 1, 10, 100]
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
for solver in solvers:
    for param in params:
        clf = LogisticRegression(solver=solver, C=param, random_state=10)
        cm = fit_predict_analyse(clf)
        print(
            solver,
            param,
            'Accuracy : {:.2f}'.format(np.trace(cm) / sum(sum(cm))))
# C=1.0, S = lbfgs

print("RFC")
params = [10, 50, 100, 200, 500]
for param in params:
    clf = RandomForestClassifier(n_estimators=param, random_state=10)
    cm = fit_predict_analyse(clf)
    print(param, 'Accuracy : {:.2f}'.format(np.trace(cm) / sum(sum(cm))))
# C=100

print("KNC")
params = [1, 5, 10, 20, 50, 100]
for param in params:
    clf = KNeighborsClassifier(n_neighbors=param)
    cm = fit_predict_analyse(clf)
    print(param, 'Accuracy : {:.2f}'.format(np.trace(cm) / sum(sum(cm))))
# C =10
