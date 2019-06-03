# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:20:42 2019
@author: Kraken

Project: Titanic Kaggle
"""

import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

import preprocessor
import featureAnalysis

# Above this value, the survival flag will be true
PROBABILITY_MARGIN_SURVIVAL = 0.5

prepr = preprocessor.Preprocessor()
prepr.process_training_dataset('train.csv')

df = pd.read_csv('train.csv')

# perform feature analysis
numerical_features = ["Survived", "SibSp", "Parch", "Age", "Fare"]
feat_analysis = featureAnalysis.FeatureAnalysis()
feat_analysis.get_correlation_numericalvalues(df, numerical_features)
feat_analysis.analyse_categoricalvalues(df)

# removed cabin and name columns

prepr.process_test_dataset('test.csv')
dataframe = pd.read_csv('gender_submission.csv')
test_output = dataframe.iloc[:, 1].values

input_value, output = prepr.get_train_datasets()

# Get number of columns in training data
n_cols = input_value.shape[1]


def neural_network():
    """Generates a newral network and returns it"""
    model = Sequential()
    model.add(Dense(24, activation='relu',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer='l2',
                    bias_initializer='zeros',
                    input_shape=(n_cols,)))
    model.add(Dense(12, activation='relu',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer='l2',
                    bias_initializer='zeros'))
    model.add(Dense(1, kernel_initializer='glorot_uniform',
                    kernel_regularizer='l2',
                    bias_initializer='zeros',
                    activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


early_stopping_monitor = EarlyStopping(patience=5)
# Train model

net = neural_network()
history = net.fit(input_value,
                  output,
                  validation_split=0.2,
                  epochs=40,
                  verbose=0,
                  callbacks=[early_stopping_monitor])
test_loss = net.evaluate(prepr.get_test_dataset(), test_output)
print('Evaluate-Test loss:', test_loss)

pred_result = net.predict(prepr.get_test_dataset())

for i in range(len(pred_result)):
    pred_result[i] = 1 if pred_result[i] > PROBABILITY_MARGIN_SURVIVAL else 0

right = 0
for i in range(len(test_output)):
    if pred_result[i] == test_output[i]:
        right += 1
acc = right/len(test_output)
print('Calculated accuracy of: ' + str(acc))


fig = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend(['train', 'test'], loc='lower right')
# plt.show()

fig = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend(['train', 'test'], loc='upper right')
# plt.show()

# test_loss = model.evaluate(prepr.get_test_dataset(), test_output)
# print('Evaluate-Test loss:', test_loss)
#
# pred_result = model.predict(prepr.get_test_dataset())
# for i in range(len(pred_result)):
#     pred_result[i] = 1 if pred_result[i] > PROBABILITY_MARGIN_SURVIVAL else 0
#
# right = 0
# for i in range(len(test_output)):
#     if pred_result[i] == test_output[i]:
#         right += 1
# acc = right/len(test_output)
# print('Calculated accuracy of: ' + str(acc))

#
# #optimizers = ['adam', 'rmsprop']
# #init_wt = ['glorot_uniform', 'normal', 'uniform']
# epochs = [30, 60]
# batches = [8, 16, 32]
# early_stopping = [True]
# solver = ['lbfgs', 'adam', 'sgd']
# hidden_layers = [10, (3,3)]
# parameters = {
#     "solver": solver,
#     "max_iter": epochs,
#     "batch_size": batches,
#     "hidden_layer_sizes": hidden_layers,
#     "early_stopping": early_stopping}
#
# grid = GridSearchCV(estimator=MLPClassifier(),
#                    param_grid=parameters,
#                    scoring='accuracy',
#                    cv=10,
#                    n_jobs=-1)
#
# grid_results = grid.fit(input_value, output)
# best_accuracy = grid_results.best_score_
# best_parameters = grid_results.best_params_
# with open('./results','w') as fileh:
#     fileh.write(grid_results.cv_results_)
# print(grid_results.cv_results_)
# means = grid_results.cv_results_['mean_test_score']
# std = grid_results.cv_results_['std_test_score']
# params = grid_results.cv_results_['params']
# print(means)
# print(std)
# print(params)
# print(best_accuracy)
# print(best_parameters)
