# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:20:42 2019
@author: Kraken

Project: Titanic Kaggle
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam

import preprocessor
import featureAnalysis

# Above this value, the survival flag will be true
PROBABILITY_MARGIN_SURVIVAL = 0.5

prepr = preprocessor.Preprocessor()
prepr.process_training_dataset('train.csv')

df = pd.read_csv('train.csv')

# perform feature analysis
numerical_features = ["Survived", "SibSp", "Parch", "Age", "Fare"]
#feat_analysis = featureAnalysis.FeatureAnalysis()
#feat_analysis.get_correlation_numericalvalues(df, numerical_features)
#feat_analysis.analyse_categoricalvalues(df)

# removed cabin and name columns
input_value, output = prepr.get_train_datasets()

# Get number of columns in training data
n_cols = input_value.shape[1]

def neural_network(l_rate):
    model = Sequential()
    model.add(Dense(256, activation='relu',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer='l2',
                    bias_initializer='zeros',
                    input_shape=(n_cols,)))
    model.add(Dense(256, activation='relu',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer='l2',
                    bias_initializer='zeros'))
    model.add(Dense(1, kernel_initializer='glorot_uniform',
                    kernel_regularizer='l2',
                    bias_initializer='zeros',
                    activation='sigmoid'))
    model.compile(optimizer=Adam(lr=l_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


early_stopping_monitor = EarlyStopping(patience=50)
# Train model
lr = 0.0001
net = neural_network(lr)
history = net.fit(input_value,
                  output,
                  validation_split=0.15,
                  epochs=200,
                  verbose=1,
                  callbacks=[early_stopping_monitor])

print('Loss: %.3f'%(history.history['loss'][-1]))
print('Accuracy: %.3f'%(history.history['acc'][-1]))
print(len(history.history['loss']))

# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(1, figsize=(12,7))
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# =============================================================================
# Testing
# =============================================================================
prepr.process_test_dataset('test.csv')
pred_result = net.predict(prepr.get_test_dataset())

for i in range(len(pred_result)):
    pred_result[i] = 1 if pred_result[i] > PROBABILITY_MARGIN_SURVIVAL else 0

# Generate file
file_data=[]    
for index, value in enumerate(pred_result):
    file_data.append((index+891, value[0]))

import csv
with open('data_submission.csv', 'w+', newline='') as d:
    writer = csv.DictWriter(d, fieldnames=["Passenger ID", "Survived"])
    writer.writeheader()
    writer = csv.writer(d, quoting=csv.QUOTE_NONNUMERIC)   
    writer.writerows(file_data)
    