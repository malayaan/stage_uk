from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("Loading the data...")

# Charger les données
df_test = pd.read_csv(r"data\feelings\LSTM_random_embedding\anger_mean_test.csv", encoding='utf-8')
y_test = df_test['anger_index'].astype(float)
X_test = df_test.drop('anger_index', axis=1).values

# Charger les données
df_train = pd.read_csv(r"data\feelings\LSTM_random_embedding\anger_mean_train.csv", encoding='utf-8')
y_train = df_train['anger_index'].astype(float)
X_train = df_train.drop('anger_index', axis=1).values

# Concatenate the test and train datasets
X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

# Charger les données
df_validation = pd.read_csv(r"data\hate_speech\LSTM_random_embedding\mean_no_emiji.csv", encoding='utf-8')
X_validation = df_validation.drop('class', axis=1).values

print("Data loaded successfully. Building the model...")

# Create a regression model with SVM
svm = SVR(C=1)

print("Training the model...")

# Fit the model to the data
svm.fit(X_train, y_train)

print("Model trained successfully.")

print("Making predictions on validation data...")

# Predicting the values for X_validation
y_validation_pred = svm.predict(X_validation)

print("Loading the data...")

# Charger les données
df_test = pd.read_csv(r"data\feelings\LSTM_random_embedding\anger_mean_test.csv", encoding='utf-8')
y_test = df_test['anger_index'].astype(float)
X_test = df_test.drop('anger_index', axis=1).values

# Charger les données
df_train = pd.read_csv(r"data\feelings\LSTM_random_embedding\anger_mean_train.csv", encoding='utf-8')
y_train = df_train['anger_index'].astype(float)
X_train = df_train.drop('anger_index', axis=1).values

# Concatenate the test and train datasets
X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

print("Data loaded successfully. Building the model...")

# Create a regression model with SVM
svm = SVR(C=1)

print("Training the model...")

# Fit the model to the data
svm.fit(X_train, y_train)

print("Model trained successfully.")

print("Making predictions on validation data...")

# Predicting the values for X_validation
y_validation_pred = svm.predict(X_validation)






# Creating a DataFrame with predictions and classes
df_pred = pd.DataFrame(data={'Predicted_values': y_validation_pred, 'Class': df_validation['class']})

print("Predictions made successfully. Saving to CSV...")

# Saving predictions to a new CSV file
df_pred.to_csv('predictions.csv', index=False)

print("Predictions saved successfully in 'predictions.csv'.")