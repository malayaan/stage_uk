import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Charger les données
df = pd.read_csv("data/LSTM_random_embedding/mean.csv", encoding='utf-8')
y = df['class'].astype(int)
X = df.drop('class', axis=1).values

# Convert classes to categorical
y = to_categorical(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Simple Neural Network model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=1)

# Predict the labels of your tweets
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Display classification statistics
print(classification_report(np.argmax(y_test, axis=-1), y_pred))
