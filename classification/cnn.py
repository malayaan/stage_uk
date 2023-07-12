import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical

# Charger les données
df = pd.read_csv("data/LSTM_random_embedding/mean.csv", encoding='utf-8')
y = df['class'].astype(int)
X = df.drop('class', axis=1).values

# Reshape X to be suitable for CNN
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Convert classes to categorical
y = to_categorical(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=1)

# Predict the labels of your tweets
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Display classification statistics
print(classification_report(np.argmax(y_test, axis=-1), y_pred))
