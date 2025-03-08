import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#load processed data
X = np.load("../data/archive/processed/train_X.npy")
y = np.load("../data/archive/processed/train_y.npy")

#one-hot encode labels
y = to_categorical(y, num_classes=8)

#split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#define model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')  #8 classes for chords
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

#save trained model
model.save("../models/guitar_chord_recognition.h5")

print("Model training complete!")
