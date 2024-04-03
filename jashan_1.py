# importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from sklearn.preprocessing import MinMaxScaler

# importing the data
columns = ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]
dataset = pd.read_csv("Glass Identification.csv", names = columns, index_col = "Id number")
dataset.head()

dataset.info()

y = dataset.pop("Type of glass").copy()
x = dataset.copy()

# shape of X and Y dataset
print(f"Shape of the X dataset : {x.shape}")
print(f"Shape of the Y dataset : {y.shape}")

# scaling the training data
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(x)

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)

# Shapes
print(f"Shape of x_train : {x_train.shape}")
print(f"Shape of y_train : {y_train.shape}")
print(f"Shape of x_test : {x_test.shape}")
print(f"Shape of y_test : {y_test.shape}")

# making target values into one-hot encoded format
y_train_adjusted = y_train - 1
num_classes = 7
y_train_encoded = to_categorical(y_train_adjusted, num_classes=num_classes)

y_test_adjusted = y_test - 1
num_classes = 7
y_test_encoded = to_categorical(y_test_adjusted, num_classes=num_classes)

# Initialising the model
model = Sequential()

# Input Layer
model.add(Dense(units = 1024, activation = "relu", input_shape = (9, )))
model.add(Dropout(rate = 0.1))

# Hidden Layers
hidden_layers = [512, 512, 512, 128, 64]

for layer_dim in hidden_layers:
  model.add(Dense(units = layer_dim))
  model.add(Dropout(rate = 0.1))

# Output Layer
model.add(Dense(units = 7, activation = "softmax"))

model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.002,
                                                   beta_1 = 0.9,
                                                   beta_2 = 0.998),
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

# training the data
history = model.fit(x = x_train,
                    y = y_train_encoded,
                    epochs = 150,
                    validation_data = (x_test, y_test_encoded))

plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accurcay")
plt.legend()
plt.show()