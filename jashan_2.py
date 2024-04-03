# importing all the required packages
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv("Grades.csv", index_col = ["Seat No."])

# shape of the dataset before dropping null values
print(f"Shape of the dataset before dropping null values is {dataset.shape}")

# dropping all the null values
dataset.dropna(inplace=True)

# shape of the datatset after dropping all the null values
print(f"Shape of the dataset after null values is {dataset.shape}")

# dictionary to change all the string values to float values
# using smaller values so that the weights and bias values during the training don't get too big for a certain column
grades_to_numbers = {
    "A+": 0.1, "A": 0.2, "A-": 0.3,
    "B+": 0.4, "B": 0.5, "B-": 0.6,
    "C+": 0.7, "C": 0.8, "C-": 0.9,
    "D+": 1 , "D": 1.1, "F" : 1.2, "WU" : 1.3, "W" : 1.4
}

# seperating the labels and target values
x = dataset.drop("CGPA", axis=1)
y = dataset["CGPA"].values.reshape(-1, 1)

# changing the string values to float
x = x.applymap(lambda grade: grades_to_numbers.get(grade)).values

# splitting train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# defining the architecture of the model
model = Sequential([
    # Input Layer
    Dense(units=1024, activation="relu", input_shape=(x.shape[1], )),
    Dropout(rate=0.1),

    # Hidden Layers
    Dense(units=128, activation="relu"),
    Dropout(rate=0.1),
    Dense(units=128, activation="relu"),
    Dropout(rate=0.1),
    Dense(units=64, activation="relu"),
    Dropout(rate=0.1),

    # Output Layer
    Dense(units=1, activation="linear")  # Linear activation for regression
])

model.summary()

# training the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanAbsoluteError(),  # Instantiating MeanAbsoluteError
              metrics=["mae"])

# Model training
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=30,
                    validation_data=(x_test, y_test))

# visualizing the Mean squared error for training and test data
plt.plot(history.history["val_mae"], label = "Val_mae")
plt.plot(history.history["mae"], label = "mae")
plt.legend()
plt.show()

