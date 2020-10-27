import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random


def get_data():
    train_concentrate = os.listdir(os.getcwd() + "/active/train")
    test_concentrate = os.listdir(os.getcwd() + "/active/test")
    train_relax = os.listdir(os.getcwd() + "/relax/train")
    test_relax = os.listdir(os.getcwd() + "/relax/test")

    train_data = []
    train_labels = []
    shortest = 1100
    for file1, file2 in zip(train_concentrate, train_relax):
        d1 = (np.genfromtxt(os.getcwd() + "/active/train/" + file1,
        delimiter=",", skip_header=1))[:,1:]
        d2 = (np.genfromtxt(os.getcwd() + "/relax/train/" + file2,
        delimiter=",", skip_header=1))[:,1:]
        a = int(shortest/128) * 128
        train_data.append(d1[:a])
        train_data.append(d2[:a])
        for i in range(8):
            train_labels.append(0)
        for i in range(8):
            train_labels.append(1)

    test_data = []
    test_labels = []
    for file1, file2 in zip(test_concentrate, test_relax):
        d1 = (np.genfromtxt(os.getcwd() + "/active/test/" + file1,
        delimiter=",", skip_header=1))[:,1:]
        d2 = (np.genfromtxt(os.getcwd() + "/relax/test/" + file2,
        delimiter=",", skip_header=1))[:,1:]
        a = int(shortest/128) * 128
        test_data.append(d1[:a])
        test_data.append(d2[:a])
        for i in range(8):
            test_labels.append(0)
        for i in range(8):
            test_labels.append(1)


    train_data = np.array(train_data).reshape(-1, 128, 4)
    test_data = np.array(test_data).reshape(-1, 128, 4)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    temp1 = list(zip(train_data, train_labels))
    random.shuffle(temp1)
    train_data, train_labels = zip(*temp1)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    print(train_data.shape, test_data.shape, train_labels.shape)
    return train_data, test_data, train_labels, test_labels


def get_model(training_data, training_labels):
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(keras.layers.Dense(100, activation=tf.nn.relu))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(training_data, training_labels, epochs=4)

    return model

def predict(model, data):
    prediction = model.predict(data)
    print("prediction:", np.argmax(prediction))

train_data, test_data, train_labels, test_labels = get_data()

model = get_model(train_data, train_labels)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("test_accuracy:", test_acc)
