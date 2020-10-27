import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random

TRAIN_PATH_1 = os.getcwd() + "/1010/music/train/" #0
TEST_PATH_1 = os.getcwd() + "/1010/music/test/"
TRAIN_PATH_2 = os.getcwd() + "/1010/counting/train/" #1
TEST_PATH_2 = os.getcwd() + "/1010/counting/test/"
GUESS_FREQ = 2 #/s

def get_data():
    train_1 = os.listdir(TRAIN_PATH_1)
    test_1 = os.listdir(TEST_PATH_1)
    train_2 = os.listdir(TRAIN_PATH_2)
    test_2 = os.listdir(TEST_PATH_2)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    s = int(256 / GUESS_FREQ)
    
    # train data, train labels
    for file1, file2 in zip(train_1, train_2):
        d1 = (np.genfromtxt(TRAIN_PATH_1 + file1, delimiter=",", skip_header=1))[:,1:]
        d2 = (np.genfromtxt(TRAIN_PATH_2 + file2, delimiter=",", skip_header=1))[:,1:]
        a1 = int(len(d1) / s) * s
        a2 = int(len(d2) / s) * s
        train_data.append(d1[:a1])
        train_data.append(d2[:a2])
        for i in range(int((len(d1) / s))):
            train_labels.append(0)
        for i in range(int((len(d2) / s))):
            train_labels.append(1)

    #print("shape:", np.array(train_data).shape)

    # test data, test labels
    for file1, file2 in zip(test_1, test_2):
        d1 = (np.genfromtxt(TEST_PATH_1 + file1, delimiter=",", skip_header=1))[:,1:]
        d2 = (np.genfromtxt(TEST_PATH_2 + file2, delimiter=",", skip_header=1))[:,1:]
        a1 = int(len(d1) / s) * s
        a2 = int(len(d2) / s) * s
        test_data.append(d1[:a1])
        test_data.append(d2[:a2])
        for i in range(int((len(d1) / s))):
            test_labels.append(1)
        for i in range(int((len(d2) / s))):
            test_labels.append(0)

    train_data = np.array(train_data).reshape(-1, 128, 4)
    test_data = np.array(test_data).reshape(-1, 128, 4)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    temp1 = list(zip(train_data, train_labels))
    random.shuffle(temp1)
    train_data, train_labels = zip(*temp1)

    temp1 = list(zip(test_data, test_labels))
    random.shuffle(temp1)
    test_data, test_labels = zip(*temp1)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    print(train_data.shape, test_data.shape, train_labels.shape)
    return train_data, test_data, train_labels, test_labels


def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(30, activation=tf.nn.relu))
    model.add(keras.layers.Dense(30, activation=tf.nn.relu))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, train_labels, epochs=3)

    return model


def save_model(model, name):
    directory = os.path.join(os.getcwd(), name)
    model.save(name)


train_data, test_data, train_labels, test_labels = get_data()

model = get_model()
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("test_accuracy:", test_acc)

save = input("save model? (y/n): ")
if save == "y":
    name = input("name: ")
    save_model(model, name)

