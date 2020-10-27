import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random
import matplotlib.pyplot as plt

PATH_1 = os.getcwd() + "/data/focus/" #0
PATH_2 = os.getcwd() + "/data/music/" #1
GUESS_FREQ = 2 #/s
FFT = False

def get_data():
    files_1 = os.listdir(PATH_1)
    files_2 = os.listdir(PATH_2)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    s = int(256 / GUESS_FREQ)
    
    # train data, train labels
    for file1, file2 in zip(files_1, files_2):
        d1 = (np.genfromtxt(PATH_1 + file1, delimiter=",", skip_header=1))[:,1:]
        d2 = (np.genfromtxt(PATH_2 + file2, delimiter=",", skip_header=1))[:,1:]
        a1 = int(len(d1) / s) * s
        a2 = int(len(d2) / s) * s
        train_data.append(d1[:a1])
        train_data.append(d2[:a2])
        for i in range(int((len(d1) / s))):
            train_labels.append(0)
        for i in range(int((len(d2) / s))):
            train_labels.append(1)

    #print("shape:", np.array(train_data).shape)

    train_data = np.array(train_data).reshape(-1, 128, 4)
    train_labels = np.array(train_labels)
    if FFT:
        train_data = np.fft.fft(train_data)

    temp1 = list(zip(train_data, train_labels))
    np.random.shuffle(temp1)
    train_data, train_labels = zip(*temp1)
    train_data = np.array(train_data)

    """for i in range(len(train_data[0])):
        for j in range(len(train_data[i])):
            train_data[i][j] = train_data[i][j] / 50.0"""

    train_labels = np.array(train_labels)
    #print(train_data[10])

    return train_data, train_labels


def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(400, activation=tf.nn.relu))
    model.add(keras.layers.Dense(400, activation=tf.nn.relu))
    model.add(keras.layers.Dense(400, activation=tf.nn.relu))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def save_model(model, name):
    directory = os.path.join(os.getcwd(), name)
    model.save(name)


train_data, train_labels = get_data()

model = get_model()

history = model.fit(train_data, train_labels, epochs=50, validation_split=0.3)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.style.use('seaborn-deep')

class1 = model.predict(plock ut all class 1 data punkter) 
class2 = 
bins = np.linspace(0, 1, 30)

plt.hist([class1, class2], bins, label=['x', 'y'])
plt.legend(loc='upper right')
plt.show()

"""save = input("save model? (y/n): ")
if save == "y":
    name = input("name: ")
    save_model(model, name)"""

