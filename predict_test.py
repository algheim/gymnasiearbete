import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

PATH = os.getcwd() + "/data/1014/focus/"
FILE_INDEX = 1
MODEL_PATH = os.getcwd() + "/models/1013_focus_music"
GUESS_FREQ = 2


def get_data():
    files = os.listdir(PATH)[FILE_INDEX]
    data = []
    s = int(256 / GUESS_FREQ)
    
    d1 = (np.genfromtxt(PATH + files, delimiter=",", skip_header=1))[:,1:]
    a1 = int(len(d1) / s) * s
    data.append(d1[:a1])

    data = np.array(data).reshape(-1, 128, 4)
    np.random.shuffle(data)

    return data


def predict(model, d):
    prediction = model.predict(d)
    #print("prediction:", np.argmax(prediction))
    return np.argmax(prediction)


model = tf.keras.models.load_model(MODEL_PATH)
data = get_data()
predictions = [0, 0]
for d in data:
    p = predict(model, d.reshape(-1, 128, 4))
    predictions[p] += 1

print("predictions:", predictions)