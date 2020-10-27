import pygame as p
import tensorflow as tf
from tensorflow import keras
from raw_live_data import LiveData
import numpy as np
import os

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SPEED = 1.5
win = p.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = p.time.Clock()



def updata_event():
    event = None
    for event in p.event.get():
        if event.type == p.QUIT:
            p.quit()
            quit()
        if event.type == p.MOUSEBUTTONDOWN or event.type == p.MOUSEBUTTONUP:
            event = event
    return event


def get_model(path):
    return tf.keras.models.load_model(path)


def predict(model, data):
    prediction = model.predict(data)
    print("prediction:", np.argmax(prediction))
    return np.argmax(prediction)


def move_keys(X):
    keys_pressed = p.key.get_pressed()
    x = 0
    if keys_pressed[p.K_a] or keys_pressed[p.K_LEFT]:
        x -= SPEED
    if keys_pressed[p.K_d] or keys_pressed[p.K_RIGHT]:
        x += SPEED
    if keys_pressed[p.K_SPACE]:
        x -= (X - int(SCREEN_WIDTH / 2))
    return x


def move_eeg(model, data):
    data = np.array(data).reshape(-1, 128, 4)
    p = predict(model, data)
    if p == 0:
        return -SPEED
    if p == 1:
        return SPEED


def main():
    x = int(SCREEN_WIDTH / 2)
    y = int(SCREEN_HEIGHT / 2)
    r = int(SCREEN_WIDTH * 0.1)

    model = get_model(os.getcwd() + "/models/total_test")
    live_data = LiveData()

    while True:
        updata_event()
        x += move_keys(x)
        if len(live_data.data) == 128:
            x += move_eeg(model, np.array([live_data.data]))

        win.fill((0, 0, 0))
        p.draw.circle(win, (255, 0, 0), (int(x), int(y)), r)
        p.display.update()
        clock.tick(60)

main()
