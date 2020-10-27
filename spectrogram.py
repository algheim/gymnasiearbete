from scipy import signal
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import random

path = os.getcwd()
files = os.listdir(os.getcwd() + "/data/relax")


def test():
    data = np.sin(np.linspace(0.0, 400, 1000))
    data += np.sin(np.linspace(0.0, 200, 1000))
    x = range(1000)
    #plt.plot(x, data)

    f, t, Sxx = signal.spectrogram(data, 40)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def main():
    """for i in range(10):
        print(random.randint(0, 9))"""

    print(files)
    #data = np.genfromtxt(os.getcwd() + "/data/concentrate/" + files[5], delimiter=",", skip_header=1)
    data = np.genfromtxt(os.getcwd() + "/data/relax/" + files[5], delimiter=",", skip_header=1)
    #data = np.genfromtxt("on3.csv", delimiter=",", skip_header=1)
    """plt.plot(data[:,0], data[:,1])
    plt.figure()
    plt.plot(data[:,0], data[:,2])
    plt.figure"""

    for i in range(1, 5):
        f, t, Sxx = signal.spectrogram(data[:, i], 256)
        plt.subplot(2, 2, i)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
    plt.show()

main()
