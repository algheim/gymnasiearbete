import numpy as np
import matplotlib
from scipy.signal import lfilter, lfilter_zi, firwin
import time
from pylsl import StreamInlet, resolve_byprop
import seaborn as sns
from threading import Thread

LSL_SCAN_TIMEOUT = 5
LSL_EEG_CHUNK = 12


class LiveData:
    def __init__(self):
        streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)
        if len(streams) == 0:
            raise(RuntimeError("Can't find EEG stream."))

        print("Start acquiring data.")
        self.inlet = StreamInlet(streams[0], LSL_EEG_CHUNK)
        self.data = []
        self.started = False
        self.start()
        self.FFT = False

    def update_data(self):
        while self.started:
            data_part, timestamps = self.inlet.pull_chunk(timeout=1.0, max_samples=LSL_EEG_CHUNK)
            self.data += data_part
            if len(self.data) >= 128:
                self.data = self.data[-128:]

    def start(self):
        self.started = True
        self.thread = Thread(target=self.update_data)
        self.thread.start()

    def stop(self):
        self.started = False

if __name__ == "__main__":
    live_data = LiveData()

    for i in range(10):
        print(len(live_data.data))
        time.sleep(0.1)

