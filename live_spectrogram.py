import numpy as np
import matplotlib
from scipy.signal import lfilter, lfilter_zi, firwin
from time import sleep
from pylsl import StreamInlet, resolve_byprop
import seaborn as sns
from scipy.signal import spectrogram
import sys
import pandas as pd
import time
from threading import Thread
import datetime
import os

VIEW_BUFFER = 12
VIEW_SUBSAMPLE = 2
LSL_SCAN_TIMEOUT = 5
LSL_EEG_CHUNK = 12

class FileInlet:
    def __init__(self, fname):
        self.data = pd.read_csv(fname)
        self.last_time = self.data["timestamps"][0]
        self.delta_time = time.time() - self.last_time

    def sfreq(self):
        return int(1024 / (self.data.loc[1024, "timestamps"] - self.data.loc[0, "timestamps"]))

    def ch_names(self):
        return self.data.columns[1:]
    
    def pull_chunk(self, timeout, max_samples):
        start = time.time()
        t = start - self.delta_time
        # pull all idxs which are between last_time and t
        idxs = (self.last_time < self.data["timestamps"]) & (self.data["timestamps"] <= t)
        while not any(idxs):
            if time.time() - start > timeout:
                return None, None
            #sleep one period and we will have one event, should use timeout but...
            time.sleep(1 / self.sfreq())
            t = time.time() - self.delta_time
            idxs = (self.last_time < self.data["timestamps"]) & (self.data["timestamps"] <= t)
        if np.sum(idxs) > max_samples:
            # take only last max_samples of them
            pass
        self.last_time = t
        times = self.data.loc[idxs, "timestamps"].to_numpy()
        samples = self.data.loc[idxs, self.data.columns != "timestamps"].to_numpy()
        return samples, times

class LSLViewer():
    def __init__(self, stream, fig, tsax, axes, window, scale, dejitter=True):
        """Init"""
        self.stream = stream
        self.window = window
        self.scale = scale
        self.dejitter = dejitter
        self.filt = True
        self.subsample = VIEW_SUBSAMPLE

        if isinstance(stream, str):
            self.inlet = FileInlet(stream)
            self.sfreq = self.inlet.sfreq()
            self.ch_names = self.inlet.ch_names()
            self.n_chan = len(self.ch_names)
        else:
            self.inlet = StreamInlet(stream, max_chunklen=LSL_EEG_CHUNK)

            info = self.inlet.info()
            description = info.desc()

            self.sfreq = info.nominal_srate()
            self.n_chan = info.channel_count()

            ch = description.child('channels').first_child()
            ch_names = [ch.child_value('label')]

            for i in range(self.n_chan):
                ch = ch.next_sibling()
                ch_names.append(ch.child_value('label'))

            self.ch_names = ch_names
        
        # This is to not show last channel (if removed we also need to add one more plot for spectrogram)
        self.ch_names = self.ch_names

        self.n_samples = int(self.sfreq * self.window)

        fig.canvas.mpl_connect('key_press_event', self.OnKeypress)
        fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.fig = fig
        self.tsaxes = tsax
        self.axes = axes

        sns.despine(left=True)

        self.data = np.zeros((self.n_samples, self.n_chan))
        self.times = np.arange(-self.window, 0, 1. / self.sfreq)

        impedances = np.std(self.data, axis=0)
        lines = []

        for ii in range(self.n_chan):
            line, = self.tsaxes.plot(self.times[::self.subsample],
                self.data[::self.subsample, ii] - ii, lw=1)
            lines.append(line)
        self.lines = lines

        self.tsaxes.set_ylim(-self.n_chan + 0.5, 0.5)
        ticks = np.arange(0, -self.n_chan, -1)

        self.tsaxes.set_xlabel('Time (s)')
        self.tsaxes.xaxis.grid(False)
        self.tsaxes.set_yticks(ticks)

        ticks_labels = ['%s - %.1f' % (self.ch_names[ii], impedances[ii])
                        for ii in range(self.n_chan)]
        self.tsaxes.set_yticklabels(ticks_labels)

        self.im = []
        self.window_time = 2
        for i in range(self.n_chan):
            f, t, S = spectrogram(self.data[-self.window_time*self.sfreq:, i], fs=self.sfreq, nperseg=64, noverlap=48)
            self.im.append(self.axes[i].pcolormesh(t-self.window_time, f, S, shading='gouraud'))
            self.axes[i].set_title(self.ch_names[i])
            self.axes[i].set_xlabel("Time [sec]")
        self.axes[0].set_ylabel('Frequency [Hz]')

        #self.im = [self.specaxes[i].imshow(S, cmap=matplotlib.pyplot.cm.Reds) for i in range(4)]
        #, vmin=0, vmax=1, extent=[t[0], t[-1], f[0], f[-1]])
        print(self.im[0].get_array().shape)

        self.display_every = int(0.2 / (12 / self.sfreq))

        self.bf = firwin(32, np.array([1, 40]) / (self.sfreq / 2.), width=0.05,
                         pass_zero=False)
        self.af = [1.0]

        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (self.n_chan, 1)).transpose()
        self.data_f = np.zeros((self.n_samples, self.n_chan))

        self.save_data = False

    def update_plot(self):
        k = 0
        try:
            while self.started:
                samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                            max_samples=LSL_EEG_CHUNK)
                
                if timestamps is not None and len(timestamps) > 0:
                    if self.save_data:
                        first = not os.path.isfile(self.save_file)
                        with open(self.save_file, "a+") as f:
                            if first:
                                f.write("timestamps,TP9,AF7,AF8,TP10\n")
                            for i in range(len(timestamps)):
                                f.write("{},{},{},{},{}\n".format(timestamps[i], *samples[i, :]))

                    if self.dejitter:
                        timestamps = np.arange(len(timestamps), dtype=np.float64)
                        timestamps /= self.sfreq
                        timestamps += self.times[-1] + 1. / self.sfreq
                    self.times = np.concatenate([self.times, timestamps])
                    self.n_samples = int(self.sfreq * self.window)
                    self.times = self.times[-self.n_samples:]
                    self.data = np.vstack([self.data, samples])
                    self.data = self.data[-self.n_samples:]
                    filt_samples, self.filt_state = lfilter(
                        self.bf, self.af,
                        samples,
                        axis=0, zi=self.filt_state)
                    self.data_f = np.vstack([self.data_f, filt_samples])
                    self.data_f = self.data_f[-self.n_samples:]
                    k += 1
                    if k == self.display_every:
                        if self.filt:
                            plot_data = self.data_f
                        elif not self.filt:
                            plot_data = self.data - self.data.mean(axis=0)
                        for ii in range(self.n_chan):
                            self.lines[ii].set_xdata(self.times[::self.subsample] -
                                                     self.times[-1])
                            self.lines[ii].set_ydata(plot_data[::self.subsample, ii] /
                                                     self.scale - ii)
                            impedances = np.std(plot_data, axis=0)

                        ticks_labels = ['%s - %.2f' % (self.ch_names[ii],
                                                       impedances[ii])
                                        for ii in range(self.n_chan)]
                        self.tsaxes.set_yticklabels(ticks_labels)
                        self.tsaxes.set_xlim(-self.window, 0)

                        for i in range(4):
                            _, _, S = spectrogram(plot_data[-self.window_time*self.sfreq:, i], fs=1/self.sfreq, nperseg=64, noverlap=48)
                            self.im[i].set_array(S)
                            self.im[i].autoscale()

                        self.fig.canvas.draw()
                        k = 0
                else:
                    sleep(0.2)
        except RuntimeError as e:
            print(e)
            raise

    def onclick(self, event):
        print((event.button, event.x, event.y, event.xdata, event.ydata))

    def OnKeypress(self, event):
        if event.key == '/':
            self.scale *= 1.2
        elif event.key == '*':
            self.scale /= 1.2
        elif event.key == '+':
            self.window += 1
        elif event.key == '-':
            if self.window > 1:
                self.window -= 1
        elif event.key == 'd':
            self.filt = not(self.filt)
        elif event.key == 'r':
            self.save_file = datetime.datetime.now().strftime("EEG_recording_%Y-%m-%d-%H.%M.%S.csv")
            self.save_data = not self.save_data
            if self.save_data:
                print("Recording...")
            else:
                print("Stopped recording...")

    def start(self):
        self.started = True
        self.thread = Thread(target=self.update_plot)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, close_event):
        self.started = False



def view(window, scale, refresh, version=1):
    matplotlib.use("Qt5Agg")
    sns.set(style="whitegrid")

    fig = matplotlib.pyplot.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    tsax = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = fig.add_subplot(gs[1, 3])
    axs = [ax1, ax2, ax3, ax4]

    # Process input, if file sent in ...
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        streams = [fname]
        # create fake datastream from file in sys.argv[1]
    else:
        print("Looking for an EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

        if len(streams) == 0:
            raise(RuntimeError("Can't find EEG stream."))
        print("Start acquiring data.")

    lslv = LSLViewer(streams[0], fig, tsax, axs, window, scale)
    fig.canvas.mpl_connect('close_event', lslv.stop)

    help_str = """
                toggle filter : d
                toogle full screen : f
                toggle save data: r
                zoom out : /
                zoom in : *
                increase time scale : -
                decrease time scale : +
               """
    print(help_str)
    lslv.start()
    matplotlib.pyplot.show()

if __name__ == "__main__":
    view(5, 100, 0.2)
