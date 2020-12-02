import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class realTimePlot:
    def __init__(self, title, nsamples=100):
        self.nsamples = nsamples
        self.data = np.array([])
        self.data_mean = np.array([])

        plt.ion()
        self.fig = plt.figure()
        self.fig.canvas.set_window_title(title)
        self.ax = self.fig.add_subplot(111)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.line1, = self.ax.plot(range(len(self.data)), self.data, color='red', linestyle='solid')
        self.line2, = self.ax.plot(range(len(self.data_mean)), self.data_mean, color='blue', linestyle='solid')


    def __call__(self, new_data):
        self.data = np.append(self.data, new_data)
        self.data_mean = np.append(self.data_mean, np.mean(self.data[-self.nsamples:]))
        self.plot()

    def plot(self):
        # Add new data:
        self.line1.set_ydata(self.data)
        self.line1.set_xdata(range(len(self.data)))
        self.line2.set_ydata(self.data_mean)
        self.line2.set_xdata(range(len(self.data_mean)))

        # Make sure all the data fits
        self.ax.relim()
        # Autoscale the plot
        self.ax.autoscale()
        # And draw
        self.fig.canvas.draw()
        # Delete all events
        self.fig.canvas.flush_events()
