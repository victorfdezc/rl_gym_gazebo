import gym
import rospy
import numpy as np
import copy
import threading
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Qconvergence:
    def __init__(self, env, model, tf_copy_model=None, nstates=64, nsamples=10):
        rospy.init_node('Qconvergence', anonymous=True)
        self.env = env
        self.model = model
        self.nstates = nstates
        self.nsamples = nsamples
        if tf_copy_model == None:
            self.old_model = copy.deepcopy(model)
            self.tf_model = False
        else:
            self.old_model = tf_copy_model
            self.tf_model = True

        self.Qdistances_mean = np.array([])
        self.Qdistances = np.array([])

        # plot_curve = rospy.get_param("/Qconvergence/plot_convergence_curve")
        plot_curve = True
        self.pub = rospy.Publisher("/Qdistance", Float32, queue_size=10)

        if plot_curve:
            self.updated_data = False
            plot_thread = threading.Thread(target=self.plot_convergence_curve)
            plot_thread.start()

    def __call__(self):
        states = [self.env.observation_space.sample() for x in range(self.nstates)]
        actions = [self.env.action_space.sample() for x in range(self.nstates)]
        Qpred = np.array([self.model.getQ(states[i])[0][actions[i]] for i in range(self.nstates)])
        Q_oldpred = np.array([self.old_model.getQ(states[i])[0][actions[i]] for i in range(self.nstates)])
        Qdistance = np.sum(np.square(Qpred-Q_oldpred))/self.nstates
        self.add_distance(Qdistance)

        self.updated_data = True
        self.pub.publish(self.Qdistances_mean[-1]) ## print also the mean

        if not self.tf_model:
            self.old_model = copy.deepcopy(self.model)
        else:
            self.old_model.copy_from(self.model)

    def add_distance(self, Qdistance):
        self.Qdistances = np.append(self.Qdistances, Qdistance)
        self.Qdistances_mean = np.append(self.Qdistances_mean, np.mean(self.Qdistances[-self.nsamples:]))

    def plot_convergence_curve(self):
        plt.ion()
        self.fig = plt.figure(1)
        self.fig.canvas.set_window_title("Q Convergence Curve")
        self.ax = self.fig.add_subplot(111)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.line1, = self.ax.plot(range(len(self.Qdistances)), self.Qdistances, color='red', linestyle='solid')
        self.line2, = self.ax.plot(range(len(self.Qdistances_mean)), self.Qdistances_mean, color='blue', linestyle='solid')
        while not rospy.is_shutdown():
            if self.updated_data:
                q = self.Qdistances
                q_mean = self.Qdistances_mean
                self.updated_data = False

                # Add new data:
                self.line1.set_ydata(q)
                self.line1.set_xdata(range(len(q)))
                self.line2.set_ydata(q_mean)
                self.line2.set_xdata(range(len(q_mean)))

                # Make sure all the data fits
                self.ax.relim()
                # Autoscale the plot
                self.ax.autoscale()
                # And draw
                self.fig.canvas.draw()
                # Delete all events
                self.fig.canvas.flush_events()
