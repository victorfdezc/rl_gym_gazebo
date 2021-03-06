import gym
import rospy
import numpy as np
import copy
from gym_gazebo_envs.msg import Qdistance
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Qconvergence:
    '''
    This class is used to compute the convergence of a Q-Learning model.

    Constructor Arguments:
        env (object): OpenAI Gym environment object
        model (object): the model used to approximate the Q function. This model must have a
            getQ(state) method to get the Q estimate for all actions. If it is a TensorFlow model
            it must have a copy_from(other_model) to copy the parameters of the other_model to the
            current model.
        tf_copy_model (object): if a TF model is used, it is needed another instance of the model used
            in order to have a copy to compute the distance between the previous model (this copy) and
            the current model (the instance given in the previous argument). As mentioned, this model
            must have a copy_from(other_model) method in order to update the parameters of the copy with
            the current parameters of the trained model.
        nstates (int): number of points used to estimate the distance between functions
        nsamples (int): number of samples to use to compute the moving average
        plot_curve (bool): whether or not plot the convergence curve (useful for stopping criteria)
    '''
    def __init__(self, env, model, tf_copy_model=None, nstates=64, nsamples=10, plot_curve=True):
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

        self.first = True

        self.Qdistances_mean = np.array([])
        self.Qdistances = np.array([])

        self.pub = rospy.Publisher("/Qdistance", Qdistance, queue_size=10)
        self.msg = Qdistance()

        self.plot_curve = plot_curve
        if self.plot_curve:
            plt.ion()
            self.fig = plt.figure()
            self.fig.canvas.set_window_title("Q Convergence Curve")
            self.ax = self.fig.add_subplot(111)
            self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.line1, = self.ax.plot(range(len(self.Qdistances)), self.Qdistances, color='red', linestyle='solid')
            self.line2, = self.ax.plot(range(len(self.Qdistances_mean)), self.Qdistances_mean, color='blue', linestyle='solid')


    def __call__(self):
        if not self.first:
            states = [self.env.observation_space.sample() for x in range(self.nstates)]
            actions = [self.env.action_space.sample() for x in range(self.nstates)]
            Qpred = np.array([self.model.getQ(states[i])[0][actions[i]] for i in range(self.nstates)])
            Q_oldpred = np.array([self.old_model.getQ(states[i])[0][actions[i]] for i in range(self.nstates)])
            Qdistance = np.sum(np.abs(Qpred-Q_oldpred))/self.nstates
            self.add_distance(Qdistance)

            self.msg.current_Qdistance = Qdistance
            self.msg.average_Qdistance = self.Qdistances_mean[-1]
            self.pub.publish(self.msg)

            if self.plot_curve:
                self.plot_convergence_curve()
        else:
            self.first = False

        if not self.tf_model:
            self.old_model = copy.deepcopy(self.model)
        else:
            self.old_model.copy_from(self.model)

    def add_distance(self, Qdistance):
        self.Qdistances = np.append(self.Qdistances, Qdistance)
        self.Qdistances_mean = np.append(self.Qdistances_mean, np.mean(self.Qdistances[-self.nsamples:]))

    def plot_convergence_curve(self):
        # Add new data:
        self.line1.set_ydata(self.Qdistances)
        self.line1.set_xdata(range(len(self.Qdistances)))
        self.line2.set_ydata(self.Qdistances_mean)
        self.line2.set_xdata(range(len(self.Qdistances_mean)))

        # Make sure all the data fits
        self.ax.relim()
        # Autoscale the plot
        self.ax.autoscale()
        # And draw
        self.fig.canvas.draw()
        # Delete all events
        self.fig.canvas.flush_events()
