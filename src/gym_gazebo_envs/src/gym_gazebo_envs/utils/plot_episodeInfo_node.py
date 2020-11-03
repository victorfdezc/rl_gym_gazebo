#!/usr/bin/env python

import rospy
import time
from gym_gazebo_envs.msg import RLEpisodeInfo
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Intialize plot node:
rospy.init_node('plotEpisodeInfoNode', anonymous=True)
r = rospy.Rate(1)

# Global variables to store data
x = np.array([])
y_r = np.array([])
y_s = np.array([])
data_updated = False

def moving_average(data) : # TODO:change funtion description
    # TODO: check if we can do an online moving average update for efficiency reasons... Now we are recomputing
    # the whole moving average each time the function is called
    '''
    Estamos interesados en obtener la media de los retornos a lo largo de 100 episodios, ya que,
    segun la documentacion de OpenAI Gym el agente es juzgado segun el retorno obtenido a lo largo
    de los 100 episodios.
    '''
    N = len(data)
    moving_average = np.empty(N)
    for t in range(N):
        moving_average[t] = data[max(0, t-100):(t+1)].mean()
    return moving_average

def episodeInfo_callback(data):
    global y_r, y_s, x, data_updated
    # Get data from topic
    if not(data.episode_number == 0 and data.episode_reward == 0):
        y_r = np.append(y_r, data.episode_reward)
        y_s = np.append(y_s, data.total_episode_steps)
        x = np.append(x, data.episode_number)
        data_updated = True

# Suscribe to "/episode_info" topic:
rospy.Subscriber("/episode_info", RLEpisodeInfo, episodeInfo_callback)

# We need this to plot data:
plt.ion()

# Create two figures to plot the rewards and total steps:
fig1 = plt.figure(1)
fig1.canvas.set_window_title("Episode Rewards")
fig2 = plt.figure(2)
fig2.canvas.set_window_title("Episode Steps")
# Get the axes:
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
# x axis will be always a integer data (episode num):
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
# Make the first plot:
line1, = ax1.plot(x, y_r, 'r-') # Returns a tuple of line objects, thus the comma
line2, = ax1.plot(x, moving_average(y_r), 'b-') # Returns a tuple of line objects, thus the comma
line3, = ax2.plot(x, y_s, 'g-') # Returns a tuple of line objects, thus the comma
line4, = ax2.plot(x, moving_average(y_s), 'm-') # Returns a tuple of line objects, thus the comma #TODO:change color!!

if __name__ == '__main__':
    while not rospy.is_shutdown():
        if data_updated: 
            # TODO: make a function that updates all the graphs, so we can use that function in the callback to avoid running all time a loop. So that
            # function must have all the code below to update the graphs (maybe is not a good idea because another callback could arrive
            # meanwhile we are plotting...)

            # Save data to avoid problems if the callback executes meanwhile we are plotting
            rewards = y_r # TODO: change names to avoid confusion
            steps = y_s
            episodes = x
            # Add new data:
            line1.set_ydata(rewards)
            line1.set_xdata(episodes)
            line2.set_ydata(moving_average(rewards))
            line2.set_xdata(episodes)
            line3.set_ydata(steps)
            line3.set_xdata(episodes)
            line4.set_ydata(moving_average(steps))
            line4.set_xdata(episodes)
            
            data_updated = False

            # Make sure all the data fits
            ax1.relim()
            ax2.relim()
            # Autoscale the plot
            ax1.autoscale()
            ax2.autoscale()
            # And draw
            fig1.canvas.draw()
            fig2.canvas.draw()
            # Delete all events
            fig1.canvas.flush_events()
            fig2.canvas.flush_events()

            rospy.loginfo("Average Reward/Steps for the last 100 episodes: {:0.2f} / {:0.2f}".format(rewards[-100:].mean(),steps[-100:].mean()))

        r.sleep()