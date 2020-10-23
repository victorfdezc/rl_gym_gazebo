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

def episodeInfo_callback(data):
    global y_r, y_s, x
    # Get data from topic
    y_r = np.append(y_r, data.episode_reward)
    y_s = np.append(y_s, data.total_episode_steps)
    x = np.append(x, data.episode_number)

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
line2, = ax2.plot(x, y_s, 'b-') # Returns a tuple of line objects, thus the comma

if __name__ == '__main__':
    while not rospy.is_shutdown():
        # Add new data:
        line1.set_ydata(y_r)
        line1.set_xdata(x)
        line2.set_ydata(y_s)
        line2.set_xdata(x)
        
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

        r.sleep()