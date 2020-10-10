#!/usr/bin/env python

import rospy
import time
from gym_gazebo_envs.msg import RLEpisodeInfo
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

rospy.init_node('plotEpisodeInfoNode', anonymous=True)
r = rospy.Rate(1)

x = np.array([])
y = np.array([])

def episodeInfo_callback(data):
    global y, x
    y = np.append(y, data.episode_reward)
    #y=np.delete(y,0)
    x = np.append(x, data.episode_number)
    #x=np.delete(x,0)
    
rospy.Subscriber("/episode_info", RLEpisodeInfo, episodeInfo_callback)

# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

fig = plt.figure()
fig.canvas.set_window_title("Episode Rewards")
ax = fig.add_subplot(111)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

if __name__ == '__main__':
    while not rospy.is_shutdown():
        line1.set_ydata(y)
        line1.set_xdata(x)
        ax = plt.gca()  # get the current axes
        ax.relim()      # make sure all the data fits
        ax.autoscale()  # auto-scale
        fig.canvas.draw()
        fig.canvas.flush_events()
        r.sleep()
