#!/usr/bin/env python

import rospy
import time
from gym_gazebo_envs.msg import RLEpisodeInfo, episodeStatistics
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Intialize plot node:
rospy.init_node('episodeStatisticsNode', anonymous=True)
r = rospy.Rate(1)

try:
    plot = rospy.get_param("/episode_statistics/plot")
except:
    plot = True

# Global variables to store data
eps = np.array([])
rwd = np.array([])
nstp = np.array([])
tstp = np.array([])
ma_rwd = np.array([])
ma_nstp = np.array([])
ma_tstp = np.array([])
data_updated = False

def episodeInfo_callback(data):
    global eps, rwd, nstp, tstp, ma_rwd, ma_nstp, ma_tstp, data_updated
    # Get data from topic
    if not(data.episode_number == 0 and data.episode_reward == 0):
        eps = np.append(eps, data.episode_number)
        rwd = np.append(rwd, data.episode_reward)
        nstp = np.append(nstp, data.total_episode_steps)
        tstp = np.append(tstp, data.average_step_time)

        # Compute average taking into account the last 100 episodes
        # (standard way of knowing the performance of our agent in
        # OpenAI Gym)
        ma_rwd = np.append(ma_rwd, rwd[-100:].mean())
        ma_nstp = np.append(ma_nstp, nstp[-100:].mean())
        ma_tstp = np.append(ma_tstp, tstp[-100:].mean())

        data_updated = True

# Suscribe to "/episode_info" topic:
rospy.Subscriber("/episode_info", RLEpisodeInfo, episodeInfo_callback)
# Create a new publisher to publish moving average:
pub = rospy.Publisher("/episode_statistics", episodeStatistics, queue_size=10)
msg = episodeStatistics()

if plot:
    # We need this to plot data:
    plt.ion()

    # Create two figures to plot the rewards and total steps:
    fig1 = plt.figure(1)
    fig1.canvas.set_window_title("Episode Rewards")
    fig2 = plt.figure(2)
    fig2.canvas.set_window_title("Episode Steps")
    fig3 = plt.figure(3)
    fig3.canvas.set_window_title("Average Step Time (sec)")
    # Get the axes:
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    # x axis will be always a integer data (episode num):
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Make the first plot:
    line1, = ax1.plot(eps, rwd, color='red', linestyle='solid') # Returns a tuple of line objects, thus the comma
    line2, = ax1.plot(eps, ma_rwd, color='blue', linestyle='solid')
    line3, = ax2.plot(eps, nstp, color='lime', linestyle='solid')
    line4, = ax2.plot(eps, ma_nstp, color='darkviolet', linestyle='solid')
    line5, = ax3.plot(eps, tstp, color='darkorange', linestyle='solid')
    line6, = ax3.plot(eps, ma_tstp, color='darkgreen', linestyle='solid')

if __name__ == '__main__':
    while not rospy.is_shutdown():
        if data_updated: 
            # Save data to avoid problems if the callback executes meanwhile we are plotting
            rewards = rwd
            steps = nstp
            step_time = tstp
            ma_rewards = ma_rwd
            ma_nsteps = ma_nstp
            ma_tstep = ma_tstp
            std_rewards = rewards[-100:].std()
            std_nsteps = steps[-100:].std()
            std_tstep = step_time[-100:].std()
            episodes = eps

            data_updated = False

            rospy.loginfo("\nEpisode: {:d}\n    \\
                Average (last 100 episodes):\n     Rewards: {:0.2f}\n    Steps: {:0.2f}\n    Step Time (sec): {:0.6f}\n     \\
                Standard Deviation (last 100 episodes):\n       Rewards: {:0.2f}\n      Steps: {:0.2f}      Step Time: {:0.6f}\n\\
                ---------------".format(episodes[-1],ma_rewards[-1],ma_nsteps[-1],ma_tstep[-1], std_rewards, std_nsteps, std_tstep))
            msg.episode_number = episodes[-1]
            msg.episode_reward = ma_rewards[-1]
            msg.total_episode_steps = ma_nsteps[-1]
            msg.average_step_time = ma_tstep[-1]
            msg.std_reward = std_rewards
            msg.std_steps = std_nsteps
            msg.std_step_time = std_tstep
            pub.publish(msg)
            
            if plot:
                # Add new data:
                line1.set_ydata(rewards)
                line1.set_xdata(episodes)
                line2.set_ydata(ma_rewards)
                line2.set_xdata(episodes)

                line3.set_ydata(steps)
                line3.set_xdata(episodes)
                line4.set_ydata(ma_nsteps)
                line4.set_xdata(episodes)

                line5.set_ydata(step_time)
                line5.set_xdata(episodes)
                line6.set_ydata(ma_tstep)
                line6.set_xdata(episodes)

                # Make sure all the data fits
                ax1.relim()
                ax2.relim()
                ax3.relim()
                # Autoscale the plot
                ax1.autoscale()
                ax2.autoscale()
                ax3.autoscale()
                # And draw
                fig1.canvas.draw()
                fig2.canvas.draw()
                fig3.canvas.draw()
                # Delete all events
                fig1.canvas.flush_events()
                fig2.canvas.flush_events()
                fig3.canvas.flush_events()

        r.sleep()