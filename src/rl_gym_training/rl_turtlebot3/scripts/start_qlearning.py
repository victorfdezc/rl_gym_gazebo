#!/usr/bin/env python

import gym
import numpy
import time
from rl_algorithms import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from gym_gazebo_envs.robotEnvs.turtlebot3Envs.tasksEnvs import turtlebot3_obstacle_avoidance


if __name__ == '__main__':
    # TODO: check env variables are fine (total_episode_steps, episode_num, total_episode_reward...)

    rospy.init_node('turtlebot3_obstacle_avoidance_qlearn', anonymous=True, log_level=rospy.WARN) # TODO: log_level!!!!!

    # Create the Gym environment
    env = gym.make('TurtleBot3ObstacleAvoidance-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rl_turtlebot3') ############## modified
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot3/alpha")
    Epsilon = rospy.get_param("/turtlebot3/epsilon")
    Gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        # if (x%10) == 0 or ((x-1)%10) == 0: env.render(mode="human")
        # else: env.render(mode="close")
        if x > 1000:
            env.render(mode="sim")
        else:
            env.render("close")

        # rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logdebug("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # TODO: avoid printing too much information and check time we spend in learning steps
            # and env.step process. Or check only how much time is until we perform the next action.
            # Make the algorithm learn based on the results
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logdebug("NOT DONE")
                state = nextState
            else:
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logwarn(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + " - Steps: " + str(i) + " - Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
