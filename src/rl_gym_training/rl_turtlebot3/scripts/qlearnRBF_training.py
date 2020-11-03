#!/usr/bin/env python

import gym
from gym import wrappers
import numpy as np
import time
import rospy
import rospkg
from rl_algorithms import qlearnRBF
from sklearn.kernel_approximation import RBFSampler
# Import the environment to register it
from gym_gazebo_envs.robotEnvs.turtlebot3Envs.tasksEnvs import turtlebot3_obstacle_avoidance_v1


if __name__ == '__main__':
    # Start ROS node
    rospy.init_node('turtlebot3_obstacle_avoidance_qlearnRBF', anonymous=True)

    # Create the Gym environment
    env = gym.make('TurtleBot3ObstacleAvoidance-v1')

    # Loads parameters from the ROS param server. Parameters are stored in a 
    # .yaml file inside the /config directory. They are loaded at runtime by 
    # the launch file:
    lr = rospy.get_param("/turtlebot3_qlearnRBF/learning_rate")
    epsilon = rospy.get_param("/turtlebot3_qlearnRBF/epsilon")
    gamma = rospy.get_param("/turtlebot3_qlearnRBF/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3_qlearnRBF/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3_qlearnRBF/nepisodes")

    rbf_samplers = [("rbf1", RBFSampler(gamma=0.05, n_components=500)),
                    ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
                    ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
                    ("rbf4", RBFSampler(gamma=0.1, n_components=500)),
                    ("rbf5", RBFSampler(gamma=5.0, n_components=500)),
                    ("rbf6", RBFSampler(gamma=2.0, n_components=500))]
    observation_examples =  np.array([env.observation_space.sample() for x in range(20000)])

    # Initialises Q-Learning
    qlearn = qlearnRBF.QLearnRBF(env=env, epsilon=epsilon, lr=lr, gamma=gamma, 
                rbf_samplers=rbf_samplers, observation_examples=observation_examples)

    # Init Gym Monitor
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rl_turtlebot3')
    outdir = pkg_path + '/training_results_qlearnRBF'
    env = wrappers.Monitor(env, outdir, force=True)

    start_time = time.time()
    # Run the number of episodes specified
    for n in range(nepisodes):
        # if (x%10) == 0 or ((x-1)%10) == 0: env.render(mode="human")
        # else: env.render(mode="close")
        # if x > 1000:
        #     env.render(mode="sim")
        # else:
        #     env.render("close")

        # Epsilon decay
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        state = env.reset()

        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            # Choose an action based on the state
            action = qlearn.chooseAction(state)
            # Execute the action in the environment
            next_state, reward, done, info = env.step(action)

            # Make the algorithm learn based on the results
            qlearn.learn(state, action, reward, next_state)
            state = next_state

            episode_reward += reward
            episode_steps += 1

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo(("Episode: " + str(n + 1) + " - Reward: " + str(episode_reward) + " - Steps: " + str(episode_steps) 
                        + " - Epsilon: " + str(round(qlearn.epsilon, 2)) + " - Time: %d:%02d:%02d" % (h, m, s)))

    # Once the training is finished, we close the environment
    env.close()
