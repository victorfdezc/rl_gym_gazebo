#!/usr/bin/env python

import gym
from gym import wrappers
import time
import rospy
import rospkg
from rl_algorithms import qlearn
# Import the environment to register it
from gym_gazebo_envs.robotEnvs.turtlebot3Envs.tasksEnvs import turtlebot3_obstacle_avoidance_v0


def build_state(features):
    '''
    Because we are using tabular Q-Learning, we must transform the real observations to a
    discrete state to build a state label for the Q-table
    '''
    return int("".join(map(lambda feature: str(int(feature)), features)))


if __name__ == '__main__':
    # Start ROS node
    rospy.init_node('turtlebot3_obstacle_avoidance_qlearn', anonymous=True)

    # Create the Gym environment
    env = gym.make('TurtleBot3ObstacleAvoidance-v0')

    # Loads parameters from the ROS param server. Parameters are stored in a 
    # .yaml file inside the /config directory. They are loaded at runtime by 
    # the launch file:
    lr = rospy.get_param("/turtlebot3_qlearn/learning_rate")
    epsilon = rospy.get_param("/turtlebot3_qlearn/epsilon")
    gamma = rospy.get_param("/turtlebot3_qlearn/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3_qlearn/epsilon_discount")
    min_epsilon = rospy.get_param("/turtlebot3_qlearn/min_epsilon")
    nepisodes = rospy.get_param("/turtlebot3_qlearn/nepisodes")

    # Initialises Q-Learning
    qlearn = qlearn.QLearn(env=env, epsilon=epsilon, lr=lr, gamma=gamma)

    # Init Gym Monitor
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rl_turtlebot3')
    outdir = pkg_path + '/training_results_qlearn'
    env = wrappers.Monitor(env, outdir, force=False, resume=True)

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
        if qlearn.epsilon > min_epsilon:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = build_state(observation)

        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            # Choose an action based on the state
            action = qlearn.chooseAction(state)
            # Execute the action in the environment
            observation, reward, done, info = env.step(action)
            next_state = build_state(observation)

            # Make the algorithm learn based on the results
            qlearn.learn(state, action, reward, next_state)
            state = next_state

            episode_reward += reward
            episode_steps += 1

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo(("Episode: " + str(n + 1) + " - Reward: " + str(round(episode_reward,2)) + " - Steps: " + str(episode_steps) 
                        + " - Epsilon: " + str(round(qlearn.epsilon, 2)) + " - Time: %d:%02d:%02d" % (h, m, s)))

    # Once the training is finished, we close the environment
    env.close()
