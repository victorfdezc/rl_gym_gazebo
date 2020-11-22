#!/usr/bin/env python

import gym
from gym import wrappers
import numpy as np
import time
import rospy
import rospkg
import tensorflow as tf
from rl_algorithms import deep_qlearn
# Import the environment to register it
from gym_gazebo_envs.robotEnvs.turtlebot3Envs.tasksEnvs import turtlebot3_reactive_path_planning_v0

class standarize:
  '''
  Standarize each one of the attributes to have zero mean and unit variance
  '''
  def __init__(self,data):
    self.n_attributes = np.shape(data)[1]
    self.mean = np.array([data[:,a].mean() for a in range(self.n_attributes)])
    self.std = np.array([data[:,a].std() for a in range(self.n_attributes)])

  def __call__(self,data):
    return np.array([(data[a]-self.mean[a])/self.std[a] for a in range(self.n_attributes)]).T


if __name__ == '__main__':
    # Start ROS node
    rospy.init_node('turtlebot3_reactive_path_planning_deepqlearn', anonymous=True)

    # Create the Gym environment
    env = gym.make('TurtleBot3ReactivePathPlanning-v0')

    # Loads parameters from the ROS param server. Parameters are stored in a 
    # .yaml file inside the /config directory. They are loaded at runtime by 
    # the launch file:
    lr = rospy.get_param("/turtlebot3_rpp_dql/learning_rate")
    epsilon = rospy.get_param("/turtlebot3_rpp_dql/epsilon")
    gamma = rospy.get_param("/turtlebot3_rpp_dql/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3_rpp_dql/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3_rpp_dql/nepisodes")
    angle_ranges = rospy.get_param("/turtlebot3_rpp_dql/angle_ranges")
    max_distance = rospy.get_param("/turtlebot3_rpp_dql/max_distance")
    max_distance_error = rospy.get_param("/turtlebot3_rpp_dql/max_distance_error")

    # Train the standarizer:
    scaler_ex1 = np.random.random((20000, len(angle_ranges)))*max_distance
    scaler_ex2 = np.random.random((20000, 1))*max_distance_error
    scaler_ex3 = np.random.random((20000, 1))*2*np.pi - np.pi
    scaler_ex = np.concatenate((scaler_ex1,scaler_ex2,scaler_ex3),axis=1)
    scaler = standarize(scaler_ex)

    # Input size
    input_size = len(env.observation_space.sample())
    # Output size
    output_size = env.action_space.n
    # TODO: put hidden layers, copy period...etc as parameters in .yaml
    # Hidden layers
    hidden_layer_sizes = [800,800,400]
    # Update rate target network
    copy_period = 100
    # Create main DQN and target DQN
    model = deep_qlearn.DQN(input_size, output_size, hidden_layer_sizes, epsilon=1.0, lr=1e-3, 
                gamma=0.99, min_experiences=100, max_experiences=5000,batch_size=16)
    target_network = deep_qlearn.DQN(input_size, output_size, hidden_layer_sizes, epsilon=1.0, lr=1e-3, 
                        gamma=0.99, min_experiences=100, max_experiences=5000, batch_size=16)

    init = tf.compat.v1.global_variables_initializer()
    session = tf.compat.v1.InteractiveSession()
    session.run(init)
    model.set_session(session)
    target_network.set_session(session)

    # Init Gym Monitor
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rl_turtlebot3')
    outdir = pkg_path + '/training_results_deepqlearn'
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
        if model.epsilon > 0.05:
            model.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = scaler(observation)

        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            # Choose an action based on the state
            action = model.chooseAction(state)
            next_observation, reward, done, info = env.step(action)
            # Standarize the new observation:
            next_state = scaler(next_observation)
            
            # Update experience buffer
            model.add_experience(state, action, reward, next_state, done)
            # And train the model
            model.learn(target_network)

            state = next_state

            episode_reward += reward
            episode_steps += 1

            if episode_steps % copy_period == 0:
                target_network.copy_from(model)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo(("Episode: " + str(n + 1) + " - Reward: " + str(episode_reward) + " - Steps: " + str(episode_steps) 
                        + " - Epsilon: " + str(round(model.epsilon, 2)) + " - Time: %d:%02d:%02d" % (h, m, s)))

    # Once the training is finished, we close the environment
    env.close()
