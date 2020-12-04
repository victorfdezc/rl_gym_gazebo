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
from gym_gazebo_envs.utils.q_convergence import Qconvergence
from gym_gazebo_envs.utils.real_time_plot import realTimePlot

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

    # Init Gym Monitor
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rl_turtlebot3')
    outdir = pkg_path + '/training_results_deepqlearn_reactive_path_planning'
    env = wrappers.Monitor(env, outdir, force=False, resume=True)

    # Loads parameters from the ROS param server. Parameters are stored in a 
    # .yaml file inside the /config directory. They are loaded at runtime by 
    # the launch file:
    lr = rospy.get_param("/turtlebot3_reactive_path_planning_v0/learning_rate")
    epsilon = rospy.get_param("/turtlebot3_reactive_path_planning_v0/epsilon")
    gamma = rospy.get_param("/turtlebot3_reactive_path_planning_v0/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3_reactive_path_planning_v0/epsilon_discount")
    min_epsilon = rospy.get_param("/turtlebot3_reactive_path_planning_v0/min_epsilon")
    nepisodes = rospy.get_param("/turtlebot3_reactive_path_planning_v0/nepisodes")
    angle_ranges = rospy.get_param("/turtlebot3_reactive_path_planning_v0/angle_ranges")
    max_distance = rospy.get_param("/turtlebot3_reactive_path_planning_v0/max_distance")
    max_distance_error = rospy.get_param("/turtlebot3_reactive_path_planning_v0/max_distance_error")
    load_model = rospy.get_param("/turtlebot3_reactive_path_planning_v0/load_model")
    hidden_layer_sizes = rospy.get_param("/turtlebot3_reactive_path_planning_v0/hidden_layer_sizes")
    copy_period = rospy.get_param("/turtlebot3_reactive_path_planning_v0/copy_period")
    min_experiences = rospy.get_param("/turtlebot3_reactive_path_planning_v0/min_experiences")
    max_experiences = rospy.get_param("/turtlebot3_reactive_path_planning_v0/max_experiences")
    batch_size = rospy.get_param("/turtlebot3_reactive_path_planning_v0/batch_size")

    # Train the standarizer:
    scaler_ex = np.array([env.observation_space.sample() for x in range(20000)])
    scaler = standarize(scaler_ex)

    # Input size
    input_size = len(env.observation_space.sample())
    # Output size
    output_size = env.action_space.n

    # Create main DQN and target DQN
    model = deep_qlearn.DQN(input_size, output_size, hidden_layer_sizes, epsilon=epsilon, lr=lr, 
                gamma=gamma, min_experiences=min_experiences, max_experiences=max_experiences, batch_size=batch_size, activation_function=tf.nn.leaky_relu)
    target_network = deep_qlearn.DQN(input_size, output_size, hidden_layer_sizes, epsilon=epsilon, lr=lr,
                        gamma=gamma, min_experiences=min_experiences, max_experiences=max_experiences, batch_size=batch_size, activation_function=tf.nn.leaky_relu)
    network_copy = deep_qlearn.DQN(input_size, output_size, hidden_layer_sizes, epsilon=epsilon, lr=lr,
                        gamma=gamma, min_experiences=min_experiences, max_experiences=max_experiences, batch_size=batch_size, activation_function=tf.nn.leaky_relu)

    # Create session and saver object
    session = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()

    if load_model:
      saver.restore(session, outdir+"/model.ckpt")
      rospy.logwarn("Model loaded!")
    else:
      rospy.logwarn("Initializing new model!")
      init = tf.compat.v1.global_variables_initializer()
      session.run(init)
      # Avoid changing the graph (it could imply OOM error)
      tf.compat.v1.get_default_graph().finalize()

    model.set_session(session)
    target_network.set_session(session)
    network_copy.set_session(session)

    # Instantiate QConvergence object:
    qconvergence = Qconvergence(env,model,tf_copy_model=network_copy, nstates=256, nsamples=50, plot_curve=True)
    error_plot = realTimePlot("Episode Error")

    start_time = time.time()
    total_steps = 0
    # Run the number of episodes specified
    for n in range(nepisodes):
        # if (x%10) == 0 or ((x-1)%10) == 0: env.render(mode="human")
        # else: env.render(mode="close")
        # if x > 1000:
        #     env.render(mode="sim")
        # else:
        #     env.render("close")

        # Epsilon decay
        if model.epsilon > min_epsilon:
            model.epsilon *= epsilon_discount

        if n%50==0: 
          # Save the variables to disk.
          save_path = saver.save(session, outdir+"/model.ckpt")

        if n%2==0 and n>10:
          # Check network convergence
          qconvergence()

        # Initialize the environment and get first state of the robot
        state = env.reset()
        # state = scaler(observation)

        done = False
        episode_reward = 0
        episode_steps = 0
        episode_error = 0
        while not done:
            # Choose an action based on the state
            action = model.chooseAction(state)
            next_state, reward, done, info = env.step(action)
            # Standarize the new observation:
            # next_state = scaler(next_observation)
            
            # Update experience buffer
            model.add_experience(state, action, reward, next_state, done)
            # And train the model
            error = model.learn(target_network)

            state = next_state

            episode_reward += reward
            episode_steps += 1
            episode_error += error
            total_steps += 1

            if total_steps % copy_period == 0:
                # rospy.loginfo("Target Network Updated!")
                target_network.copy_from(model)

        error_plot(episode_error/episode_steps)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo(("Episode: " + str(n + 1) + " - Reward: " + str(round(episode_reward,2)) + " - Steps: " + str(episode_steps) 
                        + " - Epsilon: " + str(round(model.epsilon, 2)) + " - Error: " + str(round(episode_error/episode_steps, 2)) + " - Time: %d:%02d:%02d" % (h, m, s)))

    # Once the training is finished, we close the environment
    env.close()
