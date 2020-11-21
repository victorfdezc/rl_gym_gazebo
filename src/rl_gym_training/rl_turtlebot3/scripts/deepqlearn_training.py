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
from gym_gazebo_envs.robotEnvs.turtlebot3Envs.tasksEnvs import turtlebot3_reactive_path_planning-v0

class standarize:
  '''
  Esta clase aplica a cada uno de los atributos de la instancia un proceso de
  estandarización para que la media sea 0 y la varianza sea 1.
  '''
  def __init__(self,data):
    self.n_attributes = np.shape(data)[1]
    self.mean = np.array([data[:,a].mean() for a in range(self.n_attributes)])
    self.std = np.array([data[:,a].std() for a in range(self.n_attributes)])

  def __call__(self,data):
    return np.array([(data[a]-self.mean[a])/self.std[a] for a in range(self.n_attributes)]).T


if __name__ == '__main__':
    # Start ROS node
    rospy.init_node('turtlebot3_obstacle_avoidance_qlearnRBF', anonymous=True)

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

    # Train the standarizer:
    scaler_ex1 = np.random.random((20000, len(angle_ranges)))*max_distance
    scaler_ex2 = np.random.random((20000, 1))*5 #TODO: max error distance to final pose could be 5 meters?
    scaler_ex3 = np.random.random((20000, 1))*2*np.pi - np.pi #TODO: max error in direction to final pose could be PI rad?
    import pdb; pdb.set_trace()
    scaler = standarize(scaler_ex)

    # Tamaño de entrada del modelo
    input_size = len(env.observation_space.sample())
    # Tamaño de salida del modelo
    output_size = env.action_space.n
    # Capas ocultas del modelo y número de neuronas
    hidden_layer_sizes = [400,400]
    copy_period = 50 # Tasa de actualización de target network
    # Creamos tanto el modelo principal como la target network:
    model = DQN(input_size, output_size, hidden_layer_sizes, epsilon=1.0, lr=1e-3, 
                gamma=0.99, min_experiences=100, max_experiences=7500,batch_size=32)
    target_network = DQN(input_size, output_size, hidden_layer_sizes, epsilon=1.0, lr=1e-3, 
                        gamma=0.99, min_experiences=100, max_experiences=7500, batch_size=32)

    init = tf.compat.v1.global_variables_initializer()
    # Creamos una sesión
    session = tf.compat.v1.InteractiveSession()
    # Inicializamos todas las variables de los grafos
    session.run(init)
    # Definimos la sesión a usar en ambos modelos
    model.set_session(session)
    target_network.set_session(session)

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
