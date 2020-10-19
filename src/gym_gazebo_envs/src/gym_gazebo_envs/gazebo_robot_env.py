#!/usr/bin/env python

import rospy
import gym
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from gym_gazebo_envs.msg import RLEpisodeInfo
import os
import subprocess
import time


'''
This is going to be the main class of this package. This class inherits from the main class of
OpenAI Gym, that is, the class Env. You can check this class in OpenAI gym Github:
    https://github.com/openai/gym/blob/master/gym/core.py
It's important to know how this class works in order to make a functional environment using Gazebo.
If we check the OpenAI Gym Env class, we will see that the main methods we must need to know are:
step, reset, render, close and seed. Also, we must know about the action_space, observation_space and
reward_range attributes.
So, because GazeboRobotEnv class inherits from gym.Env class, we must define all those methods and 
attributes to make a Gazebo environment.
Of course, this class will define a generalized environment for robots in Gazebo, so some of those methods
or attributes must be defined in another subclasses because they depend on the robot we use or even the task
to solve. This is why we need a set of classes with a hierarchical order: the main class is GazeboRobotEnv, the
next one is th class that specify the robot, and the last class is the one that specify the task to solve.
'''
class GazeboRobotEnv(gym.Env):

    def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION", gazebo_version="ros"):

        # Initialize Gazebo connection to control the Gazebo simulation
        # TODO: de este bloque de codigo revisar que necesito y que implica realmente inicializar la conexion con Gazebo.
        # Creo que el objeto del controlador no es necesario
        rospy.logdebug("START init GazeboRobotEnv")
        self.gazebo = GazeboConnection(start_init_physics_parameters,reset_world_or_sim)
        self.gazebo_version = gazebo_version
        self.seed()


        # Initialize episode variables:
        self.episode_num = 0
        self.total_episode_reward = 0.0
        self.total_episode_steps = 0
        self.episode_done = False
        # And create a topic publisher to publish episode info:
        self.episode_pub = rospy.Publisher('/episode_info', RLEpisodeInfo, queue_size=1)        

        rospy.logdebug("END init GazeboRobotEnv")

    #--------------------- gym.Env main Methods ---------------------#
    def step(self, action):
        '''
        This function is used to run one timestep of the environment's dynamics. To
        do that, we get an action, a_t to execute in the environment, so we will end up
        in the state s_(t+1) with a reward r_(t+1). This means that after executing a_t
        we will receive the observation corresponding to the state s_(t+1) and the reward
        of arriving to that state. Besides, we will receive a done flag that means if the
        episode has finished or not.

        This function accepts an action and returns a tuple (observation, reward, done, info)

        # TODO: add params and return like OpenAI

        - arguments: action
        - return: obs, reward, done, info
        '''

        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        rospy.logdebug("START STEP OpenAIROS")

        # Start the Gazebo simulation:
        self.gazebo.unpauseSim()
        # Execute the given action in the simulation (that is, run one timestep):
        self._execute_action(action)

        # Pause the simulation again meanwhile we process all the information result of
        # performing the previous action:
        self.gazebo.pauseSim()
        # Get the observation (from robot sensors, for example) of the new state we arrived
        # because of performing the previous action:
        obs = self._get_obs()
        # Check if the episode has finished:
        done = self._is_done(obs)
        # We compute the reward achieved because of arriving to a new state (this reward can also depend
        # on the fact that the episode has finished or not):
        reward = self._compute_reward(obs, done)
        # Add this new reward to the total reward of this epsiode:
        self.total_episode_reward += reward

        # Add a new step to the total episode steps:
        self.total_episode_steps += 1

        # This dictionary is used to debug. In this case we don't need it so we return an empty dict:
        info = {}

        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, info

    def reset(self):
        '''
        This method resets the environment to an initial state and returns an initial observation.
        '''

        # TODO: check this method, I think it can be simplified so we don't need to call so many functions.
        # TODO: aqui es donde deberia hacer lo de resetear los controladores a partir de esperar un tiempo a 
        # que el robot vuelva a un estado inicial.

        rospy.logdebug("Reseting GazeboRobotEnvironment")
        self._update_episode()
        self._init_env_variables()
        self._reset_sim()
        obs = self._get_obs()
        rospy.logdebug("END Reseting GazeboRobotEnvironment")
        return obs

    def render(self, mode='human'):
        # TODO: check the example of how to implement this method!!
        # 3 modes: human, sim, close

        # Check if "gzclient" process is running:
        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')

        if mode == "sim":
            if proccount > 0:
                # If "gzclient" is running we only make sure that the simulation is running as fast as possible:
                self.gazebo.setPhysicsParameters(update_rate=-1)
            else:
                if self.gazebo_version == "ros":
                    # Open "gzclient" in case it is not open
                    subprocess.Popen("rosrun gazebo_ros gzclient __name:=gzclient", shell = True) # TODO: code blocks here when calling this node
                else:
                    subprocess.Popen("gzclient", shell = True)
                time.sleep(1)
                self.gazebo.setPhysicsParameters(update_rate=-1)

        elif mode == "human":
            if proccount > 0:
                # If "gzclient" is running we only make sure that the simulation is running as fast as possible:
                self.gazebo.setPhysicsParameters(update_rate=1000.0)
            else:
                if self.gazebo_version == "ros":
                    # Open "gzclient" in case it is not open
                    subprocess.Popen("rosrun gazebo_ros gzclient __name:=gzclient", shell = True)
                else:
                    subprocess.Popen("gzclient", shell = True)
                time.sleep(1)
                self.gazebo.setPhysicsParameters(update_rate=1000.0)

        elif mode == "close":
            self.gazebo.setPhysicsParameters(update_rate=-1)
            if proccount > 0:
                # Kill all "gzclient" process
                os.popen("killall -9 gzclient")

    def close(self):
        '''
        This method is used to close the environment.
        
        Use it for closing GUIS and other systems that need closing.
        
        TODO: modificar este metodo para que cierre Gazebo, tanto GUI como simulacion
        '''
        rospy.logdebug("Closing GazeboRobotEnvironment")
        
        # Shutdown the ROS node:
        rospy.signal_shutdown("Closing GazeboRobotEnvironment")

    def seed(self, seed=None):
        '''
        TODO: mirar para que sirve este metodo y por que llama al metodo de gym...
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    #----------------------------------------------------------------#


    #--------------------- Extension Methods ------------------------#
    def _update_episode(self):
        """
        Publishes the total reward of the episode and
        increases the episode number by one.
        :return:

        TODO: creo que lo mejor es eliminar esta funcion de mierda y poner todo esto
        en el metodo reset ya que lo unico que hace es publicar por el topico y reinicializar
        las variables del episodio
        """
        rospy.logdebug("PUBLISHING REWARD...") # cambiado por logwarn
        self._publish_reward_topic(
                                    self.total_episode_reward,
                                    self.episode_num
                                    )
        rospy.logdebug("PUBLISHING REWARD...DONE="+str(self.total_episode_reward)+",EP="+str(self.episode_num)) # cambiado por logwarn

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:

        TODO: igual que la funcion de arriba, si eso eliminarla, o quizas esta si se pueda quedar porque
        tiene mas lineas de codigo mas concretas, pero la funcion de _update_episode es una mierda
        """
        reward_msg = RLEpisodeInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.episode_pub.publish(reward_msg)

    def _reset_sim(self):
        '''
        This method is used to reset a Gazebo simulation
        '''
        rospy.logdebug("RESET SIM START")

        # TODO: why we need to check that all the systems are fine before
        # resetting?
        self.gazebo.unpauseSim()
        self._check_all_systems_ready() #TODO: why checking systems here?
        # TODO: why we need set an init pose if reseting the simulation also do this?
        self._set_final_state() #TODO: recuerda que para poner un estado inicial debes despausar la simulacion y luego la vuelves a pausar (ya que poner un estado inicial sera equivalente a ejecutar una accion)

        # Pause the simulation
        self.gazebo.pauseSim()
        # Reset the Gazebo simulation
        self.gazebo.resetSim()
        # And unpause the simulation to check that all systems (like sensors) are fine
        # and publishing data
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()
        self._set_initial_state()
        # Finally, pause the simulation again:
        self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        
        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # Set to false Done, because its calculated asyncronously
        self.episode_done = False
        self.total_episode_reward = 0.0
        self.total_episode_steps = 0
        self.episode_num += 1


    def _check_all_systems_ready(self): # TODO: decir que esto se define en la clase del robot
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()


    # TODO:Decir que las proximas clases se definen en la clase de la task
    def _set_initial_state(self): # TODO: check description
        '''
        Set the robot in a initial state to reset the simulation. For example, in case
        of a mobile robot, the initial state could be a linear and angular speeds equal zero,
        or in case of a robot arm, a initial pose with speed and acceleration equal zero.
        This can also be seen as a soft way of resetting robot controllers.
        '''
        raise NotImplementedError()

    def _set_final_state(self): # TODO: change description
        '''
        Set the robot in a final state to reset the simulation. For example, in case
        of a mobile robot, the initial state could be a linear and angular speeds equal zero,
        or in case of a robot arm, a initial pose with speed and acceleration equal zero.
        This can also be seen as a soft way of resetting robot controllers.
        '''
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _execute_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos): # TODO: method not used... use it to set the initial pose of the robots or remove it
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()

    #------------------------------------------------------------------#