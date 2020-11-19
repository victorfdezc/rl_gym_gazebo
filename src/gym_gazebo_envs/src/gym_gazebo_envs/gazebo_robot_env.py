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
This is going to be the main class of this framework. This class inherits from the main class of
OpenAI Gym, that is, the class Env. You can check this class in OpenAI gym Github:
    https://github.com/openai/gym/blob/master/gym/core.py
It's important to know how this class works in order to make a functional environment using Gazebo.
If we check the OpenAI Gym Env class, we will see that the main methods we must need to know are:
step, reset, render, close and seed. Also, we must know about the action_space, observation_space and
reward_range attributes.
So, because GazeboRobotEnv class inherits from gym.Env class, we must implement all those methods to 
make a Gazebo environment.
Of course, this class will define a generalized environment for robots in Gazebo, so some of those methods
depend on other auxiliar methods that have been implemented in the next subclasses because they depend on 
the robot we use or even the task to solve. This is why we need a set of classes with a hierarchical order: 
the main class is GazeboRobotEnv, the next one is the class that specify the robot, and the last class is the 
one that specify the task to solve.
The Env attributes will also be defined in the next subclasses because they depend on the task to solve.
'''
class GazeboRobotEnv(gym.Env):

    def __init__(self, reset_world_or_sim="SIMULATION", gazebo_version="ros"):

        # Initialize Gazebo connection to control the Gazebo simulation
        self.gazebo = GazeboConnection(reset_world_or_sim)
        
        self.gazebo_version = gazebo_version
        self.seed()
        # TODO: make a _reset_sim here so the robot is in the initial state? And avoid making the sim reset in GazeboConnection.
        # However, in this initialization, the reset will not make anything because later, the turtlebot env checks the topics
        # by unpausing the simulation.

        # Initialize episode variables:
        self.episode_num = 0
        self.total_episode_reward = 0.0
        self.total_episode_steps = 0
        self.average_step_time = 0.0
        self.total_step_time = 0.0
        self.ts = None
        self.episode_done = False
        self.episode_success = False
        # And create a topic publisher to publish episode info:
        self.episode_pub = rospy.Publisher('/episode_info', RLEpisodeInfo, queue_size=10)

        self.metadata = {'render.modes': ['human', 'sim', 'close']}


    #--------------------- gym.Env main Methods ---------------------#
    def step(self, action):
        '''
        This function is used to run one timestep of the environment's dynamics. To
        do that, we get an action, a_t to execute in the environment, so we will end up
        in the state s_(t+1) with a reward r_(t+1). This means that after executing a_t
        we will read the observation corresponding to the state s_(t+1) and we will compute
        the reward of arriving to that state. We will also compute a done flag that means if the
        episode has finished or not.

        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        '''
        # Run a step in the simulation given the action:
        self._sim_step(action)

        # Get the observation (from robot sensors, for example) of the new state we arrived
        # because of performing the previous action:
        obs = self._get_obs()
        # Check if the episode has finished:
        done, success = self._is_done(obs)
        # We compute the reward achieved because of arriving to a new state (this reward can also depend
        # on the fact that the episode has finished or not):
        reward = self._compute_reward(obs, done, success)
        # Add this new reward to the total reward of this epsiode:
        self.total_episode_reward += reward

        # Add a new step to the total episode steps:
        self.total_episode_steps += 1
        # Compute total step time:
        if self.ts == None: 
            self.ts = time.time()
        else:
            t = time.time()
            self.total_step_time += (t-self.ts)
            self.ts = t           

        # This dictionary is used to debug. In this case we don't need it so we return an empty dict:
        info = {}

        return obs, reward, done, info

    def reset(self):
        '''
        This method resets the environment to an initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        '''
        try:
            self.average_step_time = self.total_step_time/(self.total_episode_steps-1)
        except:
            self.average_step_time = 0.0
        self._publish_episode_info_topic()
        self._update_env_variables()
        self._reset_sim()
        obs = self._get_obs()
        return obs

    def render(self, mode='human'):
        '''
        Method to render the simulation using the Gazebo client GUI. We have 3 possible modes:
            - human: render the simulation at real time speed.
            - sim: render the simulation as fast as possible.
            - close: no render, the simulation will run as fast as possible.
        Args:
            mode (str): the mode to render with
        '''
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
        It also closes the whole simulation, including gzclient and gzserver.
        '''        
        # Shutdown the ROS node:
        rospy.signal_shutdown("Closing GazeboRobotEnvironment")
        os.popen("killall -9 gzserver gzclient rosmaster roscore")

    def seed(self, seed=None):
        '''
        Sets the seed for this env's random number generator(s).

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    #----------------------------------------------------------------#


    #--------------------- Extension Methods ------------------------#
    def _sim_step(self, action):
        '''
        Run a simulation step to execute the given action in the environment.
        '''
        # Start the Gazebo simulation:
        self.gazebo.unpauseSim()

        # Execute the given action in the simulation (that is, run one timestep):
        self._execute_action(action)

        # Pause the simulation again meanwhile we process all the information result of
        # performing the previous action:
        self.gazebo.pauseSim()

    def _publish_episode_info_topic(self):
        '''
        Publish episode info to /episode_info topic.
        '''
        msg = RLEpisodeInfo()
        msg.episode_number = self.episode_num
        msg.episode_reward = self.total_episode_reward
        msg.total_episode_steps = self.total_episode_steps
        msg.average_step_time = self.average_step_time
        self.episode_pub.publish(msg)

    def _reset_sim(self):
        '''
        This method is used to reset a Gazebo simulation
        '''
        # Start the simulation
        self.gazebo.unpauseSim()
        # And set a final state to the robot
        self._set_final_state()
        # Pause the simulation
        self.gazebo.pauseSim()

        # Reset the Gazebo simulation
        self.gazebo.resetSim()

        # And unpause the simulation to set a initial state to the robot
        self.gazebo.unpauseSim()
        self._set_initial_state()
        # Then, check that all systems (like sensors) are fine and
        # publishing data
        self._check_all_systems_ready()
        # Finally, pause the simulation again
        self.gazebo.pauseSim()
        
        return True

    def _update_env_variables(self):
        '''
        Update some variables after resetting the environment.
        '''
        # Set to false Done, because its calculated asyncronously
        self.episode_done = False
        self.episode_success = False
        self.total_episode_reward = 0.0
        self.total_episode_steps = 0
        self.total_step_time = 0.0
        self.ts = None
        self.episode_num += 1

    #----------------------------------------------------#
    #-- To implement in the specific robot environment --#
    #----------------------------------------------------#
    def _check_all_systems_ready(self):
        '''
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        '''
        raise NotImplementedError()

    #------------------------------------------#
    #-- To implement in the task environment --#
    #------------------------------------------#
    def _set_initial_state(self):
        '''
        Set the robot in a initial state after resetting the simulation. For example, in case
        of a mobile robot, the initial state could be a linear and angular speeds equal zero,
        or in case of a robot arm, a initial pose with speed and acceleration equal zero.
        This can also be seen as a soft way of resetting robot controllers.
        Besides, we can use this method to set a random initial position to the robot which can be
        really useful for mobile robots.
        '''
        raise NotImplementedError()

    def _set_final_state(self):
        '''
        Set the robot in a final state before resetting the simulation. For example, in case
        of a mobile robot, the final state could be a linear and angular speeds equal zero,
        or in case of a robot arm, a final pose with speed and acceleration equal zero.
        This can also be seen as a soft way of resetting robot controllers.
        '''
        raise NotImplementedError()

    def _get_obs(self):
        '''
        Returns the observation, for example, from the sensor readings.
        '''
        raise NotImplementedError()

    def _execute_action(self, action):
        '''
        Execute the given action to the simulation.

        Args:
            action (object) : action to execute
        '''
        raise NotImplementedError()

    def _is_done(self, observations):
        '''
        Indicates whether or not the episode is done. The episode can finish in case
        the robot fails reaching its goal (e.g. the robot has crashed) or in case it 
        has reached the goal (e.g. final pose).

        Args:
            observations (object): current observation of the robot

        Returns:
            done (bool): if the episode has finished or not
            success (bool): set True if the robot has reached its goal
        '''
        raise NotImplementedError()

    def _compute_reward(self, observations, done, success):
        '''
        Calculates the reward to give to the robot.

        Args:
            observations (object): current observation of the robot
            done (bool): if the episode has finished or not
            success (bool): True when the robot has reached its goal

        Returns:
            reward (float): reward to give
        '''
        raise NotImplementedError()
    #------------------------------------------------------------------#