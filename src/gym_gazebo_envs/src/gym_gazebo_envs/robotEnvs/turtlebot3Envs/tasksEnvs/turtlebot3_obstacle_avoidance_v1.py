#!/usr/bin/env python

import rospy
import numpy
import random
from gym import spaces
from gym_gazebo_envs.robotEnvs.turtlebot3Envs import turtlebot3_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3

'''
Register an environment by ID. IDs remain stable over time and are guaranteed to resolve 
to the same environment dynamics. The goal is that results on a particular environment
should always be comparable, and not depend on the version of the code that was running.
To register an environment, we have the following arguments:
        * id (str): The official environment ID
        * entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        * reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        * nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        * max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of (maximum number of executed actions)
        * kwargs (dict): The kwargs to pass to the environment class
'''
register(
        id = 'TurtleBot3ObstacleAvoidance-v1',
        entry_point = 'gym_gazebo_envs.robotEnvs.turtlebot3Envs.tasksEnvs.turtlebot3_obstacle_avoidance_v1:TurtleBot3ObstacleAvoidanceEnv',
        max_episode_steps = 1000
    )

'''
This class is used to define a task to solve for a Turtlebot3 robot. In particular, we must define how
observations are taken, how to compute the reward, how to execute actions, when an episode has finished...
that is all the GazeboRobotEnv methods that have not been implemented yet.
Besides, we must define the attributes needed to define a Gym environment: action_space, observation_space
and reward_range.

This class is defined to make the Turtlebot3 robot avoid obstacles in the world where it moves. To do that,
each time the robot crash, it will be penalized with a very high (negative) reward, but each step the robot moves 
without crashing, the reward will depend on the action taken, so for example, usually the robot will receive more 
reward if the previous action was to move forward, because in this way the robot will move much faster (these rewards 
can be changed in the yaml file #TODO). If you don't do that, the robot can realize that turning in one direction
always can lead to not crashing, so it would reach the maximum reward always without almost moving. 
For the observations we will use the laser data. This data will be discretized, so we will have 5 laser lectures 
corresponding to the following 5 laser angle ranges (remember that the angle 0 is the front of the robot): from 
-90 degrees to -54 degrees, from -54 to -18, from -18 to 18, from 18 to 54 and from 54 to 90. So each one of these 
5 laser ranges will have only one laser reading corresponding to the lowest reading obtained in that range. 
Finally, this reading (which is a distance in meters) will be discretized again, so that we can have 2 possible
values: 0 if the lecture is less than 0.5 meters and 1 if it is not (these values can be changed from the yaml file #TODO).
Finally, the robot can take only 3 possible actions: go forward, turn left or turn right. In this way we are making
a simple environment to train the robot with a simple Q-Learning algorithm.
'''
class TurtleBot3ObstacleAvoidanceEnv(turtlebot3_env.TurtleBot3Env):
    def __init__(self):
        # Call the __init__ function of the parent class:
        super(TurtleBot3ObstacleAvoidanceEnv, self).__init__()

        #TODO: add a description of each parameter we use!!
        # First we load all the parameters defined in the .yaml file.
        
        # Actions:
        self.linear_forward_speed = rospy.get_param('/turtlebot3_qlearnRBF/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3_qlearnRBF/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3_qlearnRBF/angular_speed')
        self.step_time = rospy.get_param('/turtlebot3_qlearnRBF/step_time')
        self.reset_time = rospy.get_param('/turtlebot3_qlearnRBF/reset_time')
        
        # Observation:
        self.max_distance = rospy.get_param("/turtlebot3_qlearnRBF/max_distance")
        self.angle_ranges = rospy.get_param("/turtlebot3_qlearnRBF/angle_ranges")
        self.min_range = rospy.get_param('/turtlebot3_qlearnRBF/min_range')

        # Rewards:
        self.forward_reward = rospy.get_param("/turtlebot3_qlearnRBF/forward_reward")
        self.turn_reward = rospy.get_param("/turtlebot3_qlearnRBF/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3_qlearnRBF/end_episode_points")

        # Initial states:
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3_qlearnRBF/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3_qlearnRBF/init_linear_turn_speed')
        self.initial_poses = rospy.get_param("/turtlebot3_qlearnRBF/initial_poses")


        # Now we are going to define the attributes needed to make a Gym environment.
        
        # First we define our action_space. In this case, the action_space is discrete 
        # and it has 3 possible values (it must be a Space object, in this case, 
        # a Discrete space object):
        self.action_space = spaces.Discrete(3)
        
        # We set the reward range, that in this case can have any positive or negative value. This
        # must be a tuple:
        self.reward_range = (-numpy.inf, numpy.inf)

        # Finally we set the observation space which is a box (in this case it is bounded but it can be
        # unbounded). Specifically, a Box represents the Cartesian product of n closed intervals. Each 
        # interval has the form of one of [a, b], (-oo, b], [a, oo), or (-oo, oo). In this case we will
        # have 5 closed intervals of the form [0,max_distance]
        num_laser_readings = 5 # Number of laser ranges
        high = numpy.full((num_laser_readings), self.max_distance) 
        low = numpy.full((num_laser_readings), 0.0)
        self.observation_space = spaces.Box(low, high, dtype=numpy.float32)


    #--------------------- GazeboRobotEnv Methods ---------------------#
    def _set_initial_state(self):
        '''
        Set a initial state for the Turtlebot3. In our case, the initial state is
        a linear and angular speed equal to zero, so the controllers are 'resetted'.
        We will also set a random initial pose.
        '''
        random_pose = self.initial_poses[random.randint(0,len(self.initial_poses)-1)]

        self.gazebo.setModelState("turtlebot3_burger", random_pose[0], random_pose[1], 0,0,0, random_pose[2],random_pose[3])
        self.move_base( self.init_linear_forward_speed, self.init_linear_turn_speed, wait_time=self.reset_time)

        return True

    def _set_final_state(self): # TODO: define this method in main RobotGazeboEnv class
        '''
        Set a final state for the Turtlebot3. In our case, the final state is also
        a linear and angular speed equal to zero.
        '''
        self.move_base( self.init_linear_forward_speed, self.init_linear_turn_speed, wait_time=self.reset_time)

        return True

    def _execute_action(self, action):
        '''
        This method is used to execute an action in the environment.

        In this case, based on the action number given, we will set the linear and angular
        speed of the Turtlebot3 base.
        '''        
        # We convert the actions numbers to linear and angular speeds:
        if action == 0: #Go forward
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            # We store the last action executed to compute the reward
            self.last_action = "forward"
        elif action == 1: #Turn Left
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "turn_left"
        elif action == 2: #Turn Right
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "turn_right"
        
        # We tell to TurtleBot3 the linear and angular speed to execute
        self.move_base(linear_speed, angular_speed, wait_time=self.step_time)
        
    def _get_obs(self):
        '''
        This method is used to get the observations of the environment.

        In this case, our observations will be computed with the LIDAR readings. In particular,
        we will discretize these readings in order to have the fewer states as possible (by decreasing
        the number of laser readings to 5, and by discretizing the continuous laser readings to have only 2
        possible values).
        '''
        # We get the laser scan data
        laser_scan = self.laser_scan
        # And discretize them:
        discretized_observations = self._discretize_scan_observation(laser_scan)

        return discretized_observations
        
    def _is_done(self, observations): # TODO: ten cuidado con los argumentos de cada funcion... recuerda que estas funciones estan ya definidas previamente por lo que no puedes quitar o poner argumentos como uno quiera
        '''
        This method is used to know if a episode has finished or not. It can be based on
        the observations given as argument.

        In this case, we will use the laser readings to check that. If any of the readings
        has a value less than a given distance, we suppose that the robot is really close of
        an obstacle, so the episode must finish.

        TODO: put args and returns
        '''
        # Initialize the variable
        self.episode_done = False

        # Get the laser scan data
        laser_scan = self.laser_scan.ranges
        min_value = min(laser_scan)
        if min_value<self.min_range:
            self.episode_done = True    

        return self.episode_done

    def _compute_reward(self, observations, done): # TODO: check GoalEnv and robotics envs in gym repo
        '''
        This method is used to compute the reward to give to the agent. It can be based on
        the observations or on the fact that the episode has finished or not (both of them given
        as arguments).

        In this case, the reward is based on the fact the agent has collided or not and on the last
        action taken.

        TODO: put args and returns
        '''   
        # The reward will depend on the fact that the episode has finished or not
        if not done:
            # In the case the episode has not finished, the reward will depend on the last action executed
            if self.last_action == "forward":
                reward = self.forward_reward
            else:
                reward = self.turn_reward
        else:
            reward = self.end_episode_points
        
        return reward
    #------------------------------------------------------------------#


    #------------------------ Auxiliar Methods ------------------------#    
    def _discretize_scan_observation(self, data):
        '''
        Discretize the laser scan data. To do that, first we take only 180 readings (from 360 readings) that correspond
        to the range [-90,90] degrees (being 0 the front of the robot). Then we take those laser readings and we divide them
        into 5 sections (each one of 36 degrees). In this way the observation will be a list with 5 elements, and each element
        will have a binary value depending on the minumum distance measured in its corresponding angle range (if the lowest 
        measurement in that range is less than some threshold, the value of the element would be 0, and it would be 1 if it is 
        greater). In this way we will have discrete distances making learning faster.
        '''
        # We get only the distance values
        laser_data = data.ranges
        discretized_ranges = []

        for r in self.angle_ranges:
            # From each section we get the lowest value
            min_value = min([laser_data[i] for i in range(r[0],r[1])])
            if min_value > self.max_distance:
                discretized_ranges.append(self.max_distance)
            else:
                discretized_ranges.append(min_value)

        # We reverse the list so the first element in the list (the most left element) correspond to the
        # most left angle range in the real robot: # TODO: explain this better
        self.discretized_ranges = discretized_ranges[::-1]

        # rospy.loginfo("Discretized obs " + str(self.discretized_ranges))

        return self.discretized_ranges
    #------------------------------------------------------------------#
