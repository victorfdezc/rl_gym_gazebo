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
        id = 'TurtleBot3ObstacleAvoidance-v0',
        entry_point = 'gym_gazebo_envs.robotEnvs.turtlebot3Envs.tasksEnvs.turtlebot3_obstacle_avoidance:TurtleBot3ObstacleAvoidanceEnv',
        max_episode_steps = 1000, # TODO: put this parameter in yaml
    )

'''
This class is used to define a task to solve for a Turtlebot3 robot. In particular, we must define how
observations are taken, how to compute the reward, how to execute actions, when an episode has finished...
Besides, we must define the attributes needed to define a Gym environment: action_space, observation_space
and reward_range.

This class is defined to make the Turtlebot3 robot avoid obstacles in the world where it moves. To do that,
each time the robot crash, it will be penalized with a very high (negative) reward, but each step the robot moves 
without crashing, it will receive a small (positive) reward (these values can be changed in the yaml file #TODO). 
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
        self.number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.step_time = rospy.get_param('/turtlebot3/step_time')
        self.reset_time = rospy.get_param('/turtlebot3/reset_time')
        
        # Observation:
        self.angle_ranges = rospy.get_param("/turtlebot3/angle_ranges")
        self.distance_ranges = rospy.get_param("/turtlebot3/distance_ranges")
        self.min_range = rospy.get_param('/turtlebot3/min_range')

        # Rewards:
        self.forward_reward = rospy.get_param("/turtlebot3/forward_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")

        # Initial states:
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')
        self.initial_poses = rospy.get_param("/turtlebot3/initial_poses")


        # Now we are going to define the attributes needed to make a Gym environment.
        
        # First we define our action_space. In this case, the action_space is discrete 
        # and it has 3 possible values (it must be a Space object, in this case, 
        # a Discrete space object):
        self.action_space = spaces.Discrete(self.number_actions)
        
        # We set the reward range, that in this case can have any positive or negative value. This
        # must be a tuple:
        self.reward_range = (-numpy.inf, numpy.inf)

        # Finally we set the observation space which is a box (in this case it is bounded but it can be
        # unbounded). Specifically, a Box represents the Cartesian product of n closed intervals. Each 
        # interval has the form of one of [a, b], (-oo, b], [a, oo), or (-oo, oo). In this case we will
        # have 5 closed intervals of the form [0,2] because each interval can have 3 posible values #TODO: si cambiamos lo del yaml, cambiar tambien aqui la definicion del espacio de observaciones teniendo en cuenta los posibles valores
        num_laser_readings = 5 # Number of laser ranges
        high = numpy.full((num_laser_readings), 1) 
        low = numpy.full((num_laser_readings), 0)
        # Now we can create the Box space. Remember that the possible values for each range is an integer 
        # between 0 and 2, so it must be casted to int32:
        self.observation_space = spaces.Box(low, high, dtype=numpy.int32)


    def _set_initial_state(self):
        '''
        Set a initial state for the Turtlebot3. In our case, the initial state is
        a linear and angular speed equal to zero, so the controllers are 'resetted'.
        '''
        random_pose = self.initial_poses[random.randint(0,len(self.initial_poses)-1)]

        self.gazebo.setModelState("turtlebot3_burger", random_pose[0], random_pose[1], 0,0,0, random_pose[2],random_pose[3])
        self.move_base( self.init_linear_forward_speed, self.init_linear_turn_speed, wait_time=self.reset_time)

        return True

    def _set_final_state(self): # TODO: define this method in main RobotGazeboEnv class
        '''
        Set a initial state for the Turtlebot3. In our case, the initial state is
        a linear and angular speed equal to zero, so the controllers are 'resetted'.
        '''
        self.move_base( self.init_linear_forward_speed, self.init_linear_turn_speed, wait_time=self.reset_time)

        return True


    def _execute_action(self, action):
        '''
        This method is used to execute an action in the environment.

        In this case, based on the action number given, we will set the linear and angular
        speed of the Turtlebot3 base.
        '''
        rospy.logdebug("Start Set Action ==>"+str(action))

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
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        '''
        This method is used to get the observations of the environment.

        In this case, our observations will be computed with the LIDAR readings. In particular,
        we will discretize these readings in order to have the fewer states as possible.
        '''

        rospy.logdebug("Start Get Observation ==>")

        # We get the laser scan data
        laser_scan = self.laser_scan
        # And discretize them:
        discretized_observations = self._discretize_scan_observation(laser_scan)

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")

        return discretized_observations
        

    def _is_done(self, observations): # TODO: ten cuidado con los argumentos de cada funcion... recuerda que estas funciones estan ya definidas previamente por lo que no puedes quitar o poner argumentos como uno quiera
        '''
        This method is used to know if a episode has finished or not. It can be based on
        the observations given as argument.

        In this case, we will use the laser readings to check that. If any of the readings
        has a value less than a given distance, we suppose that the robot is really close of
        an obstacle, so the episode must finish.
        '''
        # Initialize the variable
        self.episode_done = False

        # Get the laser scan data
        laser_scan = self.laser_scan.ranges
        for i in laser_scan:
            if i<self.min_range:
                self.episode_done = True
                break

        # if self.episode_done:
        #     rospy.logerr("TurtleBot3 is Too Close to wall==>")
        # else:
        #     rospy.logwarn("TurtleBot3 is NOT close to a wall ==>")       

        return self.episode_done

    def _compute_reward(self, observations, done): # TODO: check GoalEnv and robotics envs in gym repo
        '''
        This method is used to compute the reward to give to the agent. It can be based on
        the observations or on the fact that the episode has finished or not (both of them given
        as arguments).

        In this case, the reward is based on the fact the agent has collided or not and on the last
        action taken.
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


    # Internal TaskEnv Methods
    
    def _discretize_scan_observation(self, data):
        '''
        Discretize the laser scan data. To do that, first we take only 180 readings (from 360 readings) that correspond
        to the range [-90,90] degrees (being 0 the front of the robot). Then we take those laser readings and we divide them
        into 5 sections (each one of 36 degrees). In this way the observation will be a list with 5 elements, and each element
        will have the value of the lowest value read from the corresponding 36 degrees used. Finally, each one of those measurements
        are again discretized, so instead of having a continuous distances, we will have discrete distances.
        '''
        # We get only the distance values
        laser_data = data.ranges
        discretized_ranges = []

        for r in self.angle_ranges:
            # From each section we get the lowest value
            min_value = min([laser_data[i] for i in range(r[0],r[1])])
            # And we discretize this value
            if min_value < self.distance_ranges[0]:
                discretized_ranges.append(0)
            # elif  min_value < self.distance_ranges[1]:
            #     discretized_ranges.append(1)
            else:
                discretized_ranges.append(1)

        # We reverse the list so the first element in the list (the most left element) correspond to the
        # most left angle range in the real robot: # TODO: explain this better
        self.discretized_ranges = discretized_ranges[::-1]

        rospy.logwarn("Discretized obs " + str(self.discretized_ranges))

        return self.discretized_ranges