#!/usr/bin/env python

import numpy
import rospy
import time
from gym_gazebo_envs import robot_gazebo_env
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


'''
This class is used for all Turtlebot3 robots (and even Turtlebot2 because they use the same topics
and services). As you can see, this class inherits from the parent class RobotGazeboEnv which specify
a generalized class for all robots simulated in Gazebo.

In this class we will define some of the main methods needed to take observations and actions.

Sensors: the sensors accesible are the ones considered useful for AI learning. 
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot.
        * /imu: Inertial Mesuring Unit that gives relative accelerations and orientations.
        * /scan: Laser readings from LIDAR.
        
Actuators: the actuators accesible are the ones that allow us to control the robot and take actions
        Actuators Topic List: 
        * /cmd_vel: control linear and angular velocities.

First of all, it's recommended to check how the Sensors topic react when we change the model state
(Turtlebot3 in this case) using Gazebo services. For example, when we reset the world using the 
/gazebo/reset_world service, everything work as expected: /odom updates with the new model position,
/imu updates with the new orientation and speeds, and /scan also updates the readings correctly. However,
if we reset the simulation using /gazebo/reset_simulation service, everything works correctly except the
/imu topic. In this case, the IMU breaks and stops publishing info.
So calling /gazebo/reset_simulation seems to break some Gazebo plugins, in this case the IMU. It seems that
some plugins fails with this call because /gazebo/reset_simulation also resets the simulation time. So if the
plugin doesn't have a proper "Reset" method, it can break (https://github.com/ros-simulation/gazebo_ros_pkgs/issues/169)
Therefore, with Turtlebot3 it is better to reset the world instead the whole simulation.
Finally, it has been tested /gazebo/set_model_state service in order to change the model pose. When we call it,
every sensor work as expected.

In conclusion, with Turtlebot3 we must reset the WORLD to avoid breaking the IMU and we can change the model
pose using /gazebo/set_model_state service.
'''
class TurtleBot3Env(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        '''
        Initializes a new TurtleBot3Env environment.
        '''
        rospy.logdebug("Start TurtleBot3Env INIT...")

        # We launch the init function of the parent class robot_gazebo_env.RobotGazeboEnv
        # Remember to reset the WORLD and not the whole simulation:
        super(TurtleBot3Env, self).__init__(reset_world_or_sim="WORLD")

        
        # To check any topic we need to have the simulation running:
        self.gazebo.unpauseSim()
        # Define the publisher to send control commands to Turtlebot3:
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # Check that all the sensors and actuators are working properly:
        self._check_all_systems_ready()
        # If all sensors are fine, we can proceed to subscribe to their topics:
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/imu", Imu, self._imu_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        # Once we know that all systems are fine, we can pause the simulation again:
        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished TurtleBot3Env INIT...")

        
    #--------------------- RobotGazeboEnv Methods ---------------------#
    '''
    We must implement in the Task Environment the rest of methods defined in RobotGazeboEnv
    because those methods depend on the task and how the robot is going to be trained.
    '''

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("Checking sensors connection")
        self._check_subscribers_connection("/odom", Odometry)
        self._check_subscribers_connection("/imu", Imu)
        self._check_subscribers_connection("/scan", LaserScan)
        rospy.logdebug("All sensors ready")

        rospy.logdebug("Checking actuators connection")
        self._check_publishers_connection(self._cmd_vel_pub)
        rospy.logdebug("All actuators ready")

        return True
    #------------------------------------------------------------------#


    #--------------------- TurtleBot3Env Methods ----------------------#
    '''
    Here we will implement all the methods needed to control the robot and take observations from it.
    That is, read sensors and control actuators.
    '''

    def _odom_callback(self, data):
        '''
        Receive odometry data from /odom topic
        '''
        self.odom = data
    
    def _imu_callback(self, data):
        '''
        Receive IMU data from /imu topic
        '''
        self.imu = data

    def _laser_scan_callback(self, data):
        '''
        Receive laser readings from /scan topic
        '''
        self.laser_scan = data

    def _check_subscribers_connection(self, topic, msg):
        '''
        Check that the topic is publishing data
        '''
        self.data = None
        rospy.logdebug("Waiting for " + str(topic) + " to be READY...")
        while self.data is None and not rospy.is_shutdown():
            try:
                self.data = rospy.wait_for_message(topic, msg, timeout=1.0)
                rospy.logdebug("Current " + str(topic) + " READY=>")

            except:
                rospy.logerr("Current " + str(topic) + " not ready yet, retrying...")

        return self.data

    def _check_publishers_connection(self, publisher):
        """
        Check that the given publisher is working and there is a connection with a subscriber
        """
        rate = rospy.Rate(10)  # 10hz
        while publisher.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers yet, retrying...")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("Publisher Connected")
    
    def move_base(self, linear_speed, angular_speed, wait_time):
        '''
        This function is used to move the Turtlebot3 base. To do that, the desired
        linear and angular speed are given. Once we send the control command to the
        mobile base, we wait a predefined time to make sure that the desired speed has
        been achieved.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        '''
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection(self._cmd_vel_pub)
        self._cmd_vel_pub.publish(cmd_vel_value)

        # This function doesn't work properly:
        # self.wait_until_twist_achieved(cmd_vel_value,
        #                                 epsilon,
        #                                 update_rate)
        time.sleep(wait_time/self.gazebo.rtf) ######################################################################### modified
        # TODO: make the waiting time a paremeter read from the yaml file and make that the sleep depends on rtf
    #------------------------------------------------------------------#