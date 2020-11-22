#!/usr/bin/env python

import rospy
import threading
import time
import numpy as np
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from std_msgs.msg import Float64, Float32
from geometry_msgs.msg import Vector3, Pose
from gazebo_msgs.srv import SetModelState, SpawnModel
from gazebo_msgs.msg import ModelState

'''
This class will allow us to control the Gazebo simulations by using ROS services.
'''
class GazeboConnection():

    def __init__(self, reset_world_or_sim, max_retry = 20, mavg_rtf_nitems=20):

        # Store class attributes values:
        self.reset_world_or_sim = reset_world_or_sim
        self._max_retry = max_retry # maximum times we try to call a Gazebo service

        # Gazebo services definition to control simulations:
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty) # service to unpause the simulation
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty) # service to pause the simulation
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty) # service to reset the entire simulation including time
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty) # service to reset the model poses
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        self.spawn_sdf_model = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        
        # Initialize physics parameters for Gazebo simulation:
        self.init_physics_parameters()
        self.setPhysicsParameters()

        # Always reset and pause the simulation at first to avoid problems with some robots
        self.resetSim()
        self.pauseSim()

        # Execute a thread to compute the Real Time Factor
        self.rtf = 1.0 # Real time (average)
        self.rtf_list = [1.0]*mavg_rtf_nitems # To compute moving average
        rtf_thread = threading.Thread(target=self.compute_rtf)
        rtf_thread.start()

    def pauseSim(self):
        '''
        This method is used to pause the simulation using the Gazebo services previously defined.

        NOTE: if you are using ROS based robots, meanwhile the simulation is paused, robot topics are
        frozen. This means that you cannot receive sensor information and you cannot send control references
        to the actuators if the simulation is paused.
        '''
        # Block the code until the service is available
        rospy.wait_for_service('/gazebo/pause_physics')
        
        paused_done = False
        counter = 0
        # We try to pause the simulation until we reach the maximum tries or until the service call have succeded
        while not paused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    self.pause()
                    paused_done = True
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/pause_physics service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo pause service"
                rospy.logerr(error_message)
                assert False, error_message

        self.sim_paused = True

    def unpauseSim(self):
        '''
        This method is used to unpause the simulation using the Gazebo services previously defined.
        '''

        # Block the code until the service is available
        rospy.wait_for_service('/gazebo/unpause_physics')

        unpaused_done = False
        counter = 0
        # We try to unpause the simulation until we reach the maximum tries or until the service call have succeded
        while not unpaused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    self.unpause()
                    unpaused_done = True
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/unpause_physics service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo unpause service"
                rospy.logerr(error_message)
                assert False, error_message

        self.sim_paused = False

    def resetSim(self):
        '''
        This method is used to reset the simulation depending on the parameter reset_world_or_sim, so the way
        in which we reset the simulation is different.
        This is needed because in some simulations, when reseted the simulation, some Gazebo plugins break,
        so instead of resetting the entire simulation using /gazebo/reset_simulation service, we reset only 
        the objects pose using the /gazebo/reset_world service.
        '''
        if self.reset_world_or_sim == "SIMULATION":
            self.resetSimulation()
        elif self.reset_world_or_sim == "WORLD":
            self.resetWorld()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            pass
        else:
            rospy.logerr("Wrong Reset Option:"+str(self.reset_world_or_sim))

    def resetSimulation(self):
        '''
        Resets the whole Gazebo simulation, including time.

        NOTE: this way of resetting the simulation can break some Gazebo plugins, like sensors.
        '''

        # Block the code until the service is available
        rospy.wait_for_service('/gazebo/reset_simulation')

        reset_done = False
        counter = 0
        # We try to reset the simulation until we reach the maximum tries or until the service call have succeded
        while not reset_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    self.reset_simulation()
                    reset_done = True
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/reset_simulation service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo reset service"
                rospy.logerr(error_message)
                assert False, error_message

    def resetWorld(self):
        '''
        Resets the Gazebo simulation, but only models pose.
        '''
        # Block the code until the service is available
        rospy.wait_for_service('/gazebo/reset_world')

        reset_done = False
        counter = 0
        # We try to reset the simulation until we reach the maximum tries or until the service call have succeded
        while not reset_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    self.reset_world()
                    reset_done = True
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/reset_simulation service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo reset service"
                rospy.logerr(error_message)
                assert False, error_message

    def init_physics_parameters(self):
        '''
        We initialise the physics parameters of the simulation, like gravity,
        friction coeficients and so on.

        NOTE: this parameters are the same that turtlebot3 Gazebo simulations use
        '''
        self._time_step = Float64(0.001)

        self._gravity = Vector3()
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config = ODEPhysics() #TODO: these properties can be got by calling /gazebo/get_physics_properties service. Check that these values are equal than default one
        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 150
        self._ode_config.sor_pgs_w = 1.4
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.01
        self._ode_config.contact_max_correcting_vel = 2000.0
        self._ode_config.cfm = 1e-5
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

    def setPhysicsParameters(self, update_rate = 1000.0):
        '''
        This method is used to set the physics parameters of Gazebo simulations. Mainly, this method is used 
        only to change in a fast way the simulation speed, so we can simulate faster than real time.

        If update_rate = 1000.0 we are running at real time, but if update_rate = -1 we are running the simulation
        as fast as possible.
        '''

        set_physics_msg = SetPhysicsPropertiesRequest()
        set_physics_msg.time_step = self._time_step.data
        set_physics_msg.max_update_rate = Float64(update_rate).data
        set_physics_msg.gravity = self._gravity
        set_physics_msg.ode_config = self._ode_config

        # Block the code until the service is available
        rospy.wait_for_service('/gazebo/set_physics_properties')

        changed_physics_done = False
        counter = 0
        # We try to change the simulation physics until we reach the maximum tries or until the service call have succeded
        while not changed_physics_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    self.set_physics(set_physics_msg)
                    changed_physics_done = True
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/set_physics_properties service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo set_physics_properties service"
                rospy.logerr(error_message)
                assert False, error_message

    def setModelState(self, model, x, y, z, tx, ty, tz, tw, frame="world"):
        '''
        This method is used to set a specified pose to any model in the simulation.
        '''
        state_msg = ModelState()
        state_msg.model_name = model
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = tx
        state_msg.pose.orientation.y = ty
        state_msg.pose.orientation.z = tz
        state_msg.pose.orientation.w = tw
        state_msg.reference_frame = frame

        # Block the code until the service is available
        rospy.wait_for_service('/gazebo/set_model_state')

        changed_state_done = False
        counter = 0
        # We try to change the model state until we reach the maximum tries or until the service call have succeded
        while not changed_state_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    self.set_model_state(state_msg)
                    changed_state_done = True
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/set_model_state service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo set_model_state service"
                rospy.logerr(error_message)
                assert False, error_message

    def spawnSDFmodel(self, model_name, model_path, x, y, z, tx, ty, tz, tw, frame="world"):
        '''
        This method is used to spawn any model in the simulation.
        '''
        model_name = model_name
        model_xml = open(model_path, 'r').read()
        robot_namespace = ""

        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = z
        initial_pose.orientation.x = tx
        initial_pose.orientation.y = ty
        initial_pose.orientation.z = tz
        initial_pose.orientation.w = tw

        reference_frame = frame

        # Block the code until the service is available
        rospy.wait_for_service('/gazebo/spawn_sdf_model')

        model_spawned_done = False
        counter = 0
        # We try to change the model state until we reach the maximum tries or until the service call have succeded
        while not model_spawned_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    self.spawn_sdf_model(model_name, model_xml, robot_namespace, initial_pose, reference_frame)
                    model_spawned_done = True
                except rospy.ServiceException as e:
                    counter += 1
                    rospy.logerr("/gazebo/spawn_sdf_model service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo spawn_sdf_model service"
                rospy.logerr(error_message)
                assert False, error_message

    def compute_rtf(self):
        '''
        This is a good way to estimate the Real Time Factor (RTF). This is the ratio between the simulation
        speed and the real time speed. This ratio will help us to make time.sleep specifying the real time we
        want to wait, so, automatically this wait time will adjust to the simulation time.

        NOTE: Remember that Gazebo Simulation must use the use_sim_time=True parameter, so ROS clock is updated
        with the simulation time and not real time.

        TODO: be careful! if this function is executed before the simulation is running, the rtf is not computed correctly!!
        '''
        # Topic to publish RTF:
        pub = rospy.Publisher("/real_time_factor", Float32, queue_size=1)

        first = False # Avoid publishing the first RTF computed because it is junk
        while not rospy.is_shutdown():
            # Only update RTF when Gazebo is not paused:
            if not self.sim_paused:
                # Get current simulation time in seconds:
                sim_time = rospy.get_rostime().secs + rospy.get_rostime().nsecs*1e-9
                
                # Wait approx 0.2 second on wall-clock (system clock or wall-time)
                t0 = time.time()
                t=0.0
                while t<0.2:
                    time.sleep(0.001)
                    # Only count real time when the simulation is not paused
                    if not self.sim_paused:
                        t += time.time()-t0
                        t0 = time.time()
                    else:
                        t0 = time.time()

                if first:
                    self.rtf_list.pop(0)
                    self.rtf_list.append(((rospy.get_rostime().secs + rospy.get_rostime().nsecs*1e-9)-sim_time)/t)
                    self.rtf = np.mean(self.rtf_list)
                    pub.publish(self.rtf)
                else:
                    first = True
