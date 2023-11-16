#!/usr/bin/env python
import datetime
import sys
from copy import copy, deepcopy

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import matplotlib.pyplot as plt
from git import Repo

import math
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict, Tuple
import warnings
import carla
import os
import time

from ray.tune.result import PID

from rllib_integration.GetAngle import calculate_angle_with_center_of_lane, angle_between
from rllib_integration.TestingWayPointUpdater import plot_points, plot_route, draw_route_in_order
from rllib_integration.base_experiment import BaseExperiment
from rllib_integration.helper import post_process_image
from rllib_integration.Circle import get_radii
from PIL import Image
from simple_pid import PID
from rllib_integration.lidar_to_grid_map import generate_ray_casting_grid_map
import collections

from dqn.dqn_callbacks import get_route_type
import open3d as o3d
from matplotlib import cm
from rllib_integration.sensors.sensor_interface import LABEL_COLORS


class DQNExperimentBasic(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration
        self.acceleration_pid = PID(Kp=0.2,Ki=0.2,Kd=0.0,setpoint=8.33,sample_time=None,output_limits=(0,1))
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        self.traffic = self.config["others"]["traffic"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_angle_with_center = 0
        self.last_forward_velocity = 0
        self.custom_done_arrived = False
        self.last_action = -1
        self.lidar_points_count = []
        self.reward_metric = 0
        self.current_time = 'None'
        self.min_lidar_values = 1000000
        self.max_lidar_values = -100000
        self.lidar_max_points = self.config["hero"]["lidar_max_points"]
        self.counter = 0
        self.visualiseRoute = False
        self.visualiseImage = False
        self.visualiseOccupancyGirdMap = False
        self.counterThreshold = 10
        self.last_hyp_distance_to_next_waypoint = 0
        self.last_hyp_distance_to_next_plus_1_waypoint = 0
        self.passed_waypoint = False

        self.last_closest_distance_to_next_waypoint_line = 0
        self.last_closest_distance_to_next_plus_1_waypoint_line = 0
        self.current_trailer_waypoint = 0
        self.x_dist_to_waypoint = []
        self.y_dist_to_waypoint = []
        self.angle_to_center_of_lane_degrees = []
        self.angle_to_center_of_lane_degrees_2 = []
        self.angle_to_center_of_lane_degrees_5 = []
        self.angle_to_center_of_lane_degrees_7 = []
        self.angle_to_center_of_lane_degrees_ahead_waypoints = []
        # self.angle_to_center_of_lane_degrees_ahead_waypoints_2 = []
        self.angle_between_waypoints_minus5 = []
        self.angle_between_waypoints_minus7 = []
        self.angle_between_waypoints_minus10 = []
        self.angle_between_waypoints_minus12 = []
        self.angle_between_waypoints_5 = []
        self.angle_between_waypoints_7 = []
        self.angle_between_waypoints_10 = []
        self.angle_between_waypoints_12 = []
        self.truck_bearing_to_waypoint = []
        self.truck_bearing_to_waypoint_2 = []
        self.truck_bearing_to_waypoint_5 = []
        self.truck_bearing_to_waypoint_7 = []
        self.truck_bearing_to_waypoint_10 = []
        self.distance_to_center_of_lane = []
        # self.bearing_to_ahead_waypoints_ahead_2 = []
        self.angle_between_truck_and_trailer = []
        self.trailer_bearing_to_waypoint = []
        self.trailer_bearing_to_waypoint_2 = []
        self.trailer_bearing_to_waypoint_5 = []
        self.trailer_bearing_to_waypoint_7 = []
        self.trailer_bearing_to_waypoint_10 = []
        self.forward_velocity = []
        # self.forward_velocity_x = []
        # self.forward_velocity_z = []
        self.vehicle_path = []
        self.temp_route = []
        self.hyp_distance_to_next_waypoint = []
        self.hyp_distance_to_next_plus_1_waypoint = []
        self.closest_distance_to_next_waypoint_line = []
        self.closest_distance_to_next_plus_1_waypoint_line = []
        # self.acceleration = []
        self.collisions = []
        self.lidar_data = collections.deque(maxlen=4)
        self.entry_idx = -1
        self.exit_idx = -1
        self.current_forward_velocity = 0

        self.last_no_of_collisions_truck = 0
        self.last_no_of_collisions_trailer = 0

        self.occupancy_map_x = 84
        self.occupancy_map_y = 84
        self.max_amount_of_occupancy_maps = 11
        self.radii = []
        self.mean_radius = []
        self.point_reward = []
        self.point_reward_location = []
        self.line_reward = []
        self.line_reward_location = []
        self.total_episode_reward = []

        self.custom_enable_rendering = False
        self.truck_lidar_collision = False
        self.trailer_lidar_collision = False
        self.visualiseLIDAR = False
        self.visualiseRADAR = False
        self.visualiseLIDARCircle = False
        self.lidar_window()


        self.occupancy_maps = collections.deque(maxlen=self.max_amount_of_occupancy_maps)

        for i in range(self.max_amount_of_occupancy_maps):
            self.occupancy_maps.append(np.zeros((self.occupancy_map_y,self.occupancy_map_x,1)))

        self.VIRIDIS = np.array(cm.get_cmap('plasma').colors)

        self.VID_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])
        self.COOL_RANGE = np.linspace(0.0, 1.0, self.VIRIDIS.shape[0])
        self.COOL = np.array(cm.get_cmap('winter')(self.COOL_RANGE))
        self.COOL = self.COOL[:, :3]

        repo = Repo('.')
        remote = repo.remote('origin')
        try:
            remote.fetch()
        except Exception as e:
            print(e)
            print('Nothing majorly bad happened')

        commit_hash = deepcopy(str(repo.head.commit)[:11])
        self.directory = f"/home/daniel/data-rllib-integration/data/data_{commit_hash}"

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def get_azimuth(self, sensor_centre, detection_point):
        # print(f'sensor_centre {sensor_centre}')
        # print(f'detection_point {detection_point}')
        x_sensor_centre = sensor_centre[0]
        y_sensor_centre = sensor_centre[1]

        x_detection_point = detection_point[0]
        y_detection_point = detection_point[1]

        numerator = x_detection_point
        denominator = y_detection_point

        angle = abs(math.degrees(math.atan(numerator/denominator)))

        if numerator > 0 and denominator > 0:
            pass
        elif numerator > 0 and denominator < 0:
            angle = 180- angle
        elif numerator < 0 and denominator > 0:
            angle = 360-angle
        elif numerator < 0 and denominator < 0:
            angle = angle + 180
        else:
            angle = 0
            print(f'Numerator and denominator both 0 {numerator}/{denominator}')
            # raise Exception('Numerator and denominator both 0')

        return angle

    def distance_between_to_points(self, sensor_centre, detection_point):

        return math.sqrt(
            (detection_point[0])**2
            +
            (detection_point[1])**2
            # +
            # (sensor_centre[2]-detection_point[2])**2
        )


    def lidar_window(self):
        if self.visualiseLIDAR or self.visualiseRADAR:
            name = 'lidar' if self.visualiseLIDAR else 'radar'
            self.lidar_point_list = o3d.geometry.PointCloud()
            self.radar_point_list = o3d.geometry.PointCloud()

            self.vis = o3d.visualization.Visualizer()

            self.vis.create_window(
                window_name=name,
                width=960,
                height=540,
                left=480,
                top=270)
            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 1
            self.vis.get_render_option().show_coordinate_frame = True

            self.frame = 0
            self.dt0 = datetime.datetime.now()

    def get_sidewalk_vehicle_lidar_points(self,lidar_points):
        # 8 is the number for sidewalk while 10 is for vehicles

        # We choose it from the second axis of lidar_points because
        # in sensor.py parse() we output the ObjTag in the third axis

        # lidar_points[0] is used to take the values from since we want
        # the x-axis values i.e those infront of the lidar sensor

        sidewalk_usable_indices = np.where(lidar_points[2] == 8)
        sidewalk_lidar_points = np.take(lidar_points[0], sidewalk_usable_indices)

        vehicle_usable_indices = np.where(lidar_points[2] == 10)
        vehicle_lidar_points = np.take(lidar_points[0], vehicle_usable_indices)

        return sidewalk_lidar_points[0], vehicle_lidar_points[0]

    def save_to_file(self, file_name, data):
        # Saving LIDAR point count
        counts = open(file_name, 'a')
        counts.write(f'${self.current_time}$')
        counts.write(str(data))
        counts.write(str('\n'))
        counts.close()

    def lidar_right_left_closest_points(self,sensor_data,sensor_name):
        sensor_name_no_trailer = sensor_name[:-8]
        lidar_points = sensor_data[sensor_name][1]
        lidar_range = float(self.config["hero"]["sensors"][sensor_name_no_trailer]["range"])

        # LIDAR GRAPH
        #           x
        #           |
        #           |
        # ----------+-------- y
        #           |
        #           |

        # Left and Right Lidar points
        horizontal_indices = np.where((lidar_points[0] > -1) & (lidar_points[0] < 1))
        horizontal_relevant_lidar_y_points = lidar_points[1][horizontal_indices]

        if len(horizontal_relevant_lidar_y_points) == 0:
            right = 1
            left = 1
        else:
            greater_0_indices = np.where(horizontal_relevant_lidar_y_points > 0)
            smaller_0_indices = np.where(horizontal_relevant_lidar_y_points < 0)

            if greater_0_indices[0].size != 0:
                right = min(abs(horizontal_relevant_lidar_y_points[greater_0_indices])) / lidar_range
            else:
                right = 1

            if smaller_0_indices[0].size != 0:
                left = min(abs(horizontal_relevant_lidar_y_points[smaller_0_indices])) / lidar_range
            else:
                left = 1

            if right < 0:
                right = 0

            if left < 0:
                left = 0
        return right, left


    # Rotation matrix function
    def rotate_matrix(self, x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
        """
        Rotates a point in the xy-plane counterclockwise through an angle about the origin
        https://en.wikipedia.org/wiki/Rotation_matrix
        :param x: x coordinate
        :param y: y coordinate
        :param x_shift: x-axis shift from origin (0, 0)
        :param y_shift: y-axis shift from origin (0, 0)
        :param angle: The rotation angle in degrees
        :param units: DEGREES (default) or RADIANS
        :return: Tuple of rotated x and y
        """

        # Shift to origin (0,0)
        x = x - x_shift
        y = y - y_shift

        # Convert degrees to radians
        if units == "DEGREES":
            angle = math.radians(angle)

        # Rotation matrix multiplication to get rotated x & y
        xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
        yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

        return xr, yr

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False
        self.done_collision_truck = False
        self.done_collision_trailer = False
        self.done_arrived = False
        self.custom_done_arrived = False
        self.done_far_from_path = False
        self.truck_lidar_collision = False
        self.trailer_lidar_collision = False

        self.current_trailer_waypoint = 0

        for i in range(self.max_amount_of_occupancy_maps):
            self.occupancy_maps.append(np.zeros((self.occupancy_map_y, self.occupancy_map_x,1)))

        # hero variables
        self.last_location = None
        self.last_velocity = 0
        self.last_dist_to_finish = 0


        self.last_angle_with_center = 0
        self.last_forward_velocity = 0

        self.last_no_of_collisions_truck = 0
        self.last_no_of_collisions_trailer = 0

        self.last_hyp_distance_to_next_waypoint = 0
        self.last_hyp_distance_to_next_plus_1_waypoint = 0

        self.last_closest_distance_to_next_waypoint_line = 0
        self.last_closest_distance_to_next_plus_1_waypoint_line = 0

        self.save_to_file(f"{self.directory}/hyp_distance_to_next_waypoint", self.hyp_distance_to_next_waypoint)
        self.save_to_file(f"{self.directory}/hyp_distance_to_next_plus_1_waypoint", self.hyp_distance_to_next_plus_1_waypoint)
        self.save_to_file(f"{self.directory}/closest_distance_to_next_waypoint_line", self.closest_distance_to_next_waypoint_line)
        self.save_to_file(f"{self.directory}/closest_distance_to_next_plus_1_waypoint_line", self.closest_distance_to_next_plus_1_waypoint_line)
        self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees", self.angle_to_center_of_lane_degrees)
        self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_2", self.angle_to_center_of_lane_degrees_2)
        self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_5", self.angle_to_center_of_lane_degrees_5)
        self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_7", self.angle_to_center_of_lane_degrees_7)
        self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_ahead_waypoints", self.angle_to_center_of_lane_degrees_ahead_waypoints)
        # self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_ahead_waypoints_2", self.angle_to_center_of_lane_degrees_ahead_waypoints_2)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_minus5", self.angle_between_waypoints_minus5)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_minus7", self.angle_between_waypoints_minus7)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_minus10", self.angle_between_waypoints_minus10)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_minus12", self.angle_between_waypoints_minus12)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_5", self.angle_between_waypoints_5)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_7", self.angle_between_waypoints_7)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_10", self.angle_between_waypoints_10)
        self.save_to_file(f"{self.directory}/angle_between_waypoints_12", self.angle_between_waypoints_12)
        self.save_to_file(f"{self.directory}/truck_bearing_to_waypoint", self.truck_bearing_to_waypoint)
        self.save_to_file(f"{self.directory}/truck_bearing_to_waypoint_2", self.truck_bearing_to_waypoint_2)
        self.save_to_file(f"{self.directory}/truck_bearing_to_waypoint_5", self.truck_bearing_to_waypoint_5)
        self.save_to_file(f"{self.directory}/truck_bearing_to_waypoint_7", self.truck_bearing_to_waypoint_7)
        self.save_to_file(f"{self.directory}/truck_bearing_to_waypoint_10", self.truck_bearing_to_waypoint_10)
        self.save_to_file(f"{self.directory}/distance_to_center_of_lane", self.distance_to_center_of_lane)
        # self.save_to_file(f"{self.directory}/bearing_to_ahead_waypoints_ahead_2", self.bearing_to_ahead_waypoints_ahead_2)
        self.save_to_file(f"{self.directory}/angle_between_truck_and_trailer", self.angle_between_truck_and_trailer)
        self.save_to_file(f"{self.directory}/trailer_bearing_to_waypoint", self.trailer_bearing_to_waypoint)
        self.save_to_file(f"{self.directory}/trailer_bearing_to_waypoint_2", self.trailer_bearing_to_waypoint_2)
        self.save_to_file(f"{self.directory}/trailer_bearing_to_waypoint_5", self.trailer_bearing_to_waypoint_5)
        self.save_to_file(f"{self.directory}/trailer_bearing_to_waypoint_7", self.trailer_bearing_to_waypoint_7)
        self.save_to_file(f"{self.directory}/trailer_bearing_to_waypoint_10", self.trailer_bearing_to_waypoint_10)
        self.save_to_file(f"{self.directory}/forward_velocity", self.forward_velocity)
        self.save_to_file(f"{self.directory}/line_reward", self.line_reward)
        self.save_to_file(f"{self.directory}/line_reward_location", self.line_reward_location)
        self.save_to_file(f"{self.directory}/point_reward", self.point_reward)
        self.save_to_file(f"{self.directory}/point_reward_location", self.point_reward_location)
        # self.save_to_file(f"{self.directory}/forward_velocity_x", self.forward_velocity_x)
        # self.save_to_file(f"{self.directory}/forward_velocity_z", self.forward_velocity_z)
        # self.save_to_file(f"{self.directory}/acceleration", self.acceleration)
        self.save_to_file(f"{self.directory}/route", self.temp_route)
        self.save_to_file(f"{self.directory}/path", self.vehicle_path)
        self.save_to_file(f"{self.directory}/lidar_data", self.lidar_data)
        self.save_to_file(f"{self.directory}/collisions", self.collisions)
        self.save_to_file(f"{self.directory}/radii",self.radii)
        self.save_to_file(f"{self.directory}/mean_radius",self.mean_radius)
        self.save_to_file(f"{self.directory}/total_episode_reward",self.total_episode_reward)
        self.entry_idx = -1
        self.exit_idx = -1
        self.last_action = -1


        # Saving LIDAR point count
        # file_lidar_counts = open(os.path.join('lidar_output','lidar_point_counts.txt'), 'a')
        # file_lidar_counts.write(str(self.lidar_points_count))
        # file_lidar_counts.write(str('\n'))
        # file_lidar_counts.close()
        #
        # file_lidar_counts = open(os.path.join('lidar_output', 'min_lidar_values.txt'), 'a')
        # file_lidar_counts.write(str("Min Lidar Value:" + str(self.min_lidar_values)))
        # file_lidar_counts.write(str('\n'))
        # file_lidar_counts.close()
        #
        # file_lidar_counts = open(os.path.join('lidar_output', 'max_lidar_values.txt'), 'a')
        # file_lidar_counts.write(str("Max Lidar Value:" + str(self.max_lidar_values)))
        # file_lidar_counts.write(str('\n'))
        # file_lidar_counts.close()

        self.min_lidar_values = 1000000
        self.max_lidar_values = -100000
        self.lidar_points_count = []
        self.counter = 0
        self.x_dist_to_waypoint = []
        self.y_dist_to_waypoint = []
        self.angle_to_center_of_lane_degrees = []
        self.angle_to_center_of_lane_degrees_2 = []
        self.angle_to_center_of_lane_degrees_5 = []
        self.angle_to_center_of_lane_degrees_7 = []
        self.angle_to_center_of_lane_degrees_ahead_waypoints = []
        # self.angle_to_center_of_lane_degrees_ahead_waypoints_2 = []
        self.angle_between_waypoints_minus5 = []
        self.angle_between_waypoints_minus7 = []
        self.angle_between_waypoints_minus10 = []
        self.angle_between_waypoints_minus12 = []
        self.angle_between_waypoints_5 = []
        self.angle_between_waypoints_7 = []
        self.angle_between_waypoints_10 = []
        self.angle_between_waypoints_12 = []
        self.truck_bearing_to_waypoint = []
        self.truck_bearing_to_waypoint_2 = []
        self.truck_bearing_to_waypoint_5 = []
        self.truck_bearing_to_waypoint_7 = []
        self.truck_bearing_to_waypoint_10 = []
        self.distance_to_center_of_lane = []
        self.trailer_bearing_to_waypoint = []
        self.trailer_bearing_to_waypoint_2 = []
        self.trailer_bearing_to_waypoint_5 = []
        self.trailer_bearing_to_waypoint_7 = []
        self.trailer_bearing_to_waypoint_10 = []
        # self.bearing_to_ahead_waypoints_ahead_2 = []
        self.angle_between_truck_and_trailer = []
        self.trailer_bearing_to_waypoint = []
        self.forward_velocity = []
        self.line_reward = []
        self.line_reward_location = []
        self.point_reward = []
        self.point_reward_location = []
        self.total_episode_reward = []
        # self.forward_velocity_x = []
        # self.forward_velocity_z = []
        self.vehicle_path = []
        self.temp_route = []
        self.hyp_distance_to_next_waypoint = []
        self.hyp_distance_to_next_plus_1_waypoint = []
        self.closest_distance_to_next_waypoint_line = []
        self.closest_distance_to_next_plus_1_waypoint_line = []
        self.collisions = []
        self.lidar_data = collections.deque(maxlen=4)
        self.radii = []
        self.mean_radius = []
        self.reward_metric = 0
        # self.acceleration = []








    # [33,28, 27, 17,  14, 11, 10, 5]

    def get_min_lidar_point(self,lidar_points, lidar_range) :
        if len(lidar_points) != 0:
            return np.clip(min(lidar_points)/lidar_range,0,1)
        else:
            return 1

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        return Discrete(len(self.get_actions()))
        # return Box(low=np.array([0,-1]),high=np.array([1,1]),dtype=float)
    def get_observation_space(self):
        """
        Set observation space as location of vehicle im x,y starting at (0,0) and ending at (1,1)
        :return:
        """
        lower_bound_list = [0 for x in range(360)]
        upper_bound_list = [300 for x in range(360)]


        obs_space = Dict({
            'values': Box(
                low=np.array(lower_bound_list),
                high=np.array(upper_bound_list),
                dtype=np.float32
            )
        })

        return obs_space

    def get_actions(self):
        acceleration_value = self.acceleration_pid(self.current_forward_velocity)
        print(f"Acceleration value {acceleration_value}") if self.custom_enable_rendering else None

        return {
            # Discrete with pid value
            0: [acceleration_value, 0.00, 0.0, False, False],  # Straight
            1: [acceleration_value, 0.50, 0.0, False, False],  # Right
            2: [acceleration_value, -0.50, 0.0, False, False],  # Left
        }



    def compute_action(self, action):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero"""
        # action_control = action
        action_control = self.get_actions()[int(action)]


        action = carla.VehicleControl()
        action.throttle = action_control[0]
        action.steer = action_control[1]
        action.brake = action_control[2]
        action.reverse = False
        action.hand_brake = False

        action_msg = ""

        if action_control[0] != 0:
            action_msg += f" {action_control[0]} Forward "

        if action_control[1] < 0:
            action_msg += f"{action_control[1]} Left "

        if action_control[1] > 0:
            action_msg += f"{action_control[1]} Right "

        if action_control[2] != 0:
            action_msg += f"{action_control[2]} Break "

        if action_msg == "":
            action_msg += " Coast "


        # print(f'Throttle {action.throttle} Steer {action.steer} Brake {action.brake} Reverse {action.reverse} Handbrake {action.hand_brake}')
        print(f"----------------------------------->{action_msg}") if self.custom_enable_rendering else None

        self.last_action = action


        return action

    def get_observation(self, sensor_data, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        # State for Truck
            # Current position
            # Angle
            # velocity
            # acceleration
            # collision
            # last action

        # For Trailer
            # Position
            # angle
            # collision
        self.custom_enable_rendering = core.custom_enable_rendering
        self.current_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")

        radii, mean_radius = get_radii(core.route,core.last_waypoint_index,5)

        self.entry_idx = core.entry_spawn_point_index
        self.exit_idx = core.exit_spawn_point_index

        number_of_waypoints_ahead_to_calculate_with = 0
        ahead_waypoints = 10
        ahead_waypoints_2 = 20

        # Getting truck location
        truck_transform = core.hero.get_transform()
        truck_forward_vector = truck_transform.get_forward_vector()

        x_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.x - truck_transform.location.x)
        y_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.y - truck_transform.location.y)
        hyp_distance_to_next_waypoint = math.sqrt((x_dist_to_next_waypoint) ** 2 + (y_dist_to_next_waypoint) ** 2)
        self.last_hyp_distance_to_next_waypoint = hyp_distance_to_next_waypoint

        if core.config["truckTrailerCombo"]:
            # Getting trailer location
            trailer_transform = core.hero_trailer.get_transform()
            trailer_forward_vector = trailer_transform.get_forward_vector()

        d = 3
        magnitude_of_trailer_forward_vector = math.sqrt(trailer_forward_vector.x**2+trailer_forward_vector.y**2+trailer_forward_vector.z**2)
        trailer_rear_axle_transform = carla.Transform(
            carla.Location(trailer_transform.location.x-trailer_forward_vector.x*magnitude_of_trailer_forward_vector*d,
                           trailer_transform.location.y-trailer_forward_vector.y*magnitude_of_trailer_forward_vector*d,
                           trailer_transform.location.z-trailer_forward_vector.z*magnitude_of_trailer_forward_vector*d),
            carla.Rotation(0, 0, 0))

        # print(f"BEFORE CHECKING IF PASSED LAST WAYPOINT {core.last_waypoint_index}")
        # Checking if we have passed the last way point

        distance_to_next_waypoint_line = core.distToSegment(truck_transform=truck_transform,waypoint_plus_current=1)
        self.passed_waypoint = False
        in_front_of_waypoint = core.is_in_front_of_waypoint(truck_transform.location.x, truck_transform.location.y)
        if 10 > distance_to_next_waypoint_line and (in_front_of_waypoint == 0 or in_front_of_waypoint == 1):
            core.last_waypoint_index = core.last_waypoint_index + 1
            self.last_hyp_distance_to_next_waypoint = 0
            self.last_closest_distance_to_next_waypoint_line = 0
            self.passed_waypoint = True
            print('Passed Waypoint <------------') if self.custom_enable_rendering else None
        else:
            pass

        distance_to_next_waypoint_line_trailer = core.distToSegment(truck_transform=trailer_rear_axle_transform,waypoint_no=self.current_trailer_waypoint,waypoint_plus_current=1)
        in_front_of_waypoint_trailer = core.is_in_front_of_waypoint(trailer_rear_axle_transform.location.x, trailer_rear_axle_transform.location.y,waypoint_no=self.current_trailer_waypoint)
        if 10 > distance_to_next_waypoint_line_trailer and (in_front_of_waypoint_trailer == 0 or in_front_of_waypoint_trailer == 1):
            self.current_trailer_waypoint = self.current_trailer_waypoint + 1
            print('Trailer Passed Waypoint <------------') if self.custom_enable_rendering else None
        else:
            pass

        x_virtual_lidar_centre_point = (trailer_rear_axle_transform.location.x + truck_transform.location.x) / 2
        y_virtual_lidar_centre_point = (trailer_rear_axle_transform.location.y + truck_transform.location.y) / 2
        z_virtual_lidar_centre_point = (trailer_rear_axle_transform.location.z + truck_transform.location.z) / 2

        virtual_lidar_centre_point = (x_virtual_lidar_centre_point,y_virtual_lidar_centre_point,z_virtual_lidar_centre_point)



        trailer_angle_to_center_of_lane_degrees = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index-1].location,
            current_position=trailer_transform.location,
            next_position=core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location)

        distance_to_center_of_lane = core.shortest_distance_to_center_of_lane(truck_transform=truck_transform)
        trailer_distance_to_center_of_lane = core.shortest_distance_to_center_of_lane(truck_transform=trailer_rear_axle_transform,waypoint_no=self.current_trailer_waypoint)

        if trailer_angle_to_center_of_lane_degrees < 0:
            trailer_distance_to_center_of_lane = -trailer_distance_to_center_of_lane

        forward_vector_waypoint_0 = core.route[core.last_waypoint_index + 0].get_forward_vector()

        trailer_bearing_to_waypoint = angle_between(waypoint_forward_vector=forward_vector_waypoint_0,
                                                  vehicle_forward_vector=trailer_forward_vector)

        forward_velocity = np.clip(self.get_speed(core.hero), 0, None)
        self.current_forward_velocity = forward_velocity
        # forward_velocity_x = np.clip(self.get_forward_velocity_x(core.hero), 0, None)
        # forward_velocity_z = np.clip(self.get_forward_velocity_z(core.hero), 0, None)
        # acceleration = np.clip(self.get_acceleration(core.hero), 0, None)

        lidar_points_dict = {}

        for sensor in sensor_data:
            if sensor == 'collision_truck':
                # TODO change to only take collision with road
                # TO CHECK BY CHECKING LIDAR OUTPUT WHEN IN COMPLETE TURN
                # MAYBE I WOULD ABSTAIN FROM REMOVING IT BECAUSE YOU ARE STILL pushing the truck to the limit

                # static.sidewalk

                self.last_no_of_collisions_truck = len(sensor_data[sensor][1])
                self.collisions.append(['truck',str(sensor_data[sensor][1][0].get_transform()),str(sensor_data[sensor][1][1]),self.current_time])

                print(f'COLLISIONS TRUCK {sensor_data[sensor][1][0]}')

            elif sensor == "collision_trailer":
                self.last_no_of_collisions_trailer = len(sensor_data[sensor][1])
                self.collisions.append(['trailer',str(sensor_data[sensor][1][0].get_transform()),str(sensor_data[sensor][1][1]),self.current_time])
                print(f'COLLISIONS TRAILER {sensor_data[sensor][1][0]}')

            elif sensor == "trailer_lidar_trailer":
                data = sensor_data[sensor][1]
                lidar_range = float(self.config["hero"]["sensors"]["trailer_lidar"]["range"])

                #
                # print(f'\n\nlen data {len(data[0])}\n\n')
                # print(f'percentile 75 {np.percentile(data[3],50)}')

                # print(f'min x {min(data[0])}')
                # print(f'max x {max(data[0])}')
                #
                # print(f'min y {min(data[3])}')
                # print(f'max y {max(data[3])}')




                for x_lidar_point,y_lidar_point,z_lidar_point in zip(data[0],data[1],data[3]):
                    # print(f'x y z {x_lidar_point} {y_lidar_point} {z_lidar_point}')
                    if x_lidar_point == 0 and y_lidar_point == 0:
                        continue

                    # azimuth_rounded = round(self.get_azimuth(sensor_centre=virtual_lidar_centre_point, detection_point=(x_lidar_point,y_lidar_point)))
                    azimuth_rounded = round(self.get_azimuth(sensor_centre=virtual_lidar_centre_point, detection_point=(x_lidar_point,y_lidar_point)))

                    # print(f'{azimuth_rounded} = azimuth of {virtual_lidar_centre_point} to {(x_lidar_point,y_lidar_point)}')
                    # IMPORTANT HERE Z is lidar_point[3] not 2 because 2 is the ObjTag
                    distance_from_sensor = self.distance_between_to_points(sensor_centre=virtual_lidar_centre_point, detection_point=(x_lidar_point,y_lidar_point,z_lidar_point))

                    # print(f'azimuth_rounded {azimuth_rounded}')
                    # print(f'distance_from_sensor {distance_from_sensor}')

                    assert 0 <= azimuth_rounded <= 360

                    if lidar_points_dict.get(azimuth_rounded) == None:
                        lidar_points_dict[azimuth_rounded] = distance_from_sensor
                        # lidar_points_dict[azimuth_rounded] = np.clip(distance_from_sensor/lidar_range,0,1)
                    else:
                        lidar_points_dict[azimuth_rounded] = min(distance_from_sensor, lidar_points_dict[azimuth_rounded])
                        # lidar_points_dict[azimuth_rounded] = min(np.clip(distance_from_sensor/lidar_range,0,1), lidar_points_dict[azimuth_rounded])

                # print(f'len lidar_points_dict {len(lidar_points_dict)}')
                if self.visualiseLIDAR:
                    data = sensor_data[sensor][1]

                    print(f'LEN LIDAR DATA {len(data)}')

                    # Isolate the intensity and compute a color for it
                    intensity = data[:, -1]
                    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
                    int_color = np.c_[
                        np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 0]),
                        np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 1]),
                        np.interp(intensity_col, self.VID_RANGE, self.VIRIDIS[:, 2])]

                    points = data[:, :-1]

                    points[:, :1] = -points[:, :1]

                    self.lidar_point_list.points = o3d.utility.Vector3dVector(points)
                    self.lidar_point_list.colors = o3d.utility.Vector3dVector(int_color)


                    if self.frame == 2:
                        self.vis.add_geometry(self.lidar_point_list)
                    self.vis.update_geometry(self.lidar_point_list)


            elif sensor == "trailer_radar_trailer":
                if self.visualiseRADAR:
                    data = sensor_data[sensor][1]

                    radar_data = np.zeros((len(data), 4))

                    print(f'LEN RADAR DATA {len(data)}')

                    for i, detection in enumerate(data):
                        x = detection[2] * math.cos(detection[0]) * math.cos(detection[1])
                        y = detection[2] * math.cos(detection[0]) * math.sin(detection[1])
                        z = detection[2] * math.sin(detection[0])

                        radar_data[i, :] = [x, y, z, detection[3]]

                    intensity = np.abs(radar_data[:, -1])
                    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
                    int_color = np.c_[
                        np.interp(intensity_col, self.COOL_RANGE, self.COOL[:, 0]),
                        np.interp(intensity_col, self.COOL_RANGE, self.COOL[:, 1]),
                        np.interp(intensity_col, self.COOL_RANGE, self.COOL[:, 2])]

                    points = radar_data[:, :-1]
                    points[:, :1] = -points[:, :1]
                    self.radar_point_list.points = o3d.utility.Vector3dVector(points)
                    self.radar_point_list.colors = o3d.utility.Vector3dVector(int_color)

                    if self.frame == 2:
                        self.vis.add_geometry(self.radar_point_list)
                    self.vis.update_geometry(self.radar_point_list)

                    self.vis.poll_events()
                    self.vis.update_renderer()
                    # # This can fix Open3D jittering issues:
                    time.sleep(0.005)

                    process_time = datetime.datetime.now() - self.dt0
                    sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
                    sys.stdout.flush()
                    self.dt0 = datetime.datetime.now()
                    self.frame += 1






            elif sensor == "lidar_trailer_0_left_trailer":
                lidar_points = sensor_data['lidar_trailer_0_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_0_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_0_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_0_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

        output_values = []
        max_distance = 32
        for i in range(360):
            if lidar_points_dict.get(i) == None:
                output_values.append(max_distance)
            else:
                output_values.append(np.clip(lidar_points_dict[i],0,max_distance))

        if self.visualiseLIDARCircle and self.counter % 10 == 0:
            x_values = []
            y_values = []
            false_x_values = []
            false_y_values = []
            for angle, distance in enumerate(output_values):

                if 0 <= angle <= 90:
                    angle = angle
                    if distance == max_distance:
                        false_x_values.append(distance * abs(math.sin(math.radians(angle))))
                        false_y_values.append(distance * abs(math.cos(math.radians(angle))))
                    else:
                        x_values.append(distance * abs(math.sin(math.radians(angle))))
                        y_values.append(distance * abs(math.cos(math.radians(angle))))
                elif 90 < angle <= 180:
                    angle = 180-angle
                    if distance == max_distance:
                        false_x_values.append(distance * abs(math.sin(math.radians(angle))))
                        false_y_values.append(-distance * abs(math.cos(math.radians(angle))))
                    else:
                        x_values.append(distance * abs(math.sin(math.radians(angle))))
                        y_values.append(-distance * abs(math.cos(math.radians(angle))))
                elif 180 < angle <= 270:
                    angle = angle -180
                    if distance == max_distance:
                        false_x_values.append(-distance * abs(math.sin(math.radians(angle))))
                        false_y_values.append(-distance * abs(math.cos(math.radians(angle))))
                    else:
                        x_values.append(-distance * abs(math.sin(math.radians(angle))))
                        y_values.append(-distance * abs(math.cos(math.radians(angle))))
                elif 270 < angle <= 360:
                    angle = 360-angle
                    if distance == max_distance:
                        false_x_values.append(-distance * abs(math.sin(math.radians(angle))))
                        false_y_values.append(distance * abs(math.cos(math.radians(angle))))
                    else:
                        x_values.append(-distance * abs(math.sin(math.radians(angle))))
                        y_values.append(distance * abs(math.cos(math.radians(angle))))



            plt.plot(x_values, y_values, 'bo',markersize=1)
            plt.plot(false_x_values, false_y_values, 'ro',markersize=1)
            # plt.axis([0.3, 0.7, 0.3, 0.7])
            # plt.axis([0, 1, 0, 1])
            plt.title(f'temp')
            plt.gca().invert_yaxis()
            # plt.legend(loc='upper center')
            plt.show()

        if self.custom_enable_rendering:
            print(f'Entry Points {core.entry_spawn_point_index}| Exit point {core.exit_spawn_point_index}')
            print(f"forward_velocity:{np.float32(forward_velocity)}")
            time.sleep(0.04)

        self.forward_velocity.append(np.float32(forward_velocity))
        self.distance_to_center_of_lane.append(np.float32(distance_to_center_of_lane))
        self.vehicle_path.append((truck_transform.location.x,truck_transform.location.y))
        self.temp_route = deepcopy(core.route_points)
        self.radii.append(radii)


        self.counter += 1
        return {'values':output_values},\
            {"truck_z_value":truck_transform.location.z,
             "distance_to_center_of_lane":distance_to_center_of_lane,
             "truck_acceleration": self.get_acceleration(core.hero),
             'trailer_distance_to_center_of_lane':trailer_distance_to_center_of_lane,
             "trailer_bearing_to_waypoint":trailer_bearing_to_waypoint}

    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_forward_velocity_x(self,hero):
        vel = hero.get_velocity()
        return 3.6 * vel.x

    def get_forward_velocity_z(self, hero):
        vel = hero.get_velocity()
        return 3.6 * vel.z

    def get_acceleration(self,hero):
        acc = hero.get_acceleration()
        return math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

    def completed_route(self, core):
        # -2 Since we want to be done when the truck has passed the second to last point
        # in order to have the next waypoint to calculate with
        #print("Inside Complete Route")
        #print(f"Len(core.route) -2 : {len(core.route) -2 }")
        #print(f"core.last_waypoint_index{core.last_waypoint_index}")
        if len(core.route) - 20 <= core.last_waypoint_index:
            return True
        else:
            return False

    def min_max_normalisation(self, value, min, max):
        return (value - min) / (max -min)

    def get_done_status(self, observation, core, info=None):
        """Returns whether or not the experiment has to end"""
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1

        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = hero.get_location().z < -0.5
        self.done_collision_truck = (self.last_no_of_collisions_truck > 0)
        self.done_collision_trailer = (self.last_no_of_collisions_trailer > 0)
        self.done_arrived = self.completed_route(core)

        self.done_angle = False
        self.done_distance = False

        if info['trailer_bearing_to_waypoint'] < -math.pi/2 or info['trailer_bearing_to_waypoint'] > math.pi/2:
            print('Angle too high!')
            self.done_angle = True

        if info['trailer_distance_to_center_of_lane'] < -1.7 or info['trailer_distance_to_center_of_lane'] > 1.7:
            print('Distance to centre of lane too high')
            self.done_distance = True



        output = (self.done_time_idle or
                  self.done_falling or
                  self.done_time_episode or
                  self.done_collision_truck or self.done_collision_trailer or
                  self.done_arrived or
                  self.done_angle or
                  self.done_distance
                  )

        self.custom_done_arrived = self.done_arrived

        done_status_info = {
            'done_time_idle':self.done_time_idle,
            'done_falling': self.done_falling,
            'done_time_episode':self.done_time_episode,
            'done_collision_truck': self.done_collision_truck,
            'done_collision_trailer':self.done_collision_trailer,
            'done_arrived':self.done_arrived,
            'done_angle':self.done_angle,
            'done_distance':self.done_distance,
        }

        done_reason = ""
        if self.done_time_idle:
            done_reason += "done_time_idle"
        if self.done_falling:
            done_reason += "done_falling"
        if self.done_time_episode:
            done_reason += "done_time_episode"
        if self.done_collision_truck:
            done_reason += "done_collision_truck"
        if self.done_collision_trailer:
            done_reason += "done_collision_trailer"
        if self.done_arrived:
            done_reason += "done_arrived"
        if self.done_angle:
            done_reason += "done_angle"
        if self.done_distance:
            done_reason += "done_distance"

        if done_reason != "":
            data = f"ENTRY: {core.entry_spawn_point_index} EXIT: {core.exit_spawn_point_index} - {done_reason} \n"
            self.save_to_file(f"{self.directory}/done",data)
        # print(done_status_info)
        return bool(output), done_status_info

    def compute_reward(self, observation, info, core):
        """Computes the reward"""
        # est
        reward = 0

        hyp_distance_to_next_waypoint = observation['values'][1]
        distance_to_center_of_lane = observation['values'][5]

        reward_trailer_bearing_to_waypoint = info['trailer_bearing_to_waypoint']
        reward_trailer_distance_to_center_of_lane = info['trailer_distance_to_center_of_lane']

        self.last_hyp_distance_to_next_waypoint = hyp_distance_to_next_waypoint

        # 0 straight
        # 1 right
        # 2 left

        straight_action = 0
        right_action = 1
        left_action = 2

        # WHEN ON THE RIGHT SIDE

        if 0 < reward_trailer_distance_to_center_of_lane <= 0.17:

            if 0 <= reward_trailer_bearing_to_waypoint <= (math.pi/2):
                if self.last_action == left_action:
                    reward += 1
                else:
                    reward += -1

            elif -math.radians(10) <= reward_trailer_bearing_to_waypoint < 0:
                if self.last_action == straight_action:
                    reward += 1
                else:
                    reward += -1

            elif (-math.pi/2) <= reward_trailer_bearing_to_waypoint < -math.radians(10):
                if self.last_action == right_action:
                    reward += 1
                else:
                    reward += -1

        elif 0.17 < reward_trailer_distance_to_center_of_lane <= 0.36:

            if -math.radians(12) <= reward_trailer_bearing_to_waypoint <= (math.pi/2):
                if self.last_action == left_action:
                    reward += 1
                else:
                    reward += -1

            elif -math.radians(20) <= reward_trailer_bearing_to_waypoint < -math.radians(12):
                if self.last_action == straight_action:
                    reward += 1
                else:
                    reward += -1

            elif (-math.pi / 2) <= reward_trailer_bearing_to_waypoint < -math.radians(20):
                if self.last_action == right_action:
                    reward += 1
                else:
                    reward += -1

        elif 0.36 < reward_trailer_distance_to_center_of_lane <= 1.7:
            if -math.radians(22) <= reward_trailer_bearing_to_waypoint <= (math.pi/2):
                if self.last_action == left_action:
                    reward += 1
                else:
                    reward += -1

            elif -math.radians(30) <= reward_trailer_bearing_to_waypoint < -math.radians(22):
                if self.last_action == straight_action:
                    reward += 1
                else:
                    reward += -1

            elif (-math.pi/2) <= reward_trailer_bearing_to_waypoint < -math.radians(30):
                if self.last_action == right_action:
                    reward += 1
                else:
                    reward += -1

        # WHEN ON THE LEFT SIDE
        elif -0.17 <= reward_trailer_distance_to_center_of_lane < 0:

            if math.radians(10) <= reward_trailer_bearing_to_waypoint <= (math.pi/2):
                if self.last_action == left_action:
                    reward += 1
                else:
                    reward += -1
            elif 0 <= reward_trailer_bearing_to_waypoint < math.radians(10):
                if self.last_action == straight_action:
                    reward += 1
                else:
                    reward += -1
            elif (-math.pi/2) <= reward_trailer_bearing_to_waypoint < 0:
                if self.last_action == right_action:
                    reward += 1
                else:
                    reward += -1

        elif -0.36 <= reward_trailer_distance_to_center_of_lane < -0.17:

            if math.radians(20) <= reward_trailer_bearing_to_waypoint <= (math.pi/2):
                if self.last_action == left_action:
                    reward += 1
                else:
                    reward += -1

            elif math.radians(12) <= reward_trailer_bearing_to_waypoint < math.radians(20):
                if self.last_action == straight_action:
                    reward += 1
                else:
                    reward += -1

            elif (-math.pi / 2) <= reward_trailer_bearing_to_waypoint < math.radians(12):
                if self.last_action == right_action:
                    reward += 1
                else:
                    reward += -1

        elif -1.7 <= reward_trailer_distance_to_center_of_lane < -0.36:

            if math.radians(30) <= reward_trailer_bearing_to_waypoint <= (math.pi/2):
                if self.last_action == left_action:
                    reward += 1
                else:
                    reward += -1
            elif math.radians(22) <= reward_trailer_bearing_to_waypoint < math.radians(30):
                if self.last_action == straight_action:
                    reward += 1
                else:
                    reward += -1

            elif (-math.pi / 2) <= reward_trailer_bearing_to_waypoint < math.radians(22):
                if self.last_action == right_action:
                    reward += 1
                else:
                    reward += -1

        self.total_episode_reward.append(reward)
        self.reward_metric = reward
        print(f"FINAL REWARD: {reward}") if self.custom_enable_rendering else None
        print(f"---------------------------------------------") if self.custom_enable_rendering else None
        return reward