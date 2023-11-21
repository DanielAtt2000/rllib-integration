#!/usr/bin/env python
import datetime
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

from ppo.ppo_callbacks import get_route_type


class PPOExperimentBasic(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration
        # self.acceleration_pid = PID(Kp=0.2,Ki=0.2,Kd=0.0,setpoint=8.33,sample_time=None,output_limits=(0,1))
        self.steering_pid = PID(Kp=0.1,Ki=0.1,Kd=0.1,setpoint=0,sample_time=None,output_limits=(-1,1))
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        self.traffic = self.config["others"]["traffic"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_angle_with_center = 0
        self.last_forward_velocity = 0
        self.custom_done_arrived = False
        self.last_action = [0,0,0]
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
        self.trailer_vehicle_path = []
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
        self.current_trailer_distance_away_from_the_centre_of_the_lane = 0
        self.trailerangle_to_center_of_lane_degrees = 0

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


        self.occupancy_maps = collections.deque(maxlen=self.max_amount_of_occupancy_maps)

        for i in range(self.max_amount_of_occupancy_maps):
            self.occupancy_maps.append(np.zeros((self.occupancy_map_y,self.occupancy_map_x,1)))


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
        self.save_to_file(f"{self.directory}/trailer_path", self.trailer_vehicle_path)
        self.save_to_file(f"{self.directory}/lidar_data", self.lidar_data)
        self.save_to_file(f"{self.directory}/collisions", self.collisions)
        self.save_to_file(f"{self.directory}/radii",self.radii)
        self.save_to_file(f"{self.directory}/mean_radius",self.mean_radius)
        self.save_to_file(f"{self.directory}/total_episode_reward",self.total_episode_reward)
        self.entry_idx = -1
        self.exit_idx = -1
        self.last_action = [0,0,0]


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
        self.trailer_vehicle_path = []
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
        if self.traffic:
            obs_space = Dict({
                'values': Box(
                    low=np.array(
                        [0, 0, 0, 0, 0, 0, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi,
                         -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         # traffic
                         # velocity, acceleration, yaw, relative_x, relative_y
                         0, 0, -math.pi, -200, -200,
                         0, 0, -math.pi, -200, -200,
                         ]),
                    high=np.array(
                        [100, 200, 200, 200, 200, 25, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi,
                         math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1,
                         # traffic
                         # velocity, acceleration, yaw, relative_x, relative_y
                         100, 200, math.pi, 200, 200,
                         100, 200, math.pi, 200, 200,
                         ]),
                    dtype=np.float32
                )
            })
        else:
            obs_space = Dict({
                'values': Box(
                    low=np.array(
                        [0, 0, 0, 0, 0, 0,
                         # Angle to center of lane
                         -math.pi, -math.pi, -math.pi, -math.pi, -math.pi,
                         # Truck bearing to waypoint
                         -math.pi, -math.pi, -math.pi, -math.pi, -math.pi,
                         # Trailer bearing to waypoint
                         -math.pi, -math.pi, -math.pi, -math.pi, -math.pi,
                         # Angle between truck and trailer
                         -math.pi,
                         # Angle between waypoints
                         0, 0, 0, 0, 0, 0, 0, 0,
                         # Lidar data
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         # Radius
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                         ]),
                    high=np.array(
                        [100, 200, 200, 200, 200, 25,
                         # Angle to center of lane
                         math.pi, math.pi, math.pi, math.pi, math.pi,
                         # Truck bearing to waypoint
                         math.pi, math.pi, math.pi, math.pi, math.pi,
                         # Trailer bearing to waypoint
                         math.pi, math.pi, math.pi, math.pi, math.pi,
                         # Angle between truck and trailer
                         math.pi,
                         # Angle between waypoints
                         1, 1, 1, 1, 1, 1, 1, 1,
                         # Lidar data
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         # Radius
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                         ]),
                    dtype=np.float32
                )
            })

        return obs_space

    def get_actions(self):
        # acceleration_value = self.acceleration_pid(self.current_forward_velocity)
        if self.trailerangle_to_center_of_lane_degrees < 0:
            self.current_trailer_distance_away_from_the_centre_of_the_lane = -self.current_trailer_distance_away_from_the_centre_of_the_lane
        steering_pid_value = self.steering_pid(self.current_trailer_distance_away_from_the_centre_of_the_lane)

        print(f'self.current_trailer_distance_away_from_the_centre_of_the_lane {self.current_trailer_distance_away_from_the_centre_of_the_lane}')
        print(f'steering_pid_value {steering_pid_value}')
        # print(f"Acceleration value {acceleration_value}") if self.custom_enable_rendering else None

        # all_actions = {}
        #
        # # all_actions[0] = [0, 0.00, 0.0, False, False] # Straight no acc
        # # all_actions[1] = [0, 0.00, 0.2, False, False]  # 0.2 brake
        # # all_actions[2] = [0, 0.00, 0.4, False, False]  # 0.4 brake
        # # all_actions[3] = [0, 0.00, 0.6, False, False]  # 0.6 brake
        # # all_actions[4] = [0, 0.00, 0.8, False, False]  # 0.8 brake
        # # all_actions[5] = [0, 0.00, 1.0, False, False]  # 1.0 brake
        # key_counter = 0
        #
        # for acceleration in np.arange(0.0,1.2,0.2):
        #     for steering_angle in np.arange(-1,1.2,0.2):
        #         all_actions[key_counter] = [acceleration,steering_angle,0, False, False]
        #         key_counter += 1
        #
        # return all_actions

        return {
            # 0: [0, 0.00, 0.0, False, False],  # Straight no acc
            # 1: [0, 0.00, 0.5, False, False],  # half brake
            # 1: [0, 0.00, 1.0, False, False],  # full brake
            # 1: [0, 0.00, 1.0, False, False],  # full brake

            # PID stteirng and accelration
            0: [0.2,steering_pid_value, 0.0, False, False]
            # Discrete with pid value
            # 0: [acceleration_value, 0.00, 0.0, False, False],  # Straight
            # 1: [acceleration_value, 0.80, 0.0, False, False],  # Right
            # 2: [acceleration_value, 0.60, 0.0, False, False],  # Right
            # 3: [acceleration_value, 0.40, 0.0, False, False],  # Right
            # 4: [acceleration_value, 0.20, 0.0, False, False],  # Right
            # 5: [acceleration_value, -0.80, 0.0, False, False],  # Left
            # 6: [acceleration_value, -0.60, 0.0, False, False],  # Left
            # 7: [acceleration_value, -0.40, 0.0, False, False],  # Left
            # 8: [acceleration_value, -0.20, 0.0, False, False],  # Left


            # Discrete with custom acceleration
            # 0: [0.0, 0.00, 0.0, False, False],  # Dont Move
            # 1: [0.0, 0.00, 1.0, False, False],  # Brake
            #
            # 2: [0.1, 0.00, 0.0, False, False],  # Straight
            # 3: [0.1, 0.80, 0.0, False, False],  # Right
            # 4: [0.1, 0.60, 0.0, False, False],  # Right
            # 5: [0.1, 0.40, 0.0, False, False],  # Right
            # 6: [0.1, 0.20, 0.0, False, False],  # Right
            # 7: [0.1, -0.80, 0.0, False, False],  # Left
            # 8: [0.1, -0.60, 0.0, False, False],  # Left
            # 9: [0.1, -0.40, 0.0, False, False],  # Left
            # 10: [0.1, -0.20, 0.0, False, False],  # Left
            #
            # 11: [0.3, 0.00, 0.0, False, False],  # Straight
            # 12: [0.3, 0.80, 0.0, False, False],  # Right
            # 13: [0.3, 0.60, 0.0, False, False],  # Right
            # 14: [0.3, 0.40, 0.0, False, False],  # Right
            # 15: [0.3, 0.20, 0.0, False, False],  # Right
            # 16: [0.3, -0.80, 0.0, False, False],  # Left
            # 17: [0.3, -0.60, 0.0, False, False],  # Left
            # 18: [0.3, -0.40, 0.0, False, False],  # Left
            # 19: [0.3, -0.20, 0.0, False, False],  # Left
            #
            # 20: [0.6, 0.00, 0.0, False, False],  # Straight
            # 21: [0.6, 0.80, 0.0, False, False],  # Right
            # 22: [0.6, 0.60, 0.0, False, False],  # Right
            # 23: [0.6, 0.40, 0.0, False, False],  # Right
            # 24: [0.6, 0.20, 0.0, False, False],  # Right
            # 25: [0.6, -0.80, 0.0, False, False],  # Left
            # 26: [0.6, -0.60, 0.0, False, False],  # Left
            # 27: [0.6, -0.40, 0.0, False, False],  # Left
            # 28: [0.6, -0.20, 0.0, False, False],  # Left

            # 29: [0.9, 0.00, 0.0, False, False],  # Straight
            # 30: [0.9, 0.80, 0.0, False, False],  # Right
            # 31: [0.9, 0.60, 0.0, False, False],  # Right
            # 32: [0.9, 0.40, 0.0, False, False],  # Right
            # 33: [0.9, 0.20, 0.0, False, False],  # Right
            # 34: [0.9, -0.80, 0.0, False, False],  # Left
            # 35: [0.9, -0.60, 0.0, False, False],  # Left
            # 36: [0.9, -0.40, 0.0, False, False],  # Left
            # 37: [0.9, -0.20, 0.0, False, False],  # Left

            # 0: [0.0, 0.00, 0.0, False, False],  # Coast
            # 1: [0.0, 0.00, 1.0, False, False],  # Apply Break
            # 2: [0.0, 0.75, 0.0, False, False],  # Right
            # 3: [0.0, 0.50, 0.0, False, False],  # Right
            # 4: [0.0, 0.25, 0.0, False, False],  # Right
            # 5: [0.0, -0.75, 0.0, False, False],  # Left
            # 6: [0.0, -0.50, 0.0, False, False],  # Left
            # 7: [0.0, -0.25, 0.0, False, False],  # Left
            # 8: [0.15, 0.00, 0.0, False, False],  # Straight
            # 9: [0.15, 0.75, 0.0, False, False],  # Right
            # 10: [0.15, 0.50, 0.0, False, False],  # Right
            # 11: [0.15, 0.25, 0.0, False, False],  # Right
            # 12: [0.15, -0.75, 0.0, False, False],  # Left
            # 13: [0.15, -0.50, 0.0, False, False],  # Left
            # 14: [0.15, -0.25, 0.0, False, False],  # Left
            # 15: [0.3, 0.00, 0.0, False, False],  # Straight
            # 16: [0.3, 0.75, 0.0, False, False],  # Right
            # 17: [0.3, 0.50, 0.0, False, False],  # Right
            # 18: [0.3, 0.25, 0.0, False, False],  # Right
            # 19: [0.3, -0.75, 0.0, False, False],  # Left
            # 20: [0.3, -0.50, 0.0, False, False],  # Left
            # 21: [0.3, -0.25, 0.0, False, False],  # Left
            # 22: [0.7, 0.00, 0.0, False, False],  # Straight
            # 23: [0.7, 0.75, 0.0, False, False],  # Right
            # 24: [0.7, 0.50, 0.0, False, False],  # Right
            # 25: [0.7, 0.25, 0.0, False, False],  # Right
            # 26: [0.7, -0.75, 0.0, False, False],  # Left
            # 27: [0.7, -0.50, 0.0, False, False],  # Left
            # 28: [0.7, -0.25, 0.0, False, False],  # Left
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
        # print(f"----------------------------------->{action_msg}") if self.custom_enable_rendering else None

        self.last_action = action_control


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


        if core.config["truckTrailerCombo"]:
            # Getting trailer location
            trailer_transform = core.hero_trailer.get_transform()
            trailer_forward_vector = trailer_transform.get_forward_vector()

        forward_vector_waypoint_0 = core.route[core.last_waypoint_index + 0].get_forward_vector()
        forward_vector_waypoint_2 = core.route[core.last_waypoint_index + 2].get_forward_vector()
        forward_vector_waypoint_5 = core.route[core.last_waypoint_index + 5].get_forward_vector()
        forward_vector_waypoint_7 = core.route[core.last_waypoint_index + 7].get_forward_vector()
        forward_vector_waypoint_10 = core.route[core.last_waypoint_index + 10].get_forward_vector()

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



        lidar = False

        if lidar:
            number_of_waypoints_to_plot_on_lidar = 20
            location_from_waypoint_to_vehicle_relative = np.zeros([2, number_of_waypoints_to_plot_on_lidar])

            for i in range(0,number_of_waypoints_to_plot_on_lidar,2):
                try:
                    x_dist = core.route[core.last_waypoint_index + i].location.x - (truck_transform.location.x + 2)
                    y_dist = core.route[core.last_waypoint_index + i].location.y - (truck_transform.location.y + 0.10)

                    xr, yr = self.rotate_matrix(x_dist,y_dist,360-truck_transform.rotation.yaw,0,0,units="DEGREES")


                    location_from_waypoint_to_vehicle_relative[0][i] = xr
                    location_from_waypoint_to_vehicle_relative[1][i] = yr
                    location_from_waypoint_to_vehicle_relative[0][i+1] = xr + 0.1
                    location_from_waypoint_to_vehicle_relative[1][i+1] = yr + 0.1
                except Exception as e:
                    print("At end of route, nothing major wrong")

        # print(f"UP HERE{location_from_waypoint_to_vehicle_relative}")
        # import pickle
        # def save_data(filename, data):
        #     with open(filename, 'wb') as handle:
        #         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # save_data('waypoints2.pkl',location_from_waypoint_to_vehicle_relative)
        number_of_waypoints = len(core.route)

        closest_distance_to_next_waypoint_line = core.distToSegment(truck_transform=truck_transform,waypoint_plus_current=0)
        closest_distance_to_next_plus_1_waypoint_line = core.distToSegment(truck_transform=truck_transform,waypoint_plus_current=1)

        # Hyp distance to next waypoint
        x_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.x - truck_transform.location.x)
        y_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.y - truck_transform.location.y)
        hyp_distance_to_next_waypoint = math.sqrt((x_dist_to_next_waypoint) ** 2 + (y_dist_to_next_waypoint) ** 2)

        # Hyp distance to next waypoint +1
        x_dist_to_next_waypoint = abs(core.route[
                                          core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with + 1].location.x - truck_transform.location.x)
        y_dist_to_next_waypoint = abs(core.route[
                                          core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with + 1].location.y - truck_transform.location.y)
        hyp_distance_to_next_plus_1_waypoint = math.sqrt(
            (x_dist_to_next_waypoint) ** 2 + (y_dist_to_next_waypoint) ** 2)

        distance_to_center_of_lane = core.shortest_distance_to_center_of_lane(truck_transform=truck_transform)
        trailer_distance_to_center_of_lane = core.shortest_distance_to_center_of_lane(truck_transform=trailer_rear_axle_transform,waypoint_no=self.current_trailer_waypoint)
        self.current_trailer_distance_away_from_the_centre_of_the_lane = trailer_distance_to_center_of_lane

        self.trailerangle_to_center_of_lane_degrees = calculate_angle_with_center_of_lane(
            previous_position=core.route[self.current_trailer_waypoint-1].location,
            current_position=trailer_rear_axle_transform.location,
            next_position=core.route[self.current_trailer_waypoint + 1].location)

        truck_bearing_to_waypoint = angle_between(waypoint_forward_vector=forward_vector_waypoint_0,
                                                  vehicle_forward_vector=truck_forward_vector)

        truck_bearing_to_waypoint_2 = angle_between(waypoint_forward_vector=forward_vector_waypoint_2,
                                                    vehicle_forward_vector=truck_forward_vector)

        truck_bearing_to_waypoint_5 = angle_between(waypoint_forward_vector=forward_vector_waypoint_5,
                                                    vehicle_forward_vector=truck_forward_vector)

        truck_bearing_to_waypoint_7 = angle_between(waypoint_forward_vector=forward_vector_waypoint_7,
                                                    vehicle_forward_vector=truck_forward_vector)

        truck_bearing_to_waypoint_10 = angle_between(waypoint_forward_vector=forward_vector_waypoint_10,
                                                     vehicle_forward_vector=truck_forward_vector)

        # try:
        #     bearing_to_ahead_waypoints_ahead_2 = angle_between(waypoint_forward_vector=core.route[core.last_waypoint_index + ahead_waypoints_2].get_forward_vector(),vehicle_forward_vector=truck_forward_vector)
        # except Exception as e:
        #     print(f"ERROR HERE2 {e}")
        #     bearing_to_ahead_waypoints_ahead_2 = 0


        angle_between_truck_and_trailer = angle_between(waypoint_forward_vector=truck_forward_vector,vehicle_forward_vector=trailer_forward_vector)

        trailer_bearing_to_waypoint = angle_between(waypoint_forward_vector=forward_vector_waypoint_0,vehicle_forward_vector=trailer_forward_vector)

        trailer_bearing_to_waypoint_2 = angle_between(waypoint_forward_vector=forward_vector_waypoint_2,vehicle_forward_vector=trailer_forward_vector)

        trailer_bearing_to_waypoint_5 = angle_between(waypoint_forward_vector=forward_vector_waypoint_5,vehicle_forward_vector=trailer_forward_vector)

        trailer_bearing_to_waypoint_7 = angle_between(waypoint_forward_vector=forward_vector_waypoint_7,vehicle_forward_vector=trailer_forward_vector)

        trailer_bearing_to_waypoint_10 = angle_between(waypoint_forward_vector=forward_vector_waypoint_10,vehicle_forward_vector=trailer_forward_vector)



        forward_velocity = np.clip(self.get_speed(core.hero), 0, None)
        self.current_forward_velocity = forward_velocity
        # forward_velocity_x = np.clip(self.get_forward_velocity_x(core.hero), 0, None)
        # forward_velocity_z = np.clip(self.get_forward_velocity_z(core.hero), 0, None)
        # acceleration = np.clip(self.get_acceleration(core.hero), 0, None)

        def abs_clip_normalise(value, normalisation_value):
            clipped = np.clip(abs(value),0,normalisation_value)
            return clipped/normalisation_value

        if core.last_waypoint_index - 5 > 0:
            angle_between_waypoints_minus5 = calculate_angle_with_center_of_lane(
                previous_position=core.route[core.last_waypoint_index].location,
                current_position=core.route[core.last_waypoint_index - 5].location,
                next_position=core.route[core.last_waypoint_index + 5].location)
            angle_between_waypoints_minus5 = abs_clip_normalise(angle_between_waypoints_minus5, math.pi)
        else:
            angle_between_waypoints_minus5 = 1

        if core.last_waypoint_index - 7 > 0:
            angle_between_waypoints_minus7 = calculate_angle_with_center_of_lane(
                previous_position=core.route[core.last_waypoint_index].location,
                current_position=core.route[core.last_waypoint_index - 7].location,
                next_position=core.route[core.last_waypoint_index + 7].location)
            angle_between_waypoints_minus7 = abs_clip_normalise(angle_between_waypoints_minus7, math.pi)
        else:
            angle_between_waypoints_minus7 = 1

        if core.last_waypoint_index - 10 > 0:
            angle_between_waypoints_minus10 = calculate_angle_with_center_of_lane(
                previous_position=core.route[core.last_waypoint_index].location,
                current_position=core.route[core.last_waypoint_index - 10].location,
                next_position=core.route[core.last_waypoint_index + 10].location)
            angle_between_waypoints_minus10 = abs_clip_normalise(angle_between_waypoints_minus10, math.pi)
        else:
            angle_between_waypoints_minus10 = 1

        if core.last_waypoint_index - 12 > 0:
            angle_between_waypoints_minus12 = calculate_angle_with_center_of_lane(
                previous_position=core.route[core.last_waypoint_index].location,
                current_position=core.route[core.last_waypoint_index - 12].location,
                next_position=core.route[core.last_waypoint_index + 12].location)
            angle_between_waypoints_minus12 = abs_clip_normalise(angle_between_waypoints_minus12, math.pi)
        else:
            angle_between_waypoints_minus12 = 1

        angle_between_waypoints_5 = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index+5].location,
            current_position=core.route[core.last_waypoint_index].location,
            next_position=core.route[core.last_waypoint_index + 10].location)
        angle_between_waypoints_5 = abs_clip_normalise(angle_between_waypoints_5,math.pi)

        angle_between_waypoints_7 = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index+7].location,
            current_position=core.route[core.last_waypoint_index].location,
            next_position=core.route[core.last_waypoint_index + 14].location)
        angle_between_waypoints_7 = abs_clip_normalise(angle_between_waypoints_7,math.pi)


        if len(core.route) > core.last_waypoint_index + 20:
            angle_between_waypoints_10 = calculate_angle_with_center_of_lane(
                previous_position=core.route[core.last_waypoint_index + 10].location,
                current_position=core.route[core.last_waypoint_index].location,
                next_position=core.route[core.last_waypoint_index + 20].location)
            angle_between_waypoints_10 = abs_clip_normalise(angle_between_waypoints_10, math.pi)

        else:
            angle_between_waypoints_10 = 1

        if len(core.route) > core.last_waypoint_index + 24:
            angle_between_waypoints_12 = calculate_angle_with_center_of_lane(
                previous_position=core.route[core.last_waypoint_index + 12].location,
                current_position=core.route[core.last_waypoint_index].location,
                next_position=core.route[core.last_waypoint_index + 24].location)
            angle_between_waypoints_12 = abs_clip_normalise(angle_between_waypoints_12, math.pi)

        else:
            angle_between_waypoints_12 = 1

        # Angle to center of lane

        angle_to_center_of_lane_degrees = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index-1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location)

        angle_to_center_of_lane_degrees_2 = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index - 1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + 2].location)

        angle_to_center_of_lane_degrees_5 = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index - 1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + 5].location)

        angle_to_center_of_lane_degrees_7 = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index - 1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + 7].location)

        angle_to_center_of_lane_degrees_ahead_waypoints = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index-1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + ahead_waypoints].location)
        # try:
        #     angle_to_center_of_lane_degrees_ahead_waypoints_2 = calculate_angle_with_center_of_lane(
        #         previous_position=core.route[core.last_waypoint_index-2].location,
        #         current_position=truck_transform.location,
        #         next_position=core.route[core.last_waypoint_index + ahead_waypoints_2].location)
        # except Exception as e:
        #     print(f"ERROR HERE3 {e}")
        #     angle_to_center_of_lane_degrees_ahead_waypoints_2 = 0

        if self.visualiseRoute and self.counter > 2 :
            draw_route_in_order(route=core.route)
            plot_route(route=core.route, last_waypoint_index=core.last_waypoint_index, truck_transform=truck_transform, number_of_waypoints_ahead_to_calculate_with=5)

            print(f"previous_position={core.route[core.last_waypoint_index-1].location}")
            print(f"current_position={truck_transform.location}")
            print(f"next_position={core.route[core.last_waypoint_index+number_of_waypoints_ahead_to_calculate_with].location}")
            print(f"current_waypoint={core.route[core.last_waypoint_index].location}")
            print(f"next_waypoint={core.route[core.last_waypoint_index+1].location}")
            print(f"in_front_of_waypoint={in_front_of_waypoint}")
            print(f"angle={angle_to_center_of_lane_degrees}")

            plot_points(previous_position=core.route[core.last_waypoint_index-1].location,
                        current_position=truck_transform.location,
                        next_position=core.route[core.last_waypoint_index+number_of_waypoints_ahead_to_calculate_with].location,
                        current_waypoint=core.route[core.last_waypoint_index].location,
                        next_waypoint=core.route[core.last_waypoint_index+1].location,
                        in_front_of_waypoint=in_front_of_waypoint,
                        angle=angle_to_center_of_lane_degrees)

        if self.traffic:
            # For the 5 nearest vehicles get the:
            # Relative x and y positions
            # Rotation
            # Acceleration
            # Velocity

            self.traffic_observations = []
            max_no_of_vehicles = 2
            # First find the nearest 5 vehicles
            distance_to_truck = []
            for actor in core.actors:
                actor_transform = actor.get_transform()
                distance_to_truck.append(truck_transform.location.distance(actor_transform.location))

            indices_of_shortest_distances = sorted(range(len(distance_to_truck)), key=lambda k: distance_to_truck[k])

            for i, index_of_closest_actor in enumerate(indices_of_shortest_distances):
                if i == max_no_of_vehicles:
                    break
                actor_transform = core.actors[index_of_closest_actor].get_transform()

                actor_velocity = self.get_speed(core.actors[index_of_closest_actor])
                actor_acceleration = self.get_acceleration(core.actors[index_of_closest_actor])
                actor_yaw = np.clip(actor_transform.rotation.yaw * 0.0174533,-math.pi, math.pi)  # to convert to radians
                actor_relative_x = truck_transform.location.x - actor_transform.location.x
                actor_relative_y = truck_transform.location.y - actor_transform.location.y

                self.traffic_observations.extend([actor_velocity,actor_acceleration,actor_yaw,actor_relative_x,actor_relative_y])

            if len(self.traffic_observations) < max_no_of_vehicles * 5:
                for i in range((max_no_of_vehicles * 5) - len(self.traffic_observations)):
                    self.traffic_observations.append(0)

        depth_camera_data = None
        current_occupancy_map = None
        trailer_0_left_sidewalk = 0
        trailer_0_right_sidewalk = 0
        trailer_1_left_sidewalk = 0
        trailer_1_right_sidewalk = 0
        trailer_2_left_sidewalk = 0
        trailer_2_right_sidewalk = 0
        trailer_3_left_sidewalk = 0
        trailer_3_right_sidewalk = 0
        trailer_4_left_sidewalk = 0
        trailer_4_right_sidewalk = 0
        trailer_5_left_sidewalk = 0
        trailer_5_right_sidewalk = 0
        trailer_6_left_sidewalk = 0
        trailer_6_right_sidewalk = 0
        trailer_7_left_sidewalk = 0
        trailer_7_right_sidewalk = 0

        if self.traffic:
            trailer_0_left_vehicle = 0
            trailer_0_right_vehicle = 0
            trailer_1_left_vehicle = 0
            trailer_1_right_vehicle = 0
            trailer_2_left_vehicle = 0
            trailer_2_right_vehicle = 0
            trailer_3_left_vehicle = 0
            trailer_3_right_vehicle = 0
            trailer_4_left_vehicle = 0
            trailer_4_right_vehicle = 0
            trailer_5_left_vehicle = 0
            trailer_5_right_vehicle = 0
            trailer_6_left_vehicle = 0
            trailer_6_right_vehicle = 0
            trailer_7_left_vehicle = 0
            trailer_7_right_vehicle = 0

        truck_center_sidewalk = 0
        truck_right_sidewalk = 0
        truck_left_sidewalk = 0
        truck_front_15right_sidewalk = 0
        truck_front_30right_sidewalk = 0
        truck_front_45right_sidewalk = 0
        truck_front_60right_sidewalk = 0
        truck_front_75right_sidewalk = 0
        truck_front_15left_sidewalk = 0
        truck_front_30left_sidewalk = 0
        truck_front_45left_sidewalk = 0
        truck_front_60left_sidewalk = 0
        truck_front_75left_sidewalk = 0

        if self.traffic:
            truck_center_vehicle = 0
            truck_right_vehicle = 0
            truck_left_vehicle = 0
            truck_front_15right_vehicle = 0
            truck_front_30right_vehicle = 0
            truck_front_45right_vehicle = 0
            truck_front_60right_vehicle = 0
            truck_front_75right_vehicle = 0
            truck_front_15left_vehicle = 0
            truck_front_30left_vehicle = 0
            truck_front_45left_vehicle = 0
            truck_front_60left_vehicle = 0
            truck_front_75left_vehicle = 0

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

            elif sensor == "depth_camera_truck":
                depth_camera_data = sensor_data['depth_camera_truck'][1]
                #
                # img = Image.fromarray(depth_camera_data, None)
                # img.show()
                # time.sleep(0.005)
                # img.close()

                # print(depth_camera_data.shape)

                assert depth_camera_data is not None

            elif sensor == "lidar_trailer_0_left_trailer":
                lidar_points = sensor_data['lidar_trailer_0_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_0_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_0_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_0_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)


            elif sensor == "lidar_trailer_0_right_trailer":
                lidar_points = sensor_data['lidar_trailer_0_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_0_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_0_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_0_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_1_left_trailer":
                lidar_points = sensor_data['lidar_trailer_1_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_1_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_1_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_1_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_1_right_trailer":
                lidar_points = sensor_data['lidar_trailer_1_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_1_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_1_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_1_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_2_left_trailer":
                lidar_points = sensor_data['lidar_trailer_2_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_2_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_2_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_2_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_2_right_trailer":
                lidar_points = sensor_data['lidar_trailer_2_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_2_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_2_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_2_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_3_left_trailer":
                lidar_points = sensor_data['lidar_trailer_3_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_3_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_3_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_3_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_3_right_trailer":
                lidar_points = sensor_data['lidar_trailer_3_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_3_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_3_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_3_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_4_left_trailer":
                lidar_points = sensor_data['lidar_trailer_4_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_4_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_4_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_4_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_4_right_trailer":
                lidar_points = sensor_data['lidar_trailer_4_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_4_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_4_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_4_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_5_left_trailer":
                lidar_points = sensor_data['lidar_trailer_5_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_5_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_5_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_5_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_5_right_trailer":
                lidar_points = sensor_data['lidar_trailer_5_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_5_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_5_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_5_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)
            elif sensor == "lidar_trailer_6_left_trailer":
                lidar_points = sensor_data['lidar_trailer_6_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_6_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_6_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_6_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_6_right_trailer":
                lidar_points = sensor_data['lidar_trailer_6_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_6_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_6_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_6_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_7_left_trailer":
                lidar_points = sensor_data['lidar_trailer_7_left_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_7_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_7_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_7_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_trailer_7_right_trailer":
                lidar_points = sensor_data['lidar_trailer_7_right_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer_7_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    trailer_7_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                trailer_7_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)
            elif sensor == "lidar_truck_right_truck":
                lidar_points = sensor_data['lidar_truck_right_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_left_truck":
                lidar_points = sensor_data['lidar_truck_left_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_center_truck":
                lidar_points = sensor_data['lidar_truck_center_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_center"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_center_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)


                truck_center_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_15left_truck":
                lidar_points = sensor_data['lidar_truck_front_15left_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_15left"]["range"])

                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_15left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_15left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_30left_truck":
                lidar_points = sensor_data['lidar_truck_front_30left_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_30left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_30left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_30left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_45left_truck":
                lidar_points = sensor_data['lidar_truck_front_45left_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_45left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_45left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_45left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_60left_truck":
                lidar_points = sensor_data['lidar_truck_front_60left_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_60left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_60left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_60left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_75left_truck":
                lidar_points = sensor_data['lidar_truck_front_75left_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_75left"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_75left_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_75left_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_15right_truck":
                lidar_points = sensor_data['lidar_truck_front_15right_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_15right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_15right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_15right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_30right_truck":
                lidar_points = sensor_data['lidar_truck_front_30right_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_30right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_30right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_30right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_45right_truck":
                lidar_points = sensor_data['lidar_truck_front_45right_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_45right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_45right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_45right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_60right_truck":
                lidar_points = sensor_data['lidar_truck_front_60right_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_60right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_60right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_60right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_front_75right_truck":
                lidar_points = sensor_data['lidar_truck_front_75right_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_front_75right"]["range"])
                sidewalk_lidar_points = lidar_points[0]
                if self.traffic:
                    sidewalk_lidar_points, vehicle_lidar_points = self.get_sidewalk_vehicle_lidar_points(lidar_points)
                    truck_front_75right_vehicle = self.get_min_lidar_point(vehicle_lidar_points, lidar_range)

                truck_front_75right_sidewalk = self.get_min_lidar_point(sidewalk_lidar_points, lidar_range)

            elif sensor == "lidar_truck_truck":
                lidar_points = sensor_data['lidar_truck_truck'][1]
                # print(lidar_points.shape)
                # print(f"BEFORE {lidar_points[0][len(lidar_points) - 20]}-{lidar_points[1][len(lidar_points) - 20]}")
                # print(f"BEFORE {lidar_points[0][len(lidar_points) - 19]}-{lidar_points[1][len(lidar_points) - 19]}")
                # print(f"BEFORE {lidar_points[0][len(lidar_points) - 18]}-{lidar_points[1][len(lidar_points) - 18]}")
                # print(f"BEFORE {lidar_points[0][len(lidar_points) - 17]}-{lidar_points[1][len(lidar_points) - 17]}")
                #
                lidar_points = np.append(lidar_points, location_from_waypoint_to_vehicle_relative,axis=1)
                #
                # print(f"AFTER {lidar_points[0][len(lidar_points) - 1]}-{lidar_points[1][len(lidar_points) - 1]}")
                # print(f"AFTER {lidar_points[0][len(lidar_points) - 2]}-{lidar_points[1][len(lidar_points) - 2]}")
                # print(f"AFTER {lidar_points[0][len(lidar_points) - 3]}-{lidar_points[1][len(lidar_points) - 3]}")
                # print(f"AFTER {lidar_points[0][len(lidar_points) - 4]}-{lidar_points[1][len(lidar_points) - 4]}")

                xy_resolution = 0.2

                ox = lidar_points[0][:]
                oy = lidar_points[1][:]

                current_occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = \
                    generate_ray_casting_grid_map(ox=ox, oy=oy, x_output=self.occupancy_map_x, y_output=self.occupancy_map_y,
                                                  xy_resolution=xy_resolution, breshen=True)
                # Inverted the image as a test
                # occupancy_map = occupancy_map[::-1]
                # print(f"Final image size {occupancy_map.shape}")
                # print(f"HER1{current_occupancy_map.shape}")

                self.occupancy_maps.append(current_occupancy_map)

                if self.visualiseOccupancyGirdMap and self.counter % 10 == 0 :
                    multiple_lidars = True
                    if multiple_lidars:
                        # plt.figure()
                        f, axarr = plt.subplots(1, 2)
                        axarr[0].imshow(self.occupancy_maps[0])
                        axarr[1].imshow(self.occupancy_maps[10])
                        # axarr[1].imshow(self.occupancy_maps[10])
                        xy_res = np.array(current_occupancy_map).shape
                        # plt.imshow(occupancy_map, cmap="PiYG_r")
                        # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
                        # plt.clim(-0.4, 1.4)
                        # plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
                        # plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
                        # plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
                        # plt.gca().invert_yaxis()
                        plt.show()
                    else:
                        plt.figure()
                        xy_res = np.array(current_occupancy_map).shape
                        plt.imshow(occupancy_map, cmap="PiYG_r")
                        # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
                        plt.clim(-0.4, 1.4)
                        plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
                        plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
                        plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
                        # plt.gca().invert_yaxis()
                        plt.show()

                assert current_occupancy_map is not None

        if self.visualiseImage and self.counter > self.counterThreshold:
            plt.imshow(depth_camera_data, interpolation='nearest')
            plt.show()

        # Only saving the last lidar data points
        self.lidar_data.append([
            self.current_time,
            np.float32(trailer_0_left_sidewalk),
            np.float32(trailer_0_right_sidewalk),
            np.float32(trailer_1_left_sidewalk),
            np.float32(trailer_1_right_sidewalk),
            np.float32(trailer_2_left_sidewalk),
            np.float32(trailer_2_right_sidewalk),
            np.float32(trailer_3_left_sidewalk),
            np.float32(trailer_3_right_sidewalk),
            np.float32(trailer_4_left_sidewalk),
            np.float32(trailer_4_right_sidewalk),
            np.float32(trailer_5_left_sidewalk),
            np.float32(trailer_5_right_sidewalk),
            np.float32(trailer_6_left_sidewalk),
            np.float32(trailer_6_right_sidewalk),
            np.float32(trailer_7_left_sidewalk),
            np.float32(trailer_7_right_sidewalk),
            np.float32(truck_center_sidewalk),
            np.float32(truck_right_sidewalk),
            np.float32(truck_left_sidewalk),
            np.float32(truck_front_15left_sidewalk),
            np.float32(truck_front_30left_sidewalk),
            np.float32(truck_front_45left_sidewalk),
            np.float32(truck_front_60left_sidewalk),
            np.float32(truck_front_75left_sidewalk),
            np.float32(truck_front_15right_sidewalk),
            np.float32(truck_front_30right_sidewalk),
            np.float32(truck_front_45right_sidewalk),
            np.float32(truck_front_60right_sidewalk),
            np.float32(truck_front_75right_sidewalk),
        ])

        if self.traffic:

            self.lidar_data.extend([
                np.float32(trailer_0_left_vehicle),
                np.float32(trailer_0_right_vehicle),
                np.float32(trailer_1_left_vehicle),
                np.float32(trailer_1_right_vehicle),
                np.float32(trailer_2_left_vehicle),
                np.float32(trailer_2_right_vehicle),
                np.float32(trailer_3_left_vehicle),
                np.float32(trailer_3_right_vehicle),
                np.float32(trailer_4_left_vehicle),
                np.float32(trailer_4_right_vehicle),
                np.float32(trailer_5_left_vehicle),
                np.float32(trailer_5_right_vehicle),
                np.float32(trailer_6_left_vehicle),
                np.float32(trailer_6_right_vehicle),
                np.float32(trailer_7_left_vehicle),
                np.float32(trailer_7_right_vehicle),
                np.float32(truck_center_vehicle),
                np.float32(truck_right_vehicle),
                np.float32(truck_left_vehicle),
                np.float32(truck_front_15left_vehicle),
                np.float32(truck_front_30left_vehicle),
                np.float32(truck_front_45left_vehicle),
                np.float32(truck_front_60left_vehicle),
                np.float32(truck_front_75left_vehicle),
                np.float32(truck_front_15right_vehicle),
                np.float32(truck_front_30right_vehicle),
                np.float32(truck_front_45right_vehicle),
                np.float32(truck_front_60right_vehicle),
                np.float32(truck_front_75right_vehicle)
            ])

        value_observations = [
            # np.float32(number_of_waypoints),
            # np.float32(core.last_waypoint_index/number_of_waypoints),
            np.float32(forward_velocity),
            # np.float32(acceleration),
            # np.float32(forward_veloci ty_x),
            # np.float32(forward_velocity_z),
            np.float32(hyp_distance_to_next_waypoint),
            np.float32(hyp_distance_to_next_plus_1_waypoint),
            np.float32(closest_distance_to_next_waypoint_line),
            np.float32(closest_distance_to_next_plus_1_waypoint_line),
            np.float32(distance_to_center_of_lane),

            # np.float32(hyp_distance_to_next_waypoint_line),
            np.float32(angle_to_center_of_lane_degrees),
            np.float32(angle_to_center_of_lane_degrees_2),
            np.float32(angle_to_center_of_lane_degrees_5),
            np.float32(angle_to_center_of_lane_degrees_7),
            np.float32(angle_to_center_of_lane_degrees_ahead_waypoints),
            np.float32(truck_bearing_to_waypoint),
            np.float32(truck_bearing_to_waypoint_2),
            np.float32(truck_bearing_to_waypoint_5),
            np.float32(truck_bearing_to_waypoint_7),
            np.float32(truck_bearing_to_waypoint_10),
            np.float32(trailer_bearing_to_waypoint),
            np.float32(trailer_bearing_to_waypoint_2),
            np.float32(trailer_bearing_to_waypoint_5),
            np.float32(trailer_bearing_to_waypoint_7),
            np.float32(trailer_bearing_to_waypoint_10),
            # np.float32(bearing_to_ahead_waypoints_ahead_2),
            np.float32(angle_between_truck_and_trailer),
            np.float32(angle_between_waypoints_5),
            np.float32(angle_between_waypoints_7),
            np.float32(angle_between_waypoints_10),
            np.float32(angle_between_waypoints_12),
            np.float32(angle_between_waypoints_minus5),
            np.float32(angle_between_waypoints_minus7),
            np.float32(angle_between_waypoints_minus10),
            np.float32(angle_between_waypoints_minus12),

            # np.float32(trailer_bearing_to_waypoint),
            # np.float32(acceleration)
                           ]
        trailer_lidar_data_points = [
            np.float32(trailer_0_left_sidewalk),
            np.float32(trailer_0_right_sidewalk),
            np.float32(trailer_1_left_sidewalk),
            np.float32(trailer_1_right_sidewalk),
            np.float32(trailer_2_left_sidewalk),
            np.float32(trailer_2_right_sidewalk),
            np.float32(trailer_3_left_sidewalk),
            np.float32(trailer_3_right_sidewalk),
            np.float32(trailer_4_left_sidewalk),
            np.float32(trailer_4_right_sidewalk),
            np.float32(trailer_5_left_sidewalk),
            np.float32(trailer_5_right_sidewalk),
            np.float32(trailer_6_left_sidewalk),
            np.float32(trailer_6_right_sidewalk),
            np.float32(trailer_7_left_sidewalk),
            np.float32(trailer_7_right_sidewalk),


        ]

        if self.traffic:
            trailer_lidar_data_points.extend([
                np.float32(trailer_0_left_vehicle),
                np.float32(trailer_0_right_vehicle),
                np.float32(trailer_1_left_vehicle),
                np.float32(trailer_1_right_vehicle),
                np.float32(trailer_2_left_vehicle),
                np.float32(trailer_2_right_vehicle),
                np.float32(trailer_3_left_vehicle),
                np.float32(trailer_3_right_vehicle),
                np.float32(trailer_4_left_vehicle),
                np.float32(trailer_4_right_vehicle),
                np.float32(trailer_5_left_vehicle),
                np.float32(trailer_5_right_vehicle),
                np.float32(trailer_6_left_vehicle),
                np.float32(trailer_6_right_vehicle),
                np.float32(trailer_7_left_vehicle),
                np.float32(trailer_7_right_vehicle)
            ])


        truck_lidar_data_points = [
            np.float32(truck_center_sidewalk),
            np.float32(truck_right_sidewalk),
            np.float32(truck_left_sidewalk),
            np.float32(truck_front_15left_sidewalk),
            np.float32(truck_front_30left_sidewalk),
            np.float32(truck_front_45left_sidewalk),
            np.float32(truck_front_60left_sidewalk),
            np.float32(truck_front_75left_sidewalk),
            np.float32(truck_front_15right_sidewalk),
            np.float32(truck_front_30right_sidewalk),
            np.float32(truck_front_45right_sidewalk),
            np.float32(truck_front_60right_sidewalk),
            np.float32(truck_front_75right_sidewalk)
        ]

        if self.traffic:
            truck_lidar_data_points.extend([
                np.float32(truck_center_vehicle),
                np.float32(truck_right_vehicle),
                np.float32(truck_left_vehicle),
                np.float32(truck_front_15left_vehicle),
                np.float32(truck_front_30left_vehicle),
                np.float32(truck_front_45left_vehicle),
                np.float32(truck_front_60left_vehicle),
                np.float32(truck_front_75left_vehicle),
                np.float32(truck_front_15right_vehicle),
                np.float32(truck_front_30right_vehicle),
                np.float32(truck_front_45right_vehicle),
                np.float32(truck_front_60right_vehicle),
                np.float32(truck_front_75right_vehicle)
            ])
        value_observations.extend(trailer_lidar_data_points)
        value_observations.extend(truck_lidar_data_points)
        # value_observations.extend([self.last_action[0],self.last_action[1],self.last_action[2]])

        value_observations.extend(radii)

        if self.traffic:
            value_observations.extend(self.traffic_observations)
        # value_observations.append(np.float32(mean_radius))

        # route_type_string = get_route_type(current_entry_idx=self.entry_idx, current_exit_idx=self.exit_idx)
        #
        # if route_type_string == 'easy':
        #     route_type = 0
        # elif route_type_string == 'difficult':
        #     route_type = 1
        # else:
        #     raise Exception('This should never happen')

        self.truck_lidar_collision = False
        if any(lidar_point < 0.01 for lidar_point in truck_lidar_data_points):
            self.truck_lidar_collision = True

        self.trailer_lidar_collision = False
        if any(lidar_point < 0.01 for lidar_point in trailer_lidar_data_points):
            self.trailer_lidar_collision = True



        if False:
            print(f'Entry Points {core.entry_spawn_point_index}| Exit point {core.exit_spawn_point_index}')
            # print(f'Route Type {route_type}')
            # print(f"Radii {radii}")
            # print(f'Mean radius {mean_radius}')
            # print(f"truck FRONT \t\t\t{round(truck_center_sidewalk, 2)}")
            # print(f"truck 45 \t\t{round(truck_front_left,2)}\t\t{round(truck_front_right,2)}")
            # print(f"truck sides \t\t{round(truck_left_sidewalk, 2)}\t\t{round(truck_right_sidewalk, 2)}")
            # print(f"")
            # print(f"trailer_0 \t\t{round(trailer_0_left_sidewalk,2)}\t\t{round(trailer_0_right_sidewalk,2)}")
            # print(f"trailer_1 \t\t{round(trailer_1_left_sidewalk,2)}\t\t{round(trailer_2_right_sidewalk,2)}")
            # print(f"trailer_2 \t\t{round(trailer_2_left_sidewalk,2)}\t\t{round(trailer_2_right_sidewalk,2)}")
            # print(f"trailer_3 \t\t{round(trailer_3_left_sidewalk,2)}\t\t{round(trailer_3_right_sidewalk,2)}")
            # print(f"trailer_4 \t\t{round(trailer_4_left_sidewalk,2)}\t\t{round(trailer_4_right_sidewalk,2)}")
            # print(f"trailer_5 \t\t{round(trailer_5_left_sidewalk,2)}\t\t{round(trailer_5_right_sidewalk,2)}")
            print('Sidewalk lidar information')
            print(f"truck FRONT \t\t\t{np.float32(truck_center_sidewalk)}")
            print(f"truck 15 \t\t{np.float32(truck_front_15left_sidewalk)}\t\t{np.float32(truck_front_15right_sidewalk)}")
            print(f"truck 30 \t\t{np.float32(truck_front_30left_sidewalk)}\t\t{np.float32(truck_front_30right_sidewalk)}")
            print(f"truck 45 \t\t{np.float32(truck_front_45left_sidewalk)}\t\t{np.float32(truck_front_45right_sidewalk)}")
            print(f"truck 60 \t\t{np.float32(truck_front_60left_sidewalk)}\t\t{np.float32(truck_front_60right_sidewalk)}")
            print(f"truck 75 \t\t{np.float32(truck_front_75left_sidewalk)}\t\t{np.float32(truck_front_75right_sidewalk)}")
            print(f"truck sides \t\t{np.float32(truck_left_sidewalk)}\t\t{np.float32(truck_right_sidewalk)}")
            print(f"")
            print(f"trailer_0 \t\t{np.float32(trailer_0_left_sidewalk)}\t\t{np.float32(trailer_0_right_sidewalk)}")
            print(f"trailer_1 \t\t{np.float32(trailer_1_left_sidewalk)}\t\t{np.float32(trailer_2_right_sidewalk)}")
            print(f"trailer_2 \t\t{np.float32(trailer_2_left_sidewalk)}\t\t{np.float32(trailer_2_right_sidewalk)}")
            print(f"trailer_3 \t\t{np.float32(trailer_3_left_sidewalk)}\t\t{np.float32(trailer_3_right_sidewalk)}")
            print(f"trailer_4 \t\t{np.float32(trailer_4_left_sidewalk)}\t\t{np.float32(trailer_4_right_sidewalk)}")
            print(f"trailer_5 \t\t{np.float32(trailer_5_left_sidewalk)}\t\t{np.float32(trailer_5_right_sidewalk)}")
            if self.traffic:
                print('Vehicle lidar Information')
                print(f"truck FRONT \t\t\t{np.float32(truck_center_vehicle)}")
                print(
                    f"truck 15 \t\t{np.float32(truck_front_15left_vehicle)}\t\t{np.float32(truck_front_15right_vehicle)}")
                print(
                    f"truck 30 \t\t{np.float32(truck_front_30left_vehicle)}\t\t{np.float32(truck_front_30right_vehicle)}")
                print(
                    f"truck 45 \t\t{np.float32(truck_front_45left_vehicle)}\t\t{np.float32(truck_front_45right_vehicle)}")
                print(
                    f"truck 60 \t\t{np.float32(truck_front_60left_vehicle)}\t\t{np.float32(truck_front_60right_vehicle)}")
                print(
                    f"truck 75 \t\t{np.float32(truck_front_75left_vehicle)}\t\t{np.float32(truck_front_75right_vehicle)}")
                print(f"truck sides \t\t{np.float32(truck_left_vehicle)}\t\t{np.float32(truck_right_vehicle)}")
                print(f"")
                print(f"trailer_0 \t\t{np.float32(trailer_0_left_vehicle)}\t\t{np.float32(trailer_0_right_vehicle)}")
                print(f"trailer_1 \t\t{np.float32(trailer_1_left_vehicle)}\t\t{np.float32(trailer_2_right_vehicle)}")
                print(f"trailer_2 \t\t{np.float32(trailer_2_left_vehicle)}\t\t{np.float32(trailer_2_right_vehicle)}")
                print(f"trailer_3 \t\t{np.float32(trailer_3_left_vehicle)}\t\t{np.float32(trailer_3_right_vehicle)}")
                print(f"trailer_4 \t\t{np.float32(trailer_4_left_vehicle)}\t\t{np.float32(trailer_4_right_vehicle)}")
                print(f"trailer_5 \t\t{np.float32(trailer_5_left_vehicle)}\t\t{np.float32(trailer_5_right_vehicle)}")
            print('')
            print(f"forward_velocity:{np.float32(forward_velocity)}")
            # print(f"hyp_distance_to_next_waypoint:{np.float32(hyp_distance_to_next_waypoint)}")
            # print(f"hyp_distance_to_next_plus_1_waypoint:{np.float32(hyp_distance_to_next_plus_1_waypoint)}")
            # print(f"closest_distance_to_next_waypoint_line:{np.float32(closest_distance_to_next_waypoint_line)}")
            # print(f"closest_distance_to_next_plus_1_waypoint_line:{np.float32(closest_distance_to_next_plus_1_waypoint_line)}")
            # print(f"distance_to_center_of_lane:{np.float32(distance_to_center_of_lane)}")
            # print(f"angle_to_center_of_lane_degrees:{np.float32(angle_to_center_of_lane_degrees)}")
            # print(f"angle_to_center_of_lane_degrees_2:{np.float32(angle_to_center_of_lane_degrees_2)}")
            # print(f"angle_to_center_of_lane_degrees_5:{np.float32(angle_to_center_of_lane_degrees_5)}")
            # print(f"angle_to_center_of_lane_degrees_7:{np.float32(angle_to_center_of_lane_degrees_7)}")
            # print(f"angle_to_center_of_lane_degrees_ahead_waypoints:{np.float32(angle_to_center_of_lane_degrees_ahead_waypoints)}")
            # # print(f"angle_to_center_of_lane_degrees_ahead_waypoints_2:{np.float32(angle_to_center_of_lane_degrees_ahead_waypoints_2)}")
            # print(f"angle_between_waypoints_5:{np.float32(angle_between_waypoints_5)}")
            # print(f"angle_between_waypoints_7:{np.float32(angle_between_waypoints_7)}")
            # print(f"angle_between_waypoints_10:{np.float32(angle_between_waypoints_10)}")
            # print(f"angle_between_waypoints_12:{np.float32(angle_between_waypoints_12)}")
            # print(f"angle_between_waypoints_minus5:{np.float32(angle_between_waypoints_minus5)}")
            # print(f"angle_between_waypoints_minus7:{np.float32(angle_between_waypoints_minus7)}")
            # print(f"angle_between_waypoints_minus10:{np.float32(angle_between_waypoints_minus10)}")
            # print(f"angle_between_waypoints_minus12:{np.float32(angle_between_waypoints_minus12)}")
            # print(f"truck_bearing_to_waypoint:{np.float32(truck_bearing_to_waypoint)}")
            # print(f"truck_bearing_to_waypoint_2:{np.float32(truck_bearing_to_waypoint_2)}")
            # print(f"truck_bearing_to_waypoint_5:{np.float32(truck_bearing_to_waypoint_5)}")
            # print(f"truck_bearing_to_waypoint_7:{np.float32(truck_bearing_to_waypoint_7)}")
            # print(f"truck_bearing_to_waypoint_10:{np.float32(truck_bearing_to_waypoint_10)}")
            # print(f"trailer_bearing_to_waypoint:{np.float32(trailer_bearing_to_waypoint)}")
            # print(f"trailer_bearing_to_waypoint_2:{np.float32(trailer_bearing_to_waypoint_2)}")
            # print(f"trailer_bearing_to_waypoint_5:{np.float32(trailer_bearing_to_waypoint_5)}")
            # print(f"trailer_bearing_to_waypoint_7:{np.float32(trailer_bearing_to_waypoint_7)}")
            # print(f"trailer_bearing_to_waypoint_10:{np.float32(trailer_bearing_to_waypoint_10)}")
            # # print(f"bearing_to_ahead_waypoints_ahead_2:{np.float32(bearing_to_ahead_waypoints_ahead_2)}")
            # print(f"angle_between_truck_and_trailer:{np.float32(angle_between_truck_and_trailer)}")
            if self.traffic:
                for i in range(max_no_of_vehicles):
                    print(f'Vehicle {i}')
                    newI = i*5
                    print(f"Vehicle {i} Velcoity {self.traffic_observations[newI]}")
                    print(f"Vehicle {i} Acceleration {self.traffic_observations[newI+1]}")
                    print(f"Vehicle {i} Yaw {self.traffic_observations[newI+2]}")
                    print(f"Vehicle {i} Relative X {self.traffic_observations[newI+3]}")
                    print(f"Vehicle {i} Relative Y {self.traffic_observations[newI+4]}")

            print('')
            print('')
            time.sleep(0.04)
        self.forward_velocity.append(np.float32(forward_velocity))
        # self.forward_velocity_x.append(np.float32(forward_velocity_x))
        # self.forward_velocity_z.append(np.float32(forward_velocity_z))
        self.hyp_distance_to_next_waypoint.append(np.float32(hyp_distance_to_next_waypoint))
        self.hyp_distance_to_next_plus_1_waypoint.append(np.float32(hyp_distance_to_next_plus_1_waypoint))
        self.closest_distance_to_next_waypoint_line.append(np.float32(closest_distance_to_next_waypoint_line))
        self.closest_distance_to_next_plus_1_waypoint_line.append(np.float32(closest_distance_to_next_plus_1_waypoint_line))
        self.angle_to_center_of_lane_degrees.append(np.float32(angle_to_center_of_lane_degrees))
        self.angle_to_center_of_lane_degrees_2.append(np.float32(angle_to_center_of_lane_degrees_2))
        self.angle_to_center_of_lane_degrees_5.append(np.float32(angle_to_center_of_lane_degrees_5))
        self.angle_to_center_of_lane_degrees_7.append(np.float32(angle_to_center_of_lane_degrees_7))
        self.angle_to_center_of_lane_degrees_ahead_waypoints.append(np.float32(angle_to_center_of_lane_degrees_ahead_waypoints))
        # self.angle_to_center_of_lane_degrees_ahead_waypoints_2.append(np.float32(angle_to_center_of_lane_degrees_ahead_waypoints_2))
        self.angle_between_waypoints_5.append(np.float32(angle_between_waypoints_5))
        self.angle_between_waypoints_7.append(np.float32(angle_between_waypoints_7))
        self.angle_between_waypoints_10.append(np.float32(angle_between_waypoints_10))
        self.angle_between_waypoints_12.append(np.float32(angle_between_waypoints_12))
        self.angle_between_waypoints_minus5.append(np.float32(angle_between_waypoints_minus5))
        self.angle_between_waypoints_minus7.append(np.float32(angle_between_waypoints_minus7))
        self.angle_between_waypoints_minus10.append(np.float32(angle_between_waypoints_minus10))
        self.angle_between_waypoints_minus12.append(np.float32(angle_between_waypoints_minus12))
        self.truck_bearing_to_waypoint.append(np.float32(truck_bearing_to_waypoint))
        self.truck_bearing_to_waypoint_2.append(np.float32(truck_bearing_to_waypoint_2))
        self.truck_bearing_to_waypoint_5.append(np.float32(truck_bearing_to_waypoint_5))
        self.truck_bearing_to_waypoint_7.append(np.float32(truck_bearing_to_waypoint_7))
        self.truck_bearing_to_waypoint_10.append(np.float32(truck_bearing_to_waypoint_10))
        self.distance_to_center_of_lane.append(np.float32(distance_to_center_of_lane))

        self.trailer_bearing_to_waypoint.append(np.float32(trailer_bearing_to_waypoint))
        self.trailer_bearing_to_waypoint_2.append(np.float32(trailer_bearing_to_waypoint_2))
        self.trailer_bearing_to_waypoint_5.append(np.float32(trailer_bearing_to_waypoint_5))
        self.trailer_bearing_to_waypoint_7.append(np.float32(trailer_bearing_to_waypoint_7))
        self.trailer_bearing_to_waypoint_10.append(np.float32(trailer_bearing_to_waypoint_10))
        # self.bearing_to_ahead_waypoints_ahead_2.append(np.float32(bearing_to_ahead_waypoints_ahead_2))
        self.angle_between_truck_and_trailer.append(np.float32(angle_between_truck_and_trailer))
        # self.trailer_bearing_to_waypoint.append(np.float32(trailer_bearing_to_waypoint))
        # self.acceleration.append(np.float32(acceleration))
        self.vehicle_path.append((truck_transform.location.x,truck_transform.location.y))
        self.trailer_vehicle_path.append((trailer_rear_axle_transform.location.x,trailer_rear_axle_transform.location.y))
        self.temp_route = deepcopy(core.route_points)
        self.radii.append(radii)
        # self.mean_radius.append(mean_radius)
        #
        # print(f"angle_to_center_of_lane_degrees:{np.float32(angle_to_center_of_lane_degrees)}")
        # print(f"angle_to_center_of_lane_degrees_ahead_waypoints:{np.float32(angle_to_center_of_lane_degrees_ahead_waypoints)}")
        # print(f"angle_to_center_of_lane_degrees_ahead_waypoints_2:{np.float32(angle_to_center_of_lane_degrees_ahead_waypoints_2)}")
        # print(f"bearing_to_waypoint:{np.float32(bearing_to_waypoint)}")
        # print(f"bearing_to_ahead_waypoints_ahead:{np.float32(bearing_to_ahead_waypoints_ahead)}")
        # print(f"bearing_to_ahead_waypoints_ahead_2:{np.float32(bearing_to_ahead_waypoints_ahead_2)}")
        # print(f"hyp_distance_to_next_waypoint:{np.float32(hyp_distance_to_next_waypoint)}")

        # print(f"angle_between_truck_and_trailer:{np.float32(angle_between_truck_and_trailer)}")
        # print(f"trailer_bearing_to_waypoint:{np.float32(trailer_bearing_to_waypoint)}")
        # print(f"forward_velocity_x:{np.float32(forward_velocity_x)}")
        # print(f"forward_velocity_z:{np.float32(forward_velocity_z)}")
        # print(f"acceleration:{np.float32(acceleration)}")

        self.counter += 1
        return {'values':value_observations},\
            {"truck_z_value":truck_transform.location.z,"distance_to_center_of_lane":distance_to_center_of_lane, "truck_acceleration": self.get_acceleration(core.hero),'trailer_distance_to_center_of_lane':trailer_distance_to_center_of_lane}

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

    # def is_hero_near_finish_location(self, observation):
    #
    #     final_x = self.config["hero"]["final_location_x"]
    #     final_y = self.config["hero"]["final_location_y"]
    #
    #     if abs(observation[0] - final_x) < 0.5 and abs(observation[1] - final_y) < 0.5:
    #         return True

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

    def get_done_status(self, observation, core):
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

        if core.exit_spawn_point_index in [112,3]:
            acceptable_gap_to_next_waypoint = 20
        else:
            acceptable_gap_to_next_waypoint = 10

        self.done_far_from_path = self.last_hyp_distance_to_next_waypoint > acceptable_gap_to_next_waypoint

        output = self.done_time_idle or self.done_falling or self.done_time_episode or self.done_collision_truck or self.done_collision_trailer or self.done_arrived or self.done_far_from_path or self.truck_lidar_collision or self.trailer_lidar_collision
        self.custom_done_arrived = self.done_arrived

        done_status_info = {
            'done_time_idle':self.done_time_idle,
            'done_falling': self.done_falling,
            'done_time_episode':self.done_time_episode,
            'done_collision_truck': self.done_collision_truck,
            'done_collision_trailer':self.done_collision_trailer,
            'done_far_from_path':self.done_far_from_path,
            'done_arrived':self.done_arrived,
            'truck_lidar_collision':self.truck_lidar_collision,
            'trailer_lidar_collision':self.trailer_lidar_collision,
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
        if self.done_far_from_path:
            done_reason += "done_far_from_path"
        if self.done_arrived:
            done_reason += "done_arrived"
        if self.truck_lidar_collision:
            done_reason += "truck_lidar_collision"
        if self.trailer_lidar_collision:
            done_reason += "trailer_lidar_collision"

        if done_reason != "":
            data = f"ENTRY: {core.entry_spawn_point_index} EXIT: {core.exit_spawn_point_index} - {done_reason} \n"
            self.save_to_file(f"{self.directory}/done",data)

        return bool(output), done_status_info

    def compute_reward(self, observation, info, core):
        """Computes the reward"""
        # est
        reward = 0

        forward_velocity = observation['values'][0]
        hyp_distance_to_next_waypoint = observation['values'][1]
        hyp_distance_to_next_plus_1_waypoint = observation['values'][2]
        closest_distance_to_next_waypoint_line = observation['values'][3]
        closest_distance_to_next_plus_1_waypoint_line = observation['values'][4]
        distance_to_center_of_lane = observation['values'][5]

        # print(f"in rewards forward_velocity {forward_velocity}")
        # print(f"in rewards hyp_distance_to_next_waypoint {hyp_distance_to_next_waypoint}")

        #bearing_to_waypoint = observation[5]
        # bearing_to_ahead_waypoints_ahead = observation["values"][5]
        # angle_between_truck_and_trailer = observation["values"][6]


        # print(f"Hyp distance in rewards {hyp_distance_to_next_waypoint}")
        # print(f"self.last_hyp_distance_to_next_waypoint {self.last_hyp_distance_to_next_waypoint}")
        # print(f"self.last_hyp_distance_to_next_plus_1_waypoint {self.last_hyp_distance_to_next_plus_1_waypoint}")
        # print(f"Hyp distance line in rewards {hyp_distance_to_next_waypoint_line}")
        # print(f"self.last_hyp_distance_to_next_waypoint_lines {self.last_hyp_distance_to_next_waypoint_line}")
        # print(f"self.last_hyp_distance_to_next_plus_1_waypoint_line {self.last_hyp_distance_to_next_plus_1_waypoint_line}")

        # if self.last_hyp_distance_to_next_plus_1_waypoint == 0:
        #     self.last_hyp_distance_to_next_plus_1_waypoint = hyp_distance_to_next_waypoint
        #
        # # if self.last_hyp_distance_to_next_plus_1_waypoint_line == 0:
        # #     self.last_hyp_distance_to_next_plus_1_waypoint_line = hyp_distance_to_next_waypoint_line
        #
        # if self.last_closest_distance_to_next_plus_1_waypoint_line == 0:
        #     self.last_closest_distance_to_next_plus_1_waypoint_line = closest_distance_to_next_waypoint_line
        #
        # waypoint_reward_multiply_factor = 50
        # if self.last_hyp_distance_to_next_waypoint != 0:
        #     hyp_reward = self.last_hyp_distance_to_next_waypoint - hyp_distance_to_next_waypoint
        #     hyp_reward = np.clip(hyp_reward, None, 0.5)
        #     hyp_reward = hyp_reward - 0.5
        #     reward = reward + hyp_reward* waypoint_reward_multiply_factor
        #     self.point_reward.append(hyp_reward* waypoint_reward_multiply_factor)
        #     self.point_reward_location.append(1)
        #     print(f"REWARD hyp_distance_to_next_waypoint = {hyp_reward* waypoint_reward_multiply_factor}") if self.custom_enable_rendering else None
        # else:
        #     hyp_reward = self.last_hyp_distance_to_next_plus_1_waypoint - hyp_distance_to_next_waypoint
        #     hyp_reward = np.clip(hyp_reward, None, 0.5)
        #     hyp_reward = hyp_reward - 0.5
        #     reward = reward + hyp_reward * waypoint_reward_multiply_factor
        #     self.point_reward.append(hyp_reward* waypoint_reward_multiply_factor)
        #     self.point_reward_location.append(2)
        #     print(f"REWARD hyp_distance_to_next_waypoint = {hyp_reward* waypoint_reward_multiply_factor}") if self.custom_enable_rendering else None

        self.last_hyp_distance_to_next_waypoint = hyp_distance_to_next_waypoint
        # self.last_hyp_distance_to_next_plus_1_waypoint = hyp_distance_to_next_plus_1_waypoint

        # line_reward_multiply_factor = 100
        # if self.last_closest_distance_to_next_waypoint_line != 0:
        #     hyp_reward = self.last_closest_distance_to_next_waypoint_line - closest_distance_to_next_waypoint_line
        #     hyp_reward = np.clip(hyp_reward, None, 0.5)
        #     hyp_reward = hyp_reward - 0.5
        #     reward = reward + hyp_reward* line_reward_multiply_factor
        #     self.line_reward.append(hyp_reward* line_reward_multiply_factor)
        #     self.line_reward_location.append(1)
        #     print(f"REWARD closest_distance_to_next_waypoint_line = {hyp_reward* line_reward_multiply_factor}") if self.custom_enable_rendering else None
        # else:
        #     hyp_reward = self.last_closest_distance_to_next_plus_1_waypoint_line - closest_distance_to_next_waypoint_line
        #     hyp_reward = np.clip(hyp_reward, None, 0.5)
        #     hyp_reward = hyp_reward - 0.5
        #     reward = reward + hyp_reward * line_reward_multiply_factor
        #     self.line_reward.append(hyp_reward* line_reward_multiply_factor)
        #     self.line_reward_location.append(2)
        #     print(f"REWARD closest_distance_to_next_waypoint_line = {hyp_reward* line_reward_multiply_factor}") if self.custom_enable_rendering else None
        #
        # self.last_closest_distance_to_next_waypoint_line = closest_distance_to_next_waypoint_line
        # self.last_closest_distance_to_next_plus_1_waypoint_line = closest_distance_to_next_plus_1_waypoint_line

        if self.passed_waypoint:
            # reward = reward + 100
            reward = reward + 0.1
            pass

        distance_to_center_of_lane = (1/400) * (np.clip(abs(distance_to_center_of_lane),0,4))
        reward = reward - distance_to_center_of_lane

        # to encourage faster velocity
        # proportional_forward_velocity = (100/15) * (np.clip(abs(forward_velocity), 0, 15))
        # reward = reward + (proportional_forward_velocity)
        # print(f'Reward a {1/proportional_forward_velocity}')

        # for smooth velocity
        # difference_in_velocities = self.last_forward_velocity - forward_velocity
        # proportional_difference_in_velocities = (5/120) * (np.clip(abs(difference_in_velocities), 0, 1.2))
        # reward = reward - proportional_difference_in_velocities
        # self.last_forward_velocity = forward_velocity



        # if bearing_to_waypoint == 0:
        #      reward = reward+ 50
        # else:
        #     print(f"REWARD bearing_to_waypoint {abs(1/bearing_to_waypoint)}")
        #     reward = reward+ abs(1/bearing_to_waypoint)

        # if bearing_to_ahead_waypoints_ahead == 0:
        #     reward = reward + 30
        # else:
        #     reward_bearing_to_ahead_waypoints_ahead = abs(1 / bearing_to_ahead_waypoints_ahead)
        #     reward_bearing_to_ahead_waypoints_ahead = np.clip(reward_bearing_to_ahead_waypoints_ahead,0,30)
        #     print(f"REWARD bearing_to_ahead_waypoints_ahead {reward_bearing_to_ahead_waypoints_ahead}")
        #
        #     reward = reward + reward_bearing_to_ahead_waypoints_ahead

        # if forward_velocity < 0.75:
        #     # Negative reward for no velocity
        #     print('REWARD -100 for velocity') if self.custom_enable_rendering else None
        #     reward = reward + 0

        # Negative reward each timestep
        # reward = reward + -1


        if self.done_falling:
            reward = reward + -1
            print('====> REWARD Done falling')
        if self.done_collision_truck or self.done_collision_trailer:
            print("====> REWARD Done collision")
            reward = reward + -1
        if self.truck_lidar_collision:
            print("====> REWARD Truck Lidar collision")
            reward = reward + -1
        if self.trailer_lidar_collision:
            print("====> REWARD Trailer Lidar collision")
            reward = reward + -1
        if self.done_time_idle:
            print("====> REWARD Done idle")
            reward = reward + -1
        if self.done_time_episode:
            print("====> REWARD Done max time")
            reward = reward + -1
        if self.done_far_from_path:
            print("====> REWARD Done far from path")
            reward = reward + -1
        if self.done_arrived:
            print("====> REWARD Done arrived")
            reward = reward + 1

        self.total_episode_reward.append(reward)
        self.reward_metric = reward
        # print(f"FINAL REWARD: {reward}") if self.custom_enable_rendering else None
        # print(f"---------------------------------------------") if self.custom_enable_rendering else None
        return reward