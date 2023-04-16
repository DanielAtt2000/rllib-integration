#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import matplotlib.pyplot as plt
from git import Repo

import math
import numpy as np
from gym.spaces import Box, Discrete, Dict, Tuple
import warnings
import carla
import os
import time
from rllib_integration.GetAngle import calculate_angle_with_center_of_lane, angle_between
from rllib_integration.TestingWayPointUpdater import plot_points, plot_route
from rllib_integration.base_experiment import BaseExperiment
from rllib_integration.helper import post_process_image
from rllib_integration.Circle import get_radii
from PIL import Image

from rllib_integration.lidar_to_grid_map import generate_ray_casting_grid_map
import collections

class SACExperimentBasic(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_angle_with_center = 0
        self.last_forward_velocity = 0
        self.custom_done_arrived = False
        self.last_action = [0,0,0]
        self.lidar_points_count = []
        self.reward_metric = 0

        self.min_lidar_values = 1000000
        self.max_lidar_values = -100000
        self.lidar_max_points = self.config["hero"]["lidar_max_points"]
        self.counter = 0
        self.visualiseRoute = False
        self.visualiseImage = False
        self.visualiseOccupancyGirdMap = False
        self.counterThreshold = 10
        self.last_hyp_distance_to_next_waypoint = 0

        self.x_dist_to_waypoint = []
        self.y_dist_to_waypoint = []
        self.angle_to_center_of_lane_degrees = []
        self.angle_to_center_of_lane_degrees_ahead_waypoints = []
        self.angle_to_center_of_lane_degrees_ahead_waypoints_2 = []
        self.bearing_to_waypoint = []
        self.bearing_to_ahead_waypoints_ahead = []
        self.bearing_to_ahead_waypoints_ahead_2 = []
        self.angle_between_truck_and_trailer = []
        self.trailer_bearing_to_waypoint = []
        self.forward_velocity = []
        # self.forward_velocity_x = []
        # self.forward_velocity_z = []
        self.vehicle_path = []
        self.temp_route = []
        self.hyp_distance_to_next_waypoint = []
        # self.acceleration = []
        self.truck_collisions = []
        self.trailer_collisions =[]
        self.entry_idx = -1
        self.exit_idx = -1

        self.last_no_of_collisions_truck = 0
        self.last_no_of_collisions_trailer = 0

        self.occupancy_map_x = 84
        self.occupancy_map_y = 84
        self.max_amount_of_occupancy_maps = 11
        self.radii = []



        self.occupancy_maps = collections.deque(maxlen=self.max_amount_of_occupancy_maps)

        for i in range(self.max_amount_of_occupancy_maps):
            self.occupancy_maps.append(np.zeros((self.occupancy_map_y,self.occupancy_map_x,1)))


        repo = Repo('.')
        remote = repo.remote('origin')
        remote.fetch()
        self.directory = f"/home/daniel/data-rllib-integration/data/data_{str(repo.head.commit)[:11]}"

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def save_to_file(self, file_name, data):
        # Saving LIDAR point count
        counts = open(file_name, 'a')
        counts.write(str(data))
        counts.write(str('\n'))
        counts.close()

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
        self.reward_metric = 0
        self.radii = []

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

        # self.save_to_file(f"{self.directory}/hyp_distance_to_next_waypoint", self.hyp_distance_to_next_waypoint)
        # self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees", self.angle_to_center_of_lane_degrees)
        # self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_ahead_waypoints", self.angle_to_center_of_lane_degrees_ahead_waypoints)
        # self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_ahead_waypoints_2", self.angle_to_center_of_lane_degrees_ahead_waypoints_2)
        # self.save_to_file(f"{self.directory}/bearing", self.bearing_to_waypoint)
        # self.save_to_file(f"{self.directory}/bearing_ahead_ahead", self.bearing_to_ahead_waypoints_ahead)
        # self.save_to_file(f"{self.directory}/bearing_ahead_ahead_2", self.bearing_to_ahead_waypoints_ahead_2)
        # self.save_to_file(f"{self.directory}/angle_between_truck_and_trailer", self.angle_between_truck_and_trailer)
        # self.save_to_file(f"{self.directory}/trailer_bearing_to_waypoint", self.trailer_bearing_to_waypoint)
        # self.save_to_file(f"{self.directory}/forward_velocity", self.forward_velocity)
        # # self.save_to_file(f"{self.directory}/forward_velocity_x", self.forward_velocity_x)
        # # self.save_to_file(f"{self.directory}/forward_velocity_z", self.forward_velocity_z)
        # # self.save_to_file(f"{self.directory}/acceleration", self.acceleration)
        # self.save_to_file(f"{self.directory}/route", self.temp_route)
        # self.save_to_file(f"{self.directory}/path", self.vehicle_path)
        # self.save_to_file(f"{self.directory}/truck_collisions", self.truck_collisions)
        # self.save_to_file(f"{self.directory}/trailer_collisions", self.trailer_collisions)
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
        self.angle_to_center_of_lane_degrees_ahead_waypoints = []
        self.angle_to_center_of_lane_degrees_ahead_waypoints_2 = []
        self.bearing_to_waypoint = []
        self.bearing_to_ahead_waypoints_ahead = []
        self.bearing_to_ahead_waypoints_ahead_2 = []
        self.angle_between_truck_and_trailer = []
        self.trailer_bearing_to_waypoint = []
        self.forward_velocity = []
        # self.forward_velocity_x = []
        # self.forward_velocity_z = []
        self.vehicle_path = []
        self.temp_route = []
        self.hyp_distance_to_next_waypoint = []
        self.truck_collisions = []
        self.trailer_collisions =[]
        # self.acceleration = []





    # [33,28, 27, 17,  14, 11, 10, 5]

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        """
        Set observation space as location of vehicle im x,y starting at (0,0) and ending at (1,1)
        :return:
        """
        # test
        # image_space = Dict(
        #     {"values":
            # "depth_camera": Box(
            #     low=0,
            #     high=256,
            #     shape=(84, 84, 3),
            #     dtype=np.float32
            # ),
            # "occupancyMap_now": Box(
            #     low=0,
            #     high=1,
            #     shape=(self.occupancy_map_y, self.occupancy_map_x, 1),
            #     dtype=np.float64
            # ),
            # "occupancyMap_05": Box(
            #     low=0,
            #     high=1,
            #     shape=(self.occupancy_map_y, self.occupancy_map_x, 1),
            #     dtype=np.float64
            # ),
            # "occupancyMap_1": Box(
            #     low=0,
            #     high=1,
            #     shape=(self.occupancy_map_y, self.occupancy_map_x, 1),
            #     dtype=np.float64
            # )
            # })
        return Box(
                low=np.array([0,0,-math.pi,-math.pi,-math.pi,-math.pi,-math.pi,-math.pi,-math.pi,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,]),
                high=np.array([100,100,math.pi,math.pi,math.pi,math.pi,math.pi,math.pi,math.pi,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]),
                dtype=np.float32
            )

    def get_actions(self):
        return {
            0: [0.0, 0.00, 0.0, False, False],  # Coast
            1: [0.0, 0.00, 1.0, False, False],  # Apply Break
            2: [0.0, 0.75, 0.0, False, False],  # Right
            3: [0.0, 0.50, 0.0, False, False],  # Right
            4: [0.0, 0.25, 0.0, False, False],  # Right
            6: [0.0, -0.75, 0.0, False, False],  # Left
            7: [0.0, -0.50, 0.0, False, False],  # Left
            8: [0.0, -0.25, 0.0, False, False],  # Left
            10: [0.15, 0.00, 0.0, False, False],  # Straight
            # 11: [0.15, 0.80, 0.0, False, False],  # Right
            # 12: [0.15, 0.60, 0.0, False, False],  # Right
            # 13: [0.15, 0.40, 0.0, False, False],  # Right
            # 14: [0.15, 0.20, 0.0, False, False],  # Right
            # 15: [0.15, -0.80, 0.0, False, False],  # Left
            # 16: [0.15, -0.60, 0.0, False, False],  # Left
            # 17: [0.15, -0.40, 0.0, False, False],  # Left
            # 18: [0.15, -0.20, 0.0, False, False],  # Left
            # 19: [0.3, 0.00, 0.0, False, False],  # Straight
            # 20: [0.3, 0.80, 0.0, False, False],  # Right
            # 21: [0.3, 0.60, 0.0, False, False],  # Right
            # 22: [0.3, 0.40, 0.0, False, False],  # Right
            # 23: [0.3, 0.20, 0.0, False, False],  # Right
            # 24: [0.3, -0.80, 0.0, False, False],  # Left
            # 25: [0.3, -0.60, 0.0, False, False],  # Left
            # 26: [0.3, -0.40, 0.0, False, False],  # Left
            # 27: [0.3, -0.20, 0.0, False, False],  # Left
            # 28: [0.7, 0.00, 0.0, False, False],  # Straight
            # 29: [0.7, 0.80, 0.0, False, False],  # Right
            # 30: [0.7, 0.60, 0.0, False, False],  # Right
            # 31: [0.7, 0.40, 0.0, False, False],  # Right
            # 32: [0.7, 0.20, 0.0, False, False],  # Right
            # 33: [0.7, -0.80, 0.0, False, False],  # Left
            # 34: [0.7, -0.60, 0.0, False, False],  # Left
            # 35: [0.7, -0.40, 0.0, False, False],  # Left
            # 36: [0.7, -0.20, 0.0, False, False],  # Left
            11: [0.15, 0.25, 0.0, False, False],  # Right
            12: [0.15, -0.75, 0.0, False, False],  # Left
            13: [0.15, -0.50, 0.0, False, False],  # Left
            14: [0.15, -0.25, 0.0, False, False],  # Left
            15: [0.3, 0.00, 0.0, False, False],  # Straight
            16: [0.3, 0.75, 0.0, False, False],  # Right
            17: [0.3, 0.50, 0.0, False, False],  # Right
            18: [0.3, 0.25, 0.0, False, False],  # Right
            19: [0.3, -0.75, 0.0, False, False],  # Left
            20: [0.3, -0.50, 0.0, False, False],  # Left
            21: [0.3, -0.25, 0.0, False, False],  # Left
            22: [0.7, 0.00, 0.0, False, False],  # Straight
            23: [0.7, 0.75, 0.0, False, False],  # Right
            24: [0.7, 0.50, 0.0, False, False],  # Right
            25: [0.7, 0.25, 0.0, False, False],  # Right
            26: [0.7, -0.75, 0.0, False, False],  # Left
            27: [0.7, -0.50, 0.0, False, False],  # Left
            28: [0.7, -0.25, 0.0, False, False],  # Left
            # 29: [1.0, 0.00, 0.0, False, False],  # Straight
            # 30: [1.0, 0.75, 0.0, False, False],  # Right
            # 31: [1.0, 0.50, 0.0, False, False],  # Right
            # 32: [1.0, 0.25, 0.0, False, False],  # Right
            # 33: [1.0, -0.75, 0.0, False, False],  # Left
            # 34: [1.0, -0.50, 0.0, False, False],  # Left
            # 35: [1.0, -0.25, 0.0, False, False],  # Left
        }



    def compute_action(self, action):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero"""
        action_control = self.get_actions()[int(action)]

        action = carla.VehicleControl()
        action.throttle = action_control[0]
        action.steer = action_control[1]
        action.brake = action_control[2]
        action.reverse = action_control[3]
        action.hand_brake = action_control[4]

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
        print(f"----------------------------------->{action_msg}")

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

        self.radii = get_radii(core.route[core.last_waypoint_index:],5)

        self.entry_idx = core.entry_spawn_point_index
        self.exit_idx = core.exit_spawn_point_index

        number_of_waypoints_ahead_to_calculate_with = 0
        ahead_waypoints = 10
        ahead_waypoints_2 = 20

        # Getting truck location
        truck_transform = core.hero.get_transform()

        if core.config["truckTrailerCombo"]:
            # Getting trailer location
            trailer_transform = core.hero_trailer.get_transform()

        # print(f"BEFORE CHECKING IF PASSED LAST WAYPOINT {core.last_waypoint_index}")
        # Checking if we have passed the last way point
        in_front_of_waypoint = core.is_in_front_of_waypoint(truck_transform.location.x, truck_transform.location.y)
        if in_front_of_waypoint == 0 or in_front_of_waypoint == 1:
            core.last_waypoint_index += 1
            self.last_hyp_distance_to_next_waypoint = 0
            print('Passed Waypoint <------------')
        else:
            pass

        number_of_waypoints_to_plot_on_lidar = 20
        location_from_waypoint_to_vehicle_relative = np.zeros([2,number_of_waypoints_to_plot_on_lidar])

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

        x_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.x - truck_transform.location.x)
        y_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.y - truck_transform.location.y)
        hyp_distance_to_next_waypoint = math.sqrt((x_dist_to_next_waypoint) ** 2 + (y_dist_to_next_waypoint) ** 2)

        bearing_to_waypoint = angle_between(waypoint_forward_vector=core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].get_forward_vector(),vehicle_forward_vector=truck_transform.get_forward_vector())

        try:
            bearing_to_ahead_waypoints_ahead = angle_between(waypoint_forward_vector=core.route[core.last_waypoint_index + ahead_waypoints].get_forward_vector(),vehicle_forward_vector=truck_transform.get_forward_vector())

        except Exception as e:
            print(f"ERROR HERE1 {e}")
            bearing_to_ahead_waypoints_ahead = 0

        try:
            bearing_to_ahead_waypoints_ahead_2 = angle_between(waypoint_forward_vector=core.route[core.last_waypoint_index + ahead_waypoints_2].get_forward_vector(),vehicle_forward_vector=truck_transform.get_forward_vector())
        except Exception as e:
            print(f"ERROR HERE2 {e}")
            bearing_to_ahead_waypoints_ahead_2 = 0


        angle_between_truck_and_trailer = angle_between(waypoint_forward_vector=truck_transform.get_forward_vector(),vehicle_forward_vector=trailer_transform.get_forward_vector())

        # trailer_bearing_to_waypoint = angle_between(waypoint_forward_vector=core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].get_forward_vector(),vehicle_forward_vector=trailer_transform.get_forward_vector())


        self.vehicle_path.append((truck_transform.location.x,truck_transform.location.y))
        self.temp_route = core.route_points

        forward_velocity = np.clip(self.get_speed(core.hero), 0, None)
        # forward_velocity_x = np.clip(self.get_forward_velocity_x(core.hero), 0, None)
        # forward_velocity_z = np.clip(self.get_forward_velocity_z(core.hero), 0, None)
        # acceleration = np.clip(self.get_acceleration(core.hero), 0, None)

        # Angle to center of lane

        angle_to_center_of_lane_degrees = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index-1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location)

        angle_to_center_of_lane_degrees_ahead_waypoints = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index-1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + ahead_waypoints].location)
        try:
            angle_to_center_of_lane_degrees_ahead_waypoints_2 = calculate_angle_with_center_of_lane(
                previous_position=core.route[core.last_waypoint_index-1].location,
                current_position=truck_transform.location,
                next_position=core.route[core.last_waypoint_index + ahead_waypoints_2].location)
        except Exception as e:
            print(f"ERROR HERE3 {e}")
            angle_to_center_of_lane_degrees_ahead_waypoints_2 = 0

        if self.visualiseRoute and self.counter > self.counterThreshold:
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



        depth_camera_data = None
        current_occupancy_map = None
        for sensor in sensor_data:
            if sensor == 'collision_truck':
                # TODO change to only take collision with road
                # TO CHECK BY CHECKING LIDAR OUTPUT WHEN IN COMPLETE TURN
                # MAYBE I WOULD ABSTAIN FROM REMOVING IT BECAUSE YOU ARE STILL pushing the truck to the limit

                # static.sidewalk

                self.last_no_of_collisions_truck = len(sensor_data[sensor][1])
                self.truck_collisions.append(str(sensor_data[sensor][1][0]))
                print(f'COLLISIONS TRUCK {sensor_data[sensor][1][0]}')

            elif sensor == "collision_trailer":
                self.last_no_of_collisions_trailer = len(sensor_data[sensor][1])
                self.trailer_collisions.append(str(sensor_data[sensor][1][0]))
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
            elif sensor == "lidar_trailer_trailer":
                trailer_90_lidar_point = 0
                trailer_neg_90_lidar_point = 0

                lidar_points = sensor_data['lidar_trailer_trailer'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_trailer"]["range"])

                horizontal_indices = np.where((lidar_points[0] > -1.5) & (lidar_points[0] < 1.5))
                horizontal_relevant_lidar_y_points = lidar_points[1][horizontal_indices]

                if len(horizontal_relevant_lidar_y_points) == 0:
                    trailer_90_lidar_point = 1
                    trailer_neg_90_lidar_point = 1
                else:
                    greater_0_indices = np.where(horizontal_relevant_lidar_y_points > 0)
                    smaller_0_indices = np.where(horizontal_relevant_lidar_y_points < 0)


                    if greater_0_indices[0].size != 0:
                        trailer_90_lidar_point = abs(min(horizontal_relevant_lidar_y_points[greater_0_indices]) / lidar_range)
                    else:
                        trailer_90_lidar_point = 1

                    if smaller_0_indices[0].size != 0:
                        trailer_neg_90_lidar_point = abs(min(horizontal_relevant_lidar_y_points[smaller_0_indices]) / lidar_range)
                    else:
                        trailer_neg_90_lidar_point = 1

                    if trailer_90_lidar_point < 0:
                        trailer_90_lidar_point = 1

                    if trailer_neg_90_lidar_point < 0:
                        trailer_neg_90_lidar_point = 1


                # if len(lidar_points[1]) == 0:
                #     trailer_90_lidar_point = 1
                #     trailer_neg_90_lidar_point = -1
                # else:
                #     left_indices = np.where(lidar_points[1] < 0)
                #     right_indices = np.where(lidar_points[1] > 0)
                #
                #     trailer_90_lidar_point = max(lidar_points[right_indices][1])/lidar_range
                #     trailer_neg_90_lidar_point = min(lidar_points[left_indices][1]) / lidar_range
                #
                #     if trailer_90_lidar_point < 0:
                #         trailer_90_lidar_point = 1
                #
                #     if trailer_neg_90_lidar_point > 0:
                #         trailer_neg_90_lidar_point = -1
            elif sensor == "lidar_truck_side_truck":
                truck_90_lidar_point = 0
                truck_neg_90_lidar_point = 0
                truck_45_lidar_point = 0
                truck_neg_45_lidar_point = 0
                truck_0_lidar_point = 0

                lidar_points = sensor_data['lidar_truck_side_truck'][1]
                lidar_range = float(self.config["hero"]["sensors"]["lidar_truck_side"]["range"])

                horizontal_indices = np.where((lidar_points[0] > -1.5) & (lidar_points[0] < 1.5 ))
                horizontal_relevant_lidar_y_points = lidar_points[1][horizontal_indices]

                vertical_indices = np.where((lidar_points[1] > -1.5) & (lidar_points[1] < 1.5))
                vertical_relevant_lidar_x_points = lidar_points[0][vertical_indices]

                angle_45_indices = np.where((lidar_points[0] > 0) & (np.absolute(np.absolute(lidar_points[0]) - np.absolute(lidar_points[1])) <= 1.5))
                angle_45_relevant_lidar_x_points = lidar_points[0][angle_45_indices]
                angle_45_relevant_lidar_y_points = lidar_points[1][angle_45_indices]

                # print(f"lidar opints {lidar_points}")
                # print(f"horizontal_indices {horizontal_indices}")
                # print(f"horizontal_relevant_lidar_y_points {horizontal_relevant_lidar_y_points}")
                # print(f"vertical_indices {vertical_indices}")
                # print(f"vertical_relevant_lidar_x_points {vertical_relevant_lidar_x_points}")
                # print(f"angle_45_indices {angle_45_indices}")
                # print(f"angle_45_relevant_lidar_x_points{angle_45_relevant_lidar_x_points}")
                # print(f"angle_45_relevant_lidar_y_points{angle_45_relevant_lidar_y_points}")

                if len(angle_45_relevant_lidar_x_points) == 0:
                    truck_45_lidar_point = 1
                    truck_neg_45_lidar_point = 1
                else:

                    greater_0_indices = np.where(angle_45_relevant_lidar_x_points > 0)
                    smaller_0_indices = np.where(angle_45_relevant_lidar_x_points < 0)


                    if greater_0_indices[0].size != 0:
                        min_point = min(angle_45_relevant_lidar_x_points[greater_0_indices]**2 + angle_45_relevant_lidar_y_points[greater_0_indices]**2)
                        truck_45_lidar_point = math.sqrt(min_point) / lidar_range
                    else:
                        truck_45_lidar_point = 1

                    if smaller_0_indices[0].size != 0:
                        min_point = min(
                            angle_45_relevant_lidar_x_points[smaller_0_indices] ** 2 + angle_45_relevant_lidar_y_points[
                                smaller_0_indices] ** 2)
                        truck_neg_45_lidar_point = math.sqrt(min_point) / lidar_range
                    else:
                        truck_neg_45_lidar_point = 1

                    if truck_45_lidar_point < 0:
                        truck_45_lidar_point = 1

                    if truck_neg_45_lidar_point < 0:
                        truck_neg_45_lidar_point = 1


                if len(horizontal_relevant_lidar_y_points) == 0:
                    truck_90_lidar_point = 1
                    truck_neg_90_lidar_point = 1
                else:
                    greater_0_indices = np.where(horizontal_relevant_lidar_y_points > 0)
                    smaller_0_indices = np.where(horizontal_relevant_lidar_y_points < 0)


                    if greater_0_indices[0].size != 0:
                        truck_90_lidar_point = abs(min(horizontal_relevant_lidar_y_points[greater_0_indices]) / lidar_range)
                    else:
                        truck_90_lidar_point = 1

                    if smaller_0_indices[0].size != 0:
                        truck_neg_90_lidar_point = abs(min(horizontal_relevant_lidar_y_points[smaller_0_indices]) / lidar_range)
                    else:
                        truck_neg_90_lidar_point = 1

                    if truck_90_lidar_point < 0:
                        truck_90_lidar_point = 1

                    if truck_neg_90_lidar_point < 0:
                        truck_neg_90_lidar_point = 1

                if len(vertical_relevant_lidar_x_points) == 0:
                    truck_0_lidar_point = 1
                else:
                    truck_0_lidar_point = abs(min(vertical_relevant_lidar_x_points) / lidar_range)

                    if truck_0_lidar_point < 0:
                        truck_0_lidar_point = 1

                print(f"truck_0_lidar_point{truck_0_lidar_point}")

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

        observations = [
            np.float32(forward_velocity),
            # np.float32(forward_velocity_x),
            # np.float32(forward_velocity_z),
            np.float32(hyp_distance_to_next_waypoint),
            np.float32(angle_to_center_of_lane_degrees),
            np.float32(angle_to_center_of_lane_degrees_ahead_waypoints),
            np.float32(angle_to_center_of_lane_degrees_ahead_waypoints_2),
            np.float32(bearing_to_waypoint),
            np.float32(bearing_to_ahead_waypoints_ahead),
            np.float32(bearing_to_ahead_waypoints_ahead_2),
            np.float32(angle_between_truck_and_trailer),
            np.float32(trailer_90_lidar_point),
            np.float32(trailer_neg_90_lidar_point),
            np.float32(truck_0_lidar_point),
            np.float32(truck_neg_90_lidar_point),
            np.float32(truck_90_lidar_point),
            np.float32(truck_45_lidar_point),
            np.float32(truck_neg_45_lidar_point),
            # np.float32(trailer_bearing_to_waypoint),
            # np.float32(acceleration)
                           ]

        observations.extend([self.last_action[0],self.last_action[1],self.last_action[2]])

        observations.extend(self.radii)

        print(f"Radii {self.radii}")


        print(f"trailer_90_lidar_point{trailer_90_lidar_point}")
        print(f"trailer_neg_90_lidar_point{trailer_neg_90_lidar_point}")
        print(f"truck_0_lidar_point{truck_0_lidar_point}")
        print(f"truck_neg_90_lidar_point{truck_neg_90_lidar_point}")
        print(f"truck_90_lidar_point{truck_90_lidar_point}")
        print(f"truck_45_lidar_point{truck_45_lidar_point}")
        print(f"truck_neg_45_lidar_point{truck_neg_45_lidar_point}")
        # self.forward_velocity.append(np.float32(forward_velocity))
        # # self.forward_velocity_x.append(np.float32(forward_velocity_x))
        # # self.forward_velocity_z.append(np.float32(forward_velocity_z))
        # self.hyp_distance_to_next_waypoint.append(np.float32(hyp_distance_to_next_waypoint))
        # self.angle_to_center_of_lane_degrees.append(np.float32(angle_to_center_of_lane_degrees))
        # self.angle_to_center_of_lane_degrees_ahead_waypoints.append(np.float32(angle_to_center_of_lane_degrees_ahead_waypoints))
        # self.angle_to_center_of_lane_degrees_ahead_waypoints_2.append(np.float32(angle_to_center_of_lane_degrees_ahead_waypoints_2))
        # self.bearing_to_waypoint.append(np.float32(bearing_to_waypoint))
        # self.bearing_to_ahead_waypoints_ahead.append(np.float32(bearing_to_ahead_waypoints_ahead))
        # self.bearing_to_ahead_waypoints_ahead_2.append(np.float32(bearing_to_ahead_waypoints_ahead_2))
        # self.angle_between_truck_and_trailer.append(np.float32(angle_between_truck_and_trailer))
        # self.trailer_bearing_to_waypoint.append(np.float32(trailer_bearing_to_waypoint))
        # self.acceleration.append(np.float32(acceleration))
        #
        # print(f"angle_to_center_of_lane_degrees:{np.float32(angle_to_center_of_lane_degrees)}")
        # print(f"angle_to_center_of_lane_degrees_ahead_waypoints:{np.float32(angle_to_center_of_lane_degrees_ahead_waypoints)}")
        # print(f"angle_to_center_of_lane_degrees_ahead_waypoints_2:{np.float32(angle_to_center_of_lane_degrees_ahead_waypoints_2)}")
        print(f"bearing_to_waypoint:{np.float32(bearing_to_waypoint)}")
        # print(f"bearing_to_ahead_waypoints_ahead:{np.float32(bearing_to_ahead_waypoints_ahead)}")
        # print(f"bearing_to_ahead_waypoints_ahead_2:{np.float32(bearing_to_ahead_waypoints_ahead_2)}")
        # print(f"hyp_distance_to_next_waypoint:{np.float32(hyp_distance_to_next_waypoint)}")
        # print(f"forward_velocity:{np.float32(forward_velocity)}")
        print(f"angle_between_truck_and_trailer:{np.float32(angle_between_truck_and_trailer)}")
        # print(f"trailer_bearing_to_waypoint:{np.float32(trailer_bearing_to_waypoint)}")
        # print(f"forward_velocity_x:{np.float32(forward_velocity_x)}")
        # print(f"forward_velocity_z:{np.float32(forward_velocity_z)}")
        # print(f"acceleration:{np.float32(acceleration)}")

        self.counter += 1
        return observations,{}

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

        output = self.done_time_idle or self.done_falling or self.done_time_episode or self.done_collision_truck or self.done_collision_trailer or self.done_arrived
        self.custom_done_arrived = self.done_arrived

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

        if done_reason != "":
            data = f"ENTRY: {core.entry_spawn_point_index} EXIT: {core.exit_spawn_point_index} - {done_reason} \n"
            self.save_to_file(f"{self.directory}/done",data)

        return bool(output), self.done_collision_truck, self.done_collision_trailer, (self.done_time_idle or self.done_time_episode), self.done_arrived

    def compute_reward(self, observation, core):
        """Computes the reward"""
        # est
        reward = 0

        forward_velocity = observation[0]
        hyp_distance_to_next_waypoint = observation[1]
        # print(f"in rewards forward_velocity {forward_velocity}")
        # print(f"in rewards hyp_distance_to_next_waypoint {hyp_distance_to_next_waypoint}")

        #bearing_to_waypoint = observation[5]
        # bearing_to_ahead_waypoints_ahead = observation["values"][5]
        # angle_between_truck_and_trailer = observation["values"][6]


        self.last_forward_velocity = forward_velocity

        print(f"Hyp distance in rewards {hyp_distance_to_next_waypoint}")
        if self.last_hyp_distance_to_next_waypoint != 0:
            hyp_reward = self.last_hyp_distance_to_next_waypoint - hyp_distance_to_next_waypoint
            reward = reward + hyp_reward*100
            print(f"REWARD hyp_distance_to_next_waypoint = {hyp_reward*100}")

        self.last_hyp_distance_to_next_waypoint = hyp_distance_to_next_waypoint



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

        if forward_velocity < 0.75:
            # Negative reward for no velocity
            print('REWARD -100 for velocity')
            reward = reward + -100


        if self.done_falling:
            reward = reward + -1000
            print('====> REWARD Done falling')
        if self.done_collision_truck or self.done_collision_trailer:
            print("====> REWARD Done collision")
            reward = reward + -1000
        if self.done_time_idle:
            print("====> REWARD Done idle")
            reward = reward + -1000
        if self.done_time_episode:
            print("====> REWARD Done max time")
            reward = reward + -1000
        if self.done_arrived:
            print("====> REWARD Done arrived")
            reward = reward + 10000

        self.reward_metric = reward
        print(f"Reward: {reward}")
        return reward