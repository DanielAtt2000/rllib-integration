#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import matplotlib.pyplot as plt


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
from PIL import Image

from rllib_integration.lidar_to_grid_map import generate_ray_casting_grid_map


class DQNExperimentBasic(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_angle_with_center = 0
        self.last_forward_velocity = 0
        self.custom_done_arrived = False
        self.last_action = None
        self.lidar_points_count = []

        self.min_lidar_values = 1000000
        self.max_lidar_values = -100000
        self.lidar_max_points = self.config["hero"]["lidar_max_points"]
        self.counter = 0
        self.visualiseRoute = True
        self.visualiseImage = False
        self.visualiseOccupancyGirdMap = False
        self.counterThreshold = 10
        self.last_hyp_distance_to_next_waypoint = 0

        self.x_dist_to_waypoint = []
        self.y_dist_to_waypoint = []
        self.angle_to_center_of_lane_degrees = []
        self.angle_to_center_of_lane_degrees_ahead_waypoints = []
        self.bearing_to_waypoint = []
        self.bearing_to_ahead_waypoints_ahead = []
        self.angle_between_truck_and_trailer = []
        self.forward_velocity = []
        # self.forward_velocity_x = []
        # self.forward_velocity_z = []
        self.vehicle_path = []
        self.temp_route = []
        self.hyp_distance_to_next_waypoint = []
        # self.acceleration = []
        self.truck_collisions = []
        self.trailer_collisions =[]

        self.last_no_of_collisions_truck = 0
        self.last_no_of_collisions_trailer = 0

        from git import Repo
        repo = Repo('.')
        remote = repo.remote('origin')
        remote.fetch()
        self.directory = f"data/data_{str(repo.head.commit)[:11]}"

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def save_to_file(self, file_name, data):
        # Saving LIDAR point count
        counts = open(file_name, 'a')
        counts.write(str(data))
        counts.write(str('\n'))
        counts.close()

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

        # hero variables
        self.last_location = None
        self.last_velocity = 0
        self.last_dist_to_finish = 0


        self.last_angle_with_center = 0
        self.last_forward_velocity = 0

        self.last_no_of_collisions_truck = 0
        self.last_no_of_collisions_trailer = 0

        self.save_to_file(f"{self.directory}/hyp_distance_to_next_waypoint", self.hyp_distance_to_next_waypoint)
        self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees", self.angle_to_center_of_lane_degrees)
        self.save_to_file(f"{self.directory}/angle_to_center_of_lane_degrees_ahead_waypoints", self.angle_to_center_of_lane_degrees_ahead_waypoints)
        self.save_to_file(f"{self.directory}/bearing", self.bearing_to_waypoint)
        self.save_to_file(f"{self.directory}/bearing_ahead_ahead", self.bearing_to_ahead_waypoints_ahead)
        self.save_to_file(f"{self.directory}/angle_between_truck_and_trailer", self.angle_between_truck_and_trailer)
        self.save_to_file(f"{self.directory}/forward_velocity", self.forward_velocity)
        # self.save_to_file(f"{self.directory}/forward_velocity_x", self.forward_velocity_x)
        # self.save_to_file(f"{self.directory}/forward_velocity_z", self.forward_velocity_z)
        # self.save_to_file(f"{self.directory}/acceleration", self.acceleration)
        self.save_to_file(f"{self.directory}/route", self.temp_route)
        self.save_to_file(f"{self.directory}/path", self.vehicle_path)
        self.save_to_file(f"{self.directory}/truck_collisions", self.truck_collisions)
        self.save_to_file(f"{self.directory}/trailer_collisions", self.trailer_collisions)

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
        self.bearing_to_waypoint = []
        self.bearing_to_ahead_waypoints_ahead = []
        self.angle_between_truck_and_trailer = []
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
        image_space = Dict(
            {"values": Box(
                low=np.array([0,0,-math.pi,-math.pi,-math.pi,-math.pi,-math.pi]),
                high=np.array([100,100,math.pi,math.pi,math.pi,math.pi,math.pi]),
                dtype=np.float32
            ),
            # "depth_camera": Box(
            #     low=0,
            #     high=255,
            #     shape=(84, 84, 3),
            #     dtype=np.float32
            # )
            # "occupancyMap": Box(
            #     low=0,
            #     high=1,
            #     shape=(240, 320,1),
            #     dtype=np.float64
            # )
            })
        return image_space

    def get_actions(self):
        return {
            0: [0.0, 0.00, 0.0, False, False],  # Coast
            1: [0.0, 0.00, 1.0, False, False],  # Apply Break
            2: [0.0, 0.75, 0.0, False, False],  # Right
            3: [0.0, 0.50, 0.0, False, False],  # Right
            4: [0.0, 0.25, 0.0, False, False],  # Right
            5: [0.0, -0.75, 0.0, False, False],  # Left
            6: [0.0, -0.50, 0.0, False, False],  # Left
            7: [0.0, -0.25, 0.0, False, False],  # Left
            8: [0.3, 0.00, 0.0, False, False],  # Straight
            9: [0.3, 0.75, 0.0, False, False],  # Right
            10: [0.3, 0.50, 0.0, False, False],  # Right
            11: [0.3, 0.25, 0.0, False, False],  # Right
            12: [0.3, -0.75, 0.0, False, False],  # Left
            13: [0.3, -0.50, 0.0, False, False],  # Left
            14: [0.3, -0.25, 0.0, False, False],  # Left
            15: [0.6, 0.00, 0.0, False, False],  # Straight
            16: [0.6, 0.75, 0.0, False, False],  # Right
            17: [0.6, 0.50, 0.0, False, False],  # Right
            18: [0.6, 0.25, 0.0, False, False],  # Right
            19: [0.6, -0.75, 0.0, False, False],  # Left
            20: [0.6, -0.50, 0.0, False, False],  # Left
            21: [0.6, -0.25, 0.0, False, False],  # Left
            22: [1.0, 0.00, 0.0, False, False],  # Straight
            23: [1.0, 0.75, 0.0, False, False],  # Right
            24: [1.0, 0.50, 0.0, False, False],  # Right
            25: [1.0, 0.25, 0.0, False, False],  # Right
            26: [1.0, -0.75, 0.0, False, False],  # Left
            27: [1.0, -0.50, 0.0, False, False],  # Left
            28: [1.0, -0.25, 0.0, False, False],  # Left
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
            action_msg += " Forward "

        if action_control[1] < 0:
            action_msg += " Left "

        if action_control[1] > 0:
            action_msg += " Right "

        if action_control[2] != 0:
            action_msg += " Break "

        if action_msg == "":
            action_msg += " Coast "


        # print(f'Throttle {action.throttle} Steer {action.steer} Brake {action.brake} Reverse {action.reverse} Handbrake {action.hand_brake}')
        print(f"----------------------------------->{action_msg}")

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

        number_of_waypoints_ahead_to_calculate_with = 0
        ahead_waypoints = 10

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

        x_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.x - truck_transform.location.x)
        y_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location.y - truck_transform.location.y)
        hyp_distance_to_next_waypoint = math.sqrt((x_dist_to_next_waypoint) ** 2 + (y_dist_to_next_waypoint) ** 2)

        bearing_to_waypoint = angle_between(waypoint_forward_vector=core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].get_forward_vector(),vehicle_forward_vector=truck_transform.get_forward_vector())

        bearing_to_ahead_waypoints_ahead = angle_between(waypoint_forward_vector=core.route[core.last_waypoint_index + ahead_waypoints].get_forward_vector(),vehicle_forward_vector=truck_transform.get_forward_vector())

        angle_between_truck_and_trailer = angle_between(waypoint_forward_vector=truck_transform.get_forward_vector(),vehicle_forward_vector=trailer_transform.get_forward_vector())

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


        self.counter +=1
        depth_camera_data = None
        occupancy_map = None
        for sensor in sensor_data:
            if sensor == 'collision_truck':
                # TODO change to only take collision with road

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
            elif sensor == "lidar_truck":
                lidar_points = sensor_data['lidar_truck'][1]

                xy_resolution = 0.2
                x_output = 320
                y_output = 240

                ox = lidar_points[0][:]
                oy = lidar_points[1][:]

                occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = \
                    generate_ray_casting_grid_map(ox=ox, oy=oy, x_output=x_output, y_output=y_output,
                                                  xy_resolution=xy_resolution, breshen=True)
                # Inverted the image as a test
                # occupancy_map = occupancy_map[::-1]
                # print(f"Final image size {occupancy_map.shape}")

                if self.visualiseOccupancyGirdMap and self.counter % 10 == 0:
                    plt.figure()
                    xy_res = np.array(occupancy_map).shape
                    plt.imshow(occupancy_map, cmap="PiYG_r")
                    # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
                    plt.clim(-0.4, 1.4)
                    plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
                    plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
                    plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
                    # plt.gca().invert_yaxis()
                    plt.show()

                assert occupancy_map is not None

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
            np.float32(bearing_to_waypoint),
            np.float32(bearing_to_ahead_waypoints_ahead),
            np.float32(angle_between_truck_and_trailer),
            # np.float32(acceleration)
                           ]

        self.forward_velocity.append(np.float32(forward_velocity))
        # self.forward_velocity_x.append(np.float32(forward_velocity_x))
        # self.forward_velocity_z.append(np.float32(forward_velocity_z))
        self.hyp_distance_to_next_waypoint.append(np.float32(hyp_distance_to_next_waypoint))
        self.angle_to_center_of_lane_degrees.append(np.float32(angle_to_center_of_lane_degrees))
        self.angle_to_center_of_lane_degrees_ahead_waypoints.append(np.float32(angle_to_center_of_lane_degrees_ahead_waypoints))
        self.bearing_to_waypoint.append(np.float32(bearing_to_waypoint))
        self.bearing_to_ahead_waypoints_ahead.append(np.float32(bearing_to_ahead_waypoints_ahead))
        self.angle_between_truck_and_trailer.append(np.float32(angle_between_truck_and_trailer))
        # self.acceleration.append(np.float32(acceleration))

        print(f"angle_to_center_of_lane_degrees:{np.float32(angle_to_center_of_lane_degrees)}")
        print(f"angle_to_center_of_lane_degrees_ahead_waypoints:{np.float32(angle_to_center_of_lane_degrees_ahead_waypoints)}")
        print(f"bearing_to_waypoint:{np.float32(bearing_to_waypoint)}")
        print(f"bearing_to_ahead_waypoints_ahead:{np.float32(bearing_to_ahead_waypoints_ahead)}")
        print(f"hyp_distance_to_next_waypoint:{np.float32(hyp_distance_to_next_waypoint)}")
        # print(f"forward_velocity:{np.float32(forward_velocity)}")
        print(f"angle_between_truck_and_trailer:{np.float32(angle_between_truck_and_trailer)}")
        # print(f"forward_velocity_x:{np.float32(forward_velocity_x)}")
        # print(f"forward_velocity_z:{np.float32(forward_velocity_z)}")
        # print(f"acceleration:{np.float32(acceleration)}")
        return {"values":observations,
                # "occupancyMap":occupancy_map
                # "depth_camera":depth_camera_data
                }, \
            {
                # "occupancy_map":occupancy_map,
            # "depth_camera":depth_camera_data
             }

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
        if len(core.route) - 5 <= core.last_waypoint_index:
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

        reward = 0

        hyp_distance_to_next_waypoint = observation["values"][1]

        print(f"Hyp distance in rewards {hyp_distance_to_next_waypoint}")
        if self.last_hyp_distance_to_next_waypoint != 0:
            hyp_reward = self.last_hyp_distance_to_next_waypoint - hyp_distance_to_next_waypoint
            reward =+ hyp_reward*100
            print(f"REWARD hyp_distance_to_next_waypoint = {hyp_reward}")

        self.last_hyp_distance_to_next_waypoint = hyp_distance_to_next_waypoint


        if self.done_falling:
            reward += -1000
            print('====> REWARD Done falling')
        if self.done_collision_truck or self.done_collision_trailer:
            print("====> REWARD Done collision")
            reward += -1000
        if self.done_time_idle:
            print("====> REWARD Done idle")
            reward += -1000
        if self.done_time_episode:
            print("====> REWARD Done max time")
            reward += -1000
        if self.done_arrived:
            print("====> REWARD Done arrived")
            reward += 10000

        print(f"Reward: {reward}")
        return reward