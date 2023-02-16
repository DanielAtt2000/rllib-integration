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
from rllib_integration.TestingWayPointUpdater import plot_points
from rllib_integration.base_experiment import BaseExperiment
from rllib_integration.helper import post_process_image
from PIL import Image



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
        self.visualiseRoute = False
        self.visualiseImage = False
        self.counterThreshold = 10
        self.last_hyp_distance_to_next_waypoint = 0

        self.x_dist_to_waypoint = []
        self.y_dist_to_waypoint = []
        self.angle_with_center = []
        self.bearing_to_waypoint = []
        self.forward_velocity = []
        self.forward_velocity_x = []
        self.forward_velocity_z = []
        self.hyp_distance_to_next_waypoint = []
        self.acceleration = []
        self.no_of_collisions = []

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
        self.done_collision = False
        self.done_arrived = False
        self.custom_done_arrived = False

        # hero variables
        self.last_location = None
        self.last_velocity = 0
        self.last_dist_to_finish = 0


        self.last_angle_with_center = 0
        self.last_forward_velocity = 0

        self.last_no_of_collisions = 0

        self.save_to_file(f"{self.directory}/hyp_distance_to_next_waypoint", self.hyp_distance_to_next_waypoint)
        self.save_to_file(f"{self.directory}/angle_with_center", self.angle_with_center)
        self.save_to_file(f"{self.directory}/bearing", self.bearing_to_waypoint)
        self.save_to_file(f"{self.directory}/forward_velocity", self.forward_velocity)
        self.save_to_file(f"{self.directory}/forward_velocity_x", self.forward_velocity_x)
        self.save_to_file(f"{self.directory}/forward_velocity_z", self.forward_velocity_z)
        self.save_to_file(f"{self.directory}/acceleration", self.acceleration)

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
        self.angle_with_center = []
        self.bearing_to_waypoint = []
        self.forward_velocity = []
        self.forward_velocity_x = []
        self.forward_velocity_z = []
        self.hyp_distance_to_next_waypoint = []
        self.acceleration = []





    # [33,28, 27, 17,  14, 11, 10, 5]

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        """
        Set observation space as location of vehicle im x,y starting at (0,0) and ending at (1,1)
        :return:
        """
        image_space = Box(
                low=np.array([0,0,0,0]),
                high=np.array([100,100,2*math.pi,2*math.pi]),
                dtype=np.float32,
            )
        return image_space

    def get_actions(self):
        return {
            0: [0.3, 0.00, 0.0, False, False],  # Coast
            1: [0.3, 0.75, 0.0, False, False],  # Right
            2: [0.3, 0.50, 0.0, False, False],  # Right
            3: [0.3, 0.25, 0.0, False, False],  # Right
            4: [0.3, -0.75, 0.0, False, False],  # Left
            5: [0.3, -0.50, 0.0, False, False],  # Left
            6: [0.3, -0.25, 0.0, False, False],  # Left
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

        # Getting truck location
        truck_transform = core.hero.get_transform()




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

        if bearing_to_waypoint > 0:
            strings = [ f"-------------------------------------------\n"
                        f"bearing_to_waypoint: {bearing_to_waypoint}\n",
                        f"Truck: {truck_transform.get_forward_vector()}\n",
                        f"Waypoint : {core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].get_forward_vector()}\n",
                        f"bearing_to_waypoint: {bearing_to_waypoint}\n",
                        f"-------------------------------------------\n"]

            print(strings)
            with open('bearing.txt', 'a') as file:
                file.writelines(strings)


        forward_velocity = np.clip(self.get_speed(core.hero), 0, None)
        forward_velocity_x = np.clip(self.get_forward_velocity_x(core.hero), 0, None)
        forward_velocity_z = np.clip(self.get_forward_velocity_z(core.hero), 0, None)
        acceleration = np.clip(self.get_acceleration(core.hero), 0, None)

        # Angle to center of lane

        angle_to_center_of_lane_degrees = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index-1].location,
            current_position=truck_transform.location,
            next_position=core.route[core.last_waypoint_index + number_of_waypoints_ahead_to_calculate_with].location)


        if self.visualiseRoute and self.counter > self.counterThreshold:
            def plot_route():
                x_route = []
                y_route = []
                for point in core.route:
                    # print(f"X: {point.location.x} Y:{point.location.y}")
                    x_route.append(point.location.x)
                    y_route.append(point.location.y)

                x_min = min(x_route)
                x_max = max(x_route)

                y_min = min(y_route)
                y_max = max(y_route)
                buffer = 10

                # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
                plt.plot([x_route.pop(0)],y_route.pop(0),'bo')
                plt.plot(x_route, y_route,'y^')
                plt.plot([core.route[core.last_waypoint_index-1].location.x], [core.route[core.last_waypoint_index-1].location.y], 'ro',label='Previous Waypoint')
                plt.plot([truck_transform.location.x], [truck_transform.location.y], 'gs',label='Current Vehicle Location')
                plt.plot([core.route[core.last_waypoint_index+number_of_waypoints_ahead_to_calculate_with].location.x], [core.route[core.last_waypoint_index+number_of_waypoints_ahead_to_calculate_with].location.y], 'bo', label=f"{number_of_waypoints_ahead_to_calculate_with} waypoints ahead")
                plt.axis([x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer])
                # plt.axis([0, 1, 0, 1])
                plt.title(f'{angle_to_center_of_lane_degrees*180}')
                plt.gca().invert_yaxis()
                plt.legend(loc='upper center')
                plt.show()


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

            plot_route()

        self.counter +=1

        for sensor in sensor_data:
            if sensor == 'collision_truck':
                # TODO change to only take collision with road

                self.last_no_of_collisions = len(sensor_data[sensor][1])
                print(f'COLLISIONS {sensor_data[sensor]}')

        observations = [
            np.float32(forward_velocity),
            # np.float32(forward_velocity_x),
            # np.float32(forward_velocity_z),
            np.float32(hyp_distance_to_next_waypoint),
            np.float32(angle_to_center_of_lane_degrees),
            np.float32(bearing_to_waypoint),
            # np.float32(acceleration)
                           ]

        self.forward_velocity.append(np.float32(forward_velocity))
        # self.forward_velocity_x.append(np.float32(forward_velocity_x))
        # self.forward_velocity_z.append(np.float32(forward_velocity_z))
        self.hyp_distance_to_next_waypoint.append(np.float32(hyp_distance_to_next_waypoint))
        self.angle_with_center.append(np.float32(angle_to_center_of_lane_degrees))
        self.bearing_to_waypoint.append(np.float32(bearing_to_waypoint))
        # self.acceleration.append(np.float32(acceleration))

        print(f"angle_to_center_of_lane_degrees:{np.float32(angle_to_center_of_lane_degrees)}")
        print(f"bearing_to_waypoint:{np.float32(bearing_to_waypoint)}")
        print(f"hyp_distance_to_next_waypoint:{np.float32(hyp_distance_to_next_waypoint)}")
        print(f"forward_velocity:{np.float32(forward_velocity)}")
        # print(f"forward_velocity_x:{np.float32(forward_velocity_x)}")
        # print(f"forward_velocity_z:{np.float32(forward_velocity_z)}")
        # print(f"acceleration:{np.float32(acceleration)}")

        return observations, {}

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
        self.done_collision = self.last_no_of_collisions > 0
        self.done_arrived = self.completed_route(core)

        output = self.done_time_idle or self.done_falling or self.done_time_episode or self.done_collision or self.done_arrived
        self.custom_done_arrived = self.done_arrived

        done_reason = ""
        if self.done_time_idle:
            done_reason += "done_time_idle"
        if self.done_falling:
            done_reason += "done_falling"
        if self.done_time_episode:
            done_reason += "done_time_episode"
        if self.done_collision:
            done_reason += "done_collision"
        if self.done_arrived:
            done_reason += "done_arrived"

        if done_reason != "":
            data = f"ENTRY: {core.entry_spawn_point_index} EXIT: {core.exit_spawn_point_index} - {done_reason} \n"
            self.save_to_file(f"{self.directory}/done",data)

        return bool(output)

    def compute_reward(self, observation, core):
        """Computes the reward"""

        reward = 0

        hyp_distance_to_next_waypoint = observation[1]

        print(hyp_distance_to_next_waypoint)
        if self.last_hyp_distance_to_next_waypoint != 0:
            hyp_reward = self.last_hyp_distance_to_next_waypoint - hyp_distance_to_next_waypoint
            reward =+ hyp_reward*100
            print(f"REWARD hyp_distance_to_next_waypoint = {hyp_reward}")

        self.last_hyp_distance_to_next_waypoint = hyp_distance_to_next_waypoint


        if self.done_falling:
            reward += -1000
            print('====> REWARD Done falling')
        if self.done_collision:
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