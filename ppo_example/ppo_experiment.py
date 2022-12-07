#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import matplotlib.pyplot as plt
import math
import numpy as np
from gym.spaces import Box, Discrete, Dict
import warnings
import carla
import os

from rllib_integration.GetAngle import calculate_angle_with_center_of_lane
from rllib_integration.base_experiment import BaseExperiment
from rllib_integration.helper import post_process_image


class PPOExperiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_heading_deviation = 0
        self.last_action = None
        self.lidar_points_count = []
        self.min_lidar_values = 1000000
        self.max_lidar_values = -100000
        self.lidar_max_points = self.config["hero"]["lidar_max_points"]
        self.counter = 0
        self.visualiseRoute = False


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

        # hero variables
        self.last_location = None
        self.last_velocity = 0
        self.last_dist_to_finish = 0


        self.last_heading_deviation = 0

        self.last_no_of_collisions = 0


        # Saving LIDAR point count
        file_lidar_counts = open(os.path.join('lidar_output','lidar_point_counts.txt'), 'a')
        file_lidar_counts.write(str(self.lidar_points_count))
        file_lidar_counts.write(str('\n'))
        file_lidar_counts.close()

        file_lidar_counts = open(os.path.join('lidar_output', 'min_lidar_values.txt'), 'a')
        file_lidar_counts.write(str("Min Lidar Value:" + str(self.min_lidar_values)))
        file_lidar_counts.write(str('\n'))
        file_lidar_counts.close()

        file_lidar_counts = open(os.path.join('lidar_output', 'max_lidar_values.txt'), 'a')
        file_lidar_counts.write(str("Max Lidar Value:" + str(self.max_lidar_values)))
        file_lidar_counts.write(str('\n'))
        file_lidar_counts.close()

        self.min_lidar_values = 1000000
        self.max_lidar_values = -100000
        self.lidar_points_count = []
        self.counter = 0




    # [33,28, 27, 17,  14, 11, 10, 5]

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        return Discrete(len(self.get_actions()))

    # def get_observation_space(self):
    #     num_of_channels = 3
    #     image_space = Box(
    #         low=0.0,
    #         high=255.0,
    #         shape=(
    #             self.config["hero"]["sensors"]["birdview"]["size"],
    #             self.config["hero"]["sensors"]["birdview"]["size"],
    #             num_of_channels * self.frame_stack,
    #         ),
    #         dtype=np.uint8,
    #     )
    #     return image_space

    def get_observation_space(self):
        """
        Set observation space as location of vehicle im x,y starting at (0,0) and ending at (1,1)
        :return:
        """
        spaces = {
            # 'values': Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,float("inf"),1,1,1]), dtype=np.float32),
            'values': Box(low=np.array([0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1]), dtype=np.float32),

            'lidar': Box(low=-1000, high=1000,shape=(self.lidar_max_points,5), dtype=np.float32),

        }
        # return Box(low=np.array([float("-inf"), float("-inf"),-1.0,0,float("-inf"),0,0]), high=np.array([float("inf"),float("inf"),1.0,1.0,float("inf"),20,20]), dtype=np.float32)
        obs_space = Dict(spaces)
        # print('SAMPLE')
        # print(obs_space.sample())
        return obs_space



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

        # print(f'Throttle {action.throttle} Steer {action.steer} Brake {action.brake} Reverse {action.reverse} Handbrake {action.hand_brake}')


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


        # Getting truck location
        truck_transform = core.hero.get_transform()

        truck_normalised_transform = carla.Transform(
            carla.Location(core.normalise_map_location(truck_transform.location.x, 'x'),
                           core.normalise_map_location(truck_transform.location.y, 'y'),
                           0),
            carla.Rotation(0, 0, 0))

        if core.config["truckTrailerCombo"]:
            # Getting trailer location
            trailer_transform = core.hero_trailer.get_transform()
            trailer_normalised_transform = carla.Transform(
                carla.Location(core.normalise_map_location(trailer_transform.location.x, 'x'),
                               core.normalise_map_location(trailer_transform.location.y, 'y'),
                               0),
                carla.Rotation(0, 0, 0))

        # print(f"BEFORE CHECKING IF PASSED LAST WAYPOINT {core.last_waypoint_index}")
        # Checking if we have passed the last way point
        in_front_of_waypoint = core.is_in_front_of_waypoint(truck_normalised_transform.location.x, truck_normalised_transform.location.y)
        if in_front_of_waypoint == 0 or in_front_of_waypoint == 1:
            core.last_waypoint_index += 1
        else:
            pass
        #print(f"OBS -> Len(route) {len(core.route)}")
        #print(f'OBS -> core.last_waypoint_index {core.last_waypoint_index}')
        # print(f"AFTER CHECKING IF PASSED LAST WAYPOINT {core.last_waypoint_index}")


        # Distance to next waypoint
        x_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index+1].location.x - truck_normalised_transform.location.x, )
        y_dist_to_next_waypoint = abs(core.route[core.last_waypoint_index+1].location.y - truck_normalised_transform.location.y )
        # print(f"DISTANCE TO NEXT WAY POINT X {x_dist_to_next_waypoint}")
        # print(f"DISTANCE TO NEXT WAY POINT Y {y_dist_to_next_waypoint}")

        # Forward Velocity
        # Normalising it between 0 and 50
        forward_velocity = np.clip(self.get_speed(core.hero), 0, None)
        forward_velocity = np.clip(forward_velocity, 0, 50.0) / 50

        # Acceleration
        # TODO Normalise acceleration
        acceleration = self.get_acceleration(core.hero)


        # Angle to center of lane
        # Normalising it
        angle_to_center_of_lane_degrees = calculate_angle_with_center_of_lane(
            previous_position=core.route[core.last_waypoint_index-1].location,
            current_position=truck_normalised_transform.location,
            next_position=core.route[core.last_waypoint_index+5].location)
        angle_to_center_of_lane_degrees = np.clip(angle_to_center_of_lane_degrees,0,180) / 180

        if self.visualiseRoute and self.counter % 30 == 0:
            x_route = []
            y_route = []
            for point in core.route:
                # print(f"X: {point.location.x} Y:{point.location.y}")
                x_route.append(point.location.x)
                y_route.append(point.location.y)
            # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
            plt.plot([x_route.pop(0)],y_route.pop(0),'bo')
            plt.plot(x_route, y_route,'y^')
            plt.plot([core.route[core.last_waypoint_index-1].location.x], [core.route[core.last_waypoint_index-1].location.y], 'ro')
            plt.plot([truck_normalised_transform.location.x], [truck_normalised_transform.location.y], 'gs')
            plt.plot([core.route[core.last_waypoint_index+5].location.x], [core.route[core.last_waypoint_index+5].location.y], 'bo')
            plt.axis([0.3, 0.7, 0.3, 0.7])
            # plt.axis([0, 1, 0, 1])
            plt.title(f'{angle_to_center_of_lane_degrees*180}')
            plt.gca().invert_yaxis()
            plt.show()
        self.counter +=1

        # heading = np.sin(transform.rotation.yaw * np.pi / 180)
        #

        lidar_data_padded = None
        for sensor in sensor_data:
            if sensor == 'collision_truck':
                # TODO change to only take collision with road

                self.last_no_of_collisions = len(sensor_data[sensor][1])
                print(f'COLLISIONS {sensor_data[sensor]}')
            elif sensor == 'lidar_truck':
                lidar_data = sensor_data['lidar_truck'][1]
                # print(f'LIDAR DATA {lidar_data[0,:]}')
                # print(f'BEFORE {lidar_data}')
                # print(f'BEFORE SHAPE{lidar_data.shape}')
                self.lidar_points_count.append(len(lidar_data))
                # NO Normalisation required on LIDAR since its coordinates are relative to the actor

                # maybe need to normalise Z axis as well
                # map_location_normaliser = np.vectorize(core.normalise_map_location)
                # x_lidar_normalised = map_location_normaliser(sensor_data['lidar_truck'][1][:, 0], axis='x',assertBetween=False)
                # y_lidar_normalised = map_location_normaliser(sensor_data['lidar_truck'][1][:, 1], axis='y',assertBetween=False)
                #
                # lidar_data = np.array([x_lidar_normalised, y_lidar_normalised, lidar_data[:,2], lidar_data[:,3], lidar_data[:,4],  lidar_data[:,5]]).T
                # print(f'AFTER NORMALISATION{lidar_data}')
                # print(f'AFTER NORMALISATION SHAPE{lidar_data.shape}')

                # Deleting any lidar points with x and y positions outside the map
                # lidar_data = np.delete(lidar_data, np.where((lidar_data[:, 0] > 1))[0], axis=0)
                # lidar_data = np.delete(lidar_data, np.where((lidar_data[:, 0] < 0))[0], axis=0)
                # lidar_data = np.delete(lidar_data, np.where((lidar_data[:, 1] > 1))[0], axis=0)
                # lidar_data = np.delete(lidar_data, np.where((lidar_data[:, 1] < 0))[0], axis=0)

                # print(f'AFTER DELETE {lidar_data}')
                # print(f'AFTER DELETE SHAPE{lidar_data.shape}')

                # Padding LIDAR to have constant number of points

                if self.lidar_max_points < len(lidar_data):
                    warnings.warn(f'self.lidar_max_points < len(lidar_data)\n'
                                f'{self.lidar_max_points} < {len(lidar_data)}\n'
                                f'LOSING LIDAR DATA')

                if (self.lidar_max_points - len(lidar_data)) > 2000 :
                    warnings.warn(f"Difference between lidar_max_points and points is {self.lidar_max_points - len(lidar_data)}\n"
                                  f"WASTING MEMORY")

                number_of_rows_to_pad = self.lidar_max_points - len(lidar_data)
                lidar_data_padded = np.pad(lidar_data, [(0, number_of_rows_to_pad), (0, 0)], mode='constant', constant_values=-1)


                for row in lidar_data_padded:
                    for col in row:
                        if col > self.max_lidar_values:
                            self.max_lidar_values = col
                        elif col < self.min_lidar_values:
                            self.min_lidar_values = col

                    # if len(np.where(row[:] > 999)) != 0:
                    #     print(np.where(row[:] > 999))
                    #     print(row[:])
                    # elif len(np.where(row[:] < -999)) != 0:
                    #     print(np.where(row[:] < -999))
                    #     print(row[:])
                # print(f'AFTER PADDING{lidar_data_padded}')
                # print(f'AFTER PADDING SHAPE{lidar_data_padded.shape}')



                # lidar_data = sensor_data[sensor]
                # print(f"LIDAR ONE {sensor_data['lidar_truck'][1][0]}")
                # print(f'LIDAR Data Shape {sensor_data[sensor][1].shape}')
                # np.apply_along_axis(core.normalise_map_location(value=,axis='x'))

        # print("OBSERVATIONS START")
        # print(f"truck_normalised_transform.location.x {truck_normalised_transform.location.x}")
        # print(f"truck_normalised_transform.location.y {truck_normalised_transform.location.y}")
        # print(f"forward_velocity {forward_velocity}")
        # print(f"acceleration {acceleration}")
        # print(f"x_dist_to_next_waypoint {x_dist_to_next_waypoint}")
        # print(f"y_dist_to_next_waypoint {y_dist_to_next_waypoint}")
        # print(f"angle_to_center_of_lane_degrees {angle_to_center_of_lane_degrees}")
        # print("OBSERVATIONS STOP")


        # values = np.r_[
        #         np.float32(truck_normalised_transform.location.x),
        #         # np.float32(truck_normalised_transform.location.y),
        #         # np.float32(forward_velocity),
        #         # np.float32(acceleration),
        #         # np.float32(x_dist_to_next_waypoint),
        #         # np.float32(y_dist_to_next_waypoint),
        #         # np.float32(angle_to_center_of_lane_degrees),
        #         #LIDAR
        #         # Last action here?
        #     ]

        # print("DTYPE")
        # print(sensor_data['lidar_truck'][1][0:5,:])
        # print(sensor_data['lidar_truck'][1][0:5,:].dtype)
        # print("DTYPE")


        if lidar_data_padded is None:
            raise Exception('LIDAR DATA NOT FILLED')

        name_observations = ["truck_normalised_transform.location.x",
                             "truck_normalised_transform.location.y",
                             "forward_velocity",
                             # "acceleration",
                             "x_dist_to_next_waypoint",
                             "y_dist_to_next_waypoint",
                             "angle_to_center_of_lane_degrees"]
        observations = [
            np.float32(truck_normalised_transform.location.x),
            np.float32(truck_normalised_transform.location.y),
            np.float32(forward_velocity),
            # np.float32(acceleration),
            np.float32(x_dist_to_next_waypoint),
            np.float32(y_dist_to_next_waypoint),
            np.float32(angle_to_center_of_lane_degrees),
                           ]

        observation_file = open( os.path.join("results","run_" + str(core.current_time),"observations_" + str(core.current_time) + ".txt"), 'a+')
        for idx, obs in enumerate(observations):
            observation_file.write(f"{name_observations[idx]}:{round(obs,5)}\n")
        observation_file.close()

        return {'values': np.array(observations),
                'lidar':lidar_data_padded
        }, {}
        # return  np.r_[
        #                 np.float32(truck_normalised_transform.location.x),
        #                 np.float32(truck_normalised_transform.location.y),
        #                 np.float32(forward_velocity),
        #                 np.float32(acceleration),
        #                 np.float32(x_dist_to_next_waypoint),
        #                 np.float32(y_dist_to_next_waypoint),
        #                 np.float32(angle_to_center_of_lane_degrees),
        #                 #LIDAR
        #                 # Last action here?
        #             ], {}

    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

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
        if len(core.route) - 11 == core.last_waypoint_index:
            return True


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
        return bool(output)

    def compute_reward(self, observation, core):
        """Computes the reward"""
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)
        def compute_angle(u, v):
            return -math.atan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1])
        def find_current_waypoint(map_, hero):
            return map_.get_waypoint(hero.get_location(), project_to_road=False, lane_type=carla.LaneType.Any)
        def inside_lane(waypoint, allowed_types):
            if waypoint is not None:
                return waypoint.lane_type in allowed_types
            return False

        world = core.world
        hero = core.hero
        map_ = core.map

        reward = 0

        # Observations
        # np.float32(truck_normalised_transform.location.x),
        # np.float32(truck_normalised_transform.location.y),
        # np.float32(forward_velocity),
        # np.float32(acceleration),
        # np.float32(x_dist_to_next_waypoint),
        # np.float32(y_dist_to_next_waypoint),
        # np.float32(angle_to_center_of_lane_degrees),


        forward_velocity = observation['values'][2]
        angle_to_center_of_lane_degrees = observation['values'][5]
        # print(f"angle with center in REWARD {angle_to_center_of_lane_degrees}")

        reward_file = open(os.path.join("results",
                                        "run_" + str(core.current_time),
                                        "rewards_" + str(core.current_time) + ".txt")
                           , 'a+')


        print("Angle with center line %.5f " % (angle_to_center_of_lane_degrees*180) )
        # When the angle with the center line is 0 the highest reward is given
        if angle_to_center_of_lane_degrees == 0:
            reward += 1
            # print(f'====> REWARD for angle to center line is 0, R+= 1')
            reward_file.write(f"angle_to_center_of_lane_degrees == 0: +1 ")
        else:
            # Angle with the center line can deviate between 0 and 180 degrees
            # TODO Check this reward
            # Maybe this wil be too high?
            # Since the RL can stay there and get the reward
            reward += np.clip(1/(angle_to_center_of_lane_degrees*180),0,1)
            # print(f'====> REWARD for angle ({round(angle_to_center_of_lane_degrees,5)}) to center line { round(np.clip(1/(angle_to_center_of_lane_degrees*180),0,1),5)}',end='')
            reward_file.write(f"angle_to_center_of_lane_degrees is {round(angle_to_center_of_lane_degrees,5)}: {round(np.clip(1/(angle_to_center_of_lane_degrees*180),0,1),5)} ")


        # Positive reward for higher velocity
        # Already normalised in observations
        # reward += forward_velocity
        # print(f' REWARD for forward_velocity {forward_velocity} ')
        # reward_file.write(f"forward_velocity: {round(forward_velocity,5)} ")

        # Negative reward each time step to push for completing the task.
        # reward += -0.01
        # reward_file.write(f"negative reward: -0.01 ")




        # Current position and heading of the vehicle
        # velocity
        # Final position and heading

        # collision
        # laneInvasion
        # Time


        # # Hero-related variables
        # hero_location = hero.get_location()
        # # hero_velocity = self.get_speed(hero)
        # hero_heading = hero.get_transform().get_forward_vector()
        # hero_heading = [hero_heading.x, hero_heading.y]
        #
        # # Initialize last location
        # if self.last_location == None:
        #     self.last_location = hero_location

        # Compute deltas
        # delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + np.square(hero_location.y - self.last_location.y)))

        # Reward if going forward
        # reward = delta_distance
        # delta_velocity = hero_velocity - self.last_velocity


        # print('Distance to finish: ' + str(hero_dist_to_finish))
        # reward += 100 * (self.last_dist_to_finish - hero_dist_to_finish)

        # print('Delta distance ' + str(100 * (self.last_dist_to_finish - hero_dist_to_finish)))

        # Update variables
        # self.last_location = hero_location
        # self.last_velocity = hero_velocity
        # self.last_dist_to_finish = hero_dist_to_finish
        #




        # Reward if going faster than last step
        # if hero_velocity < 20.0:
        #     reward += 0.05 * delta_velocity

        # La duracion de estas infracciones deberia ser 2 segundos?
        # Penalize if not inside the lane
        # closest_waypoint = map_.get_waypoint(
        #     hero_location,
        #     project_to_road=False,
        #     lane_type=carla.LaneType.Any
        # )
        # if closest_waypoint is None or closest_waypoint.lane_type not in self.allowed_types:
        #     reward += -0.5
        #     self.last_heading_deviation = math.pi
        # else:
        #     if not closest_waypoint.is_junction:
        #         wp_heading = closest_waypoint.transform.get_forward_vector()
        #         wp_heading = [wp_heading.x, wp_heading.y]
        #         angle = compute_angle(hero_heading, wp_heading)
        #         self.last_heading_deviation = abs(angle)
        #
        #         if np.dot(hero_heading, wp_heading) < 0:
        #             # We are going in the wrong direction
        #             reward += -0.5
        #
        #         else:
        #             if abs(math.sin(angle)) > 0.4:
        #                 if self.last_action == None:
        #                     self.last_action = carla.VehicleControl()
        #
        #                 if self.last_action.steer * math.sin(angle) >= 0:
        #                     reward -= 0.05
        #     else:
        #         self.last_heading_deviation = 0



        if self.done_falling:
            reward += -1
            print('====> REWARD Done falling')
            reward_file.write(f"done_falling:-1 ")
        if self.done_collision:
            print("====> REWARD Done collision")
            reward += -1
            reward_file.write(f"done_collision:-1 ")
        if self.done_time_idle:
            print("====> REWARD Done idle")
            reward += -1
            reward_file.write(f"done_time_idle:-1 ")
        if self.done_time_episode:
            print("====> REWARD Done max time")
            reward += -1
            reward_file.write(f"done_time_episode:-1 ")
        if self.done_arrived:
            print("====> REWARD Done arrived")
            reward += 1
            reward_file.write(f"done_arrived:+1 ")

        # print('Reward: ' + str(reward))


        reward_file.write(f'FINAL REWARD {round(reward,5)} \n')
        reward_file.close()
        # print(f'Reward: {reward}')
        return reward