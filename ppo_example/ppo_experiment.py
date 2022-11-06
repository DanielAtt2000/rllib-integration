#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math
import numpy as np
from gym.spaces import Box, Discrete

import carla

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
        return Box(low=np.array([float("-inf"), float("-inf"),-1.0,0,float("-inf"),0,0]), high=np.array([float("inf"),float("inf"),1.0,1.0,float("inf"),20,20]), dtype=np.float32)


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

        print(f'Throttle {action.throttle} Steer {action.steer} Brake {action.brake} Reverse {action.reverse} Handbrake {action.hand_brake}')


        self.last_action = action

        return action

    def get_observation(self, sensor_data, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """

        # Current position and heading of the vehicle
        # velocity
        # Final position and heading

        # collision
        # laneInvasion
        # Time

        # changing location to value between 0 and 1
        # NORMALIZE
        # x_pos, y_pos = core.normalize_coordinates(observation["location"].location.x,
        #                                           observation["location"].location.y)

        transform = core.hero.get_transform()
        x_pos = transform.location.x
        y_pos = transform.location.y

        distToFinish = np.sqrt(np.square(np.float32(self.config["hero"]["final_location_x"]) - x_pos) + np.square(np.float32(self.config["hero"]["final_location_y"]) - y_pos))


        forward_velocity = np.clip(self.get_speed(core.hero), 0, None)
        forward_velocity = np.clip(forward_velocity, 0, 50.0) / 50
        heading = np.sin(transform.rotation.yaw * np.pi / 180)

        collisionCounter = 0
        laneInvasionCounter = 0
        for sensor in sensor_data:
            print('-----START1')

            print(sensor)
            print('-----END1')
            if sensor == 'collision':
                collisionCounter +=1
                # print('Collision Event')
                # print(sensor_data[sensor][1][0].semantic_tags)
                # print(sensor_data[sensor][1][0].type_id)

            # if sensor == 'obstacle':
            #     obstacle = sensor_data[sensor][1][0].type_id
            #     if 'static.road' != obstacle and 'static.terrain' != obstacle:
            #         print('Obstacle Event')
            #         print(obstacle)

            if sensor == 'laneInvasion':
                # print('Lane Invasion Event')

                laneInvasions = sensor_data[sensor][1][1]
                # print(laneInvasions)
                for laneInvasion in laneInvasions:
                    laneInvasionCounter +=1
                    # print(laneInvasion.type)

        # print(f"X Location {x_pos}")
        # print(f"Y Location {y_pos}")
        # print(f"Heading {heading}")
        # print(f"Forward Velocity {forward_velocity}")
        # print(f"Distance to finish {distToFinish}")
        # print(f"Collision Counter {collisionCounter}")
        # print(f"Lane Invasion Counter {laneInvasionCounter}")



        return np.r_[
                   np.float32(x_pos), np.float32(y_pos), np.float32(heading), np.float32(forward_velocity),
                   np.float32(distToFinish), np.float32(collisionCounter), np.float32(laneInvasionCounter)
               ], {}

    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def is_hero_near_finish_location(self, observation):
        final_x = self.config["hero"]["final_location_x"]
        final_y = self.config["hero"]["final_location_y"]

        if abs(observation[0] - final_x) < 0.5 and abs(observation[1] - final_y) < 0.5:
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
        self.done_collision = observation[5] > 0
        self.done_arrived = self.is_hero_near_finish_location(observation)

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
        hero_velocity = observation[3]
        hero_dist_to_finish = observation[4]
        hero_collision_counter = observation[5]
        hero_laneinvasion_counter = observation[6]



        # Current position and heading of the vehicle
        # velocity
        # Final position and heading

        # collision
        # laneInvasion
        # Time


        # Hero-related variables
        hero_location = hero.get_location()
        # hero_velocity = self.get_speed(hero)
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location

        # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + np.square(hero_location.y - self.last_location.y)))

        # Reward if going forward
        reward = delta_distance
        delta_velocity = hero_velocity - self.last_velocity


        # print('Distance to finish: ' + str(hero_dist_to_finish))
        reward += 100 * (self.last_dist_to_finish - hero_dist_to_finish)

        # print('Delta distance ' + str(100 * (self.last_dist_to_finish - hero_dist_to_finish)))

        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity
        self.last_dist_to_finish = hero_dist_to_finish





        # Reward if going faster than last step
        if hero_velocity < 20.0:
            reward += 0.05 * delta_velocity

        # La duracion de estas infracciones deberia ser 2 segundos?
        # Penalize if not inside the lane
        closest_waypoint = map_.get_waypoint(
            hero_location,
            project_to_road=False,
            lane_type=carla.LaneType.Any
        )
        if closest_waypoint is None or closest_waypoint.lane_type not in self.allowed_types:
            reward += -0.5
            self.last_heading_deviation = math.pi
        else:
            if not closest_waypoint.is_junction:
                wp_heading = closest_waypoint.transform.get_forward_vector()
                wp_heading = [wp_heading.x, wp_heading.y]
                angle = compute_angle(hero_heading, wp_heading)
                self.last_heading_deviation = abs(angle)

                if np.dot(hero_heading, wp_heading) < 0:
                    # We are going in the wrong direction
                    reward += -0.5

                else:
                    if abs(math.sin(angle)) > 0.4:
                        if self.last_action == None:
                            self.last_action = carla.VehicleControl()

                        if self.last_action.steer * math.sin(angle) >= 0:
                            reward -= 0.05
            else:
                self.last_heading_deviation = 0

        # Negative reward for changing lane
        reward += 40 * -hero_laneinvasion_counter

        if self.done_falling:
            reward += -100
        if self.done_collision:
            # print("Done collision")
            reward += -100
        if self.done_time_idle:
            # print("Done idle")
            reward += -100
        if self.done_time_episode:
            # print("Done max time")
            reward += 100
        if self.done_arrived:
            # print("Done arrived")
            reward += 100
        # print('Reward: ' + str(reward))
        return reward
