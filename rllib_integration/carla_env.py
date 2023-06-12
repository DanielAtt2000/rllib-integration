#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import pickle
import cv2
import gym
from datetime import datetime
import pandas as pd
import time

from rllib_integration.carla_core import CarlaCore


def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(f'{filename}', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_pickle(filename):
    with open(filename + '.pickle', 'rb') as handle:
        return pickle.load(handle)


# import time
class CarlaEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """

    def __init__(self, config):
        """Initializes the environment"""
        self.config = config

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

        waiting_times = open_pickle('waiting_times')
        waiting_time = waiting_times.pop(0)
        save_to_pickle('waiting_times',waiting_times)

        print(f"Waiting for {waiting_time}")
        time.sleep(waiting_time)


        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment.config)
        self.core.set_map_normalisation()
        self.done_collision_truck = False
        self.done_collision_trailer = False
        self.done_time = False
        self.done_arrived = False
        self.env_start_spawn_point = -1
        self.env_stop_spawn_point = -1
        # self.compute_action_time = []
        # self.tick_time = []
        # self.get_observation_time = []
        # self.get_done_status_time = []
        # self.compute_reward_time = []
        # self.all_time = []

        def read_data_from_pickle(filename):
            with open(filename, 'rb') as handle:
                return pickle.load(handle)
            # try:
            #
            # except:
            #     return pd.DataFrame(columns = ['filename', 'done_collision'])

        self.date_time_format = "%m%d%Y_%H%M%S%f"
        self.counter = datetime.now().strftime(self.date_time_format)
        self.collision_data_file_name = "collision_data_inner_roundabout"
        # self.collision_data = read_data_from_pickle(f"/home/daniel/data-rllib-integration/image_data/{self.collision_data_file_name}.pkl")

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, info = self.experiment.get_observation(sensor_data, self.core)

        while info['truck_z_value'] > 0.1:
            sensor_data = self.core.tick(None)
            observation, info = self.experiment.get_observation(sensor_data, self.core)


        return observation

    def save_data(self, filename, data):
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        self.env_start_spawn_point = self.core.entry_spawn_point_index
        self.env_stop_spawn_point =  self.core.exit_spawn_point_index


        # start_main = time.time()
        # start = time.time()
        control = self.experiment.compute_action(action)
        # stop = time.time()
        # time_taken = stop - start
        # self.compute_action_time.append(time_taken)
        #
        # start = time.time()
        sensor_data = self.core.tick(control)
        # stop = time.time()
        # time_taken = stop - start
        # self.tick_time.append(time_taken)
        #
        # start = time.time()
        observation, info = self.experiment.get_observation(sensor_data, self.core)
        # stop = time.time()
        # time_taken = stop - start
        # self.get_observation_time.append(time_taken)
        #
        # start = time.time()
        done, self.done_collision_truck, self.done_collision_trailer, self.done_time, self.done_arrived = self.experiment.get_done_status(observation, self.core)

        save = False
        if save:
            cv2.imwrite(f'image_data/lidar/inner/{self.counter}.png',observation['occupancyMap'])

            # self.save_data(f'image_data/depth/{self.counter}.pkl',info['depth_camera'])

            temp_dataframe = pd.DataFrame({'filename': self.counter,'values':[observation['values']], 'done_collision': (self.done_collision_truck or self.done_collision_trailer)},index=[0])
            self.collision_data = pd.concat([self.collision_data,temp_dataframe], ignore_index=True)

            self.save_data(f'image_data/{self.collision_data_file_name}.pkl', self.collision_data)

            self.counter = datetime.now().strftime(self.date_time_format)
        # stop = time.time()
        # time_taken = stop - start
        # self.get_done_status_time.append(time_taken)
        #
        # start = time.time()
        reward = self.experiment.compute_reward(observation,info, self.core)
        # stop = time.time()
        # time_taken = stop - start
        # self.compute_reward_time.append(time_taken)
        #
        # stop_main =time.time()
        # self.all_time.append(stop_main-start_main)
        #
        # print(f"Average compute_action_time {sum(self.compute_action_time)/len(self.compute_action_time)}")
        # print(f"Average tick_time {sum(self.tick_time)/len(self.tick_time)}")
        # print(f"Average get_observation_time {sum(self.get_observation_time)/len(self.get_observation_time)}")
        # print(f"Average get_done_status_time {sum(self.get_done_status_time)/len(self.get_done_status_time)}")
        # print(f"Average compute_reward_time {sum(self.compute_reward_time)/len(self.compute_reward_time)}")
        # print(f"Average ALL {sum(self.all_time)/len(self.all_time)}")
        return observation, reward, done, info
