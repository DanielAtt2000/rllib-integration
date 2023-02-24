#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import gym

from rllib_integration.carla_core import CarlaCore

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

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment.config)
        self.core.set_map_normalisation()
        # self.compute_action_time = []
        # self.tick_time = []
        # self.get_observation_time = []
        # self.get_done_status_time = []
        # self.compute_reward_time = []
        # self.all_time = []

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data, self.core)

        return observation

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
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
        done = self.experiment.get_done_status(observation, self.core)
        # stop = time.time()
        # time_taken = stop - start
        # self.get_done_status_time.append(time_taken)
        #
        # start = time.time()
        reward = self.experiment.compute_reward(observation, self.core)
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
