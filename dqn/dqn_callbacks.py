#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class DQNCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["angle_with_center"] = []

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        last_angle_with_center = worker.env.experiment.last_angle_with_center
        # if last_angle_with_center >= 0:
        episode.user_data["angle_with_center"].append(last_angle_with_center)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        last_angle_with_center = episode.user_data["angle_with_center"]
        if len(last_angle_with_center) > 0:
            last_angle_with_center = np.mean(episode.user_data["angle_with_center"])
        else:
            last_angle_with_center = 0
        episode.custom_metrics["angle_with_center"] = last_angle_with_center
