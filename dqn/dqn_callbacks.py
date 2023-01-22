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
        episode.user_data["forward_velocity"] = []
        episode.user_data["custom_done_arrived"] = -1

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        # Angle with center
        last_angle_with_center = worker.env.experiment.last_angle_with_center
        # if last_angle_with_center >= 0:
        episode.user_data["angle_with_center"].append(last_angle_with_center)

        # Forward Velocity
        last_forward_velocity = worker.env.experiment.last_forward_velocity
        episode.user_data["forward_velocity"].append(last_forward_velocity)
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        last_angle_with_center = episode.user_data["angle_with_center"]
        if len(last_angle_with_center) > 0:
            last_angle_with_center = np.mean(episode.user_data["angle_with_center"])
        else:
            last_angle_with_center = 0
        episode.custom_metrics["angle_with_center"] = last_angle_with_center

        # Forward Velocity
        last_forward_velocity = episode.user_data["forward_velocity"]
        if len(last_forward_velocity) > 0:
            last_forward_velocity = np.mean(episode.user_data["forward_velocity"])
        else:
            last_forward_velocity = 0
        episode.custom_metrics["forward_velocity"] = last_forward_velocity

        if not worker.env.experiment.custom_done_arrived:
            episode.custom_metrics["custom_done_arrived"] = 0

        elif worker.env.experiment.custom_done_arrived:
            episode.custom_metrics["custom_done_arrived"] = 1

        else:
            episode.custom_metrics["custom_done_arrived"] = -2
