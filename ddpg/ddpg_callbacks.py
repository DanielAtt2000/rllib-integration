#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from rllib_integration.GetStartStopLocation import spawn_points_2_lane_roundabout_small_difficult, \
    spawn_points_2_lane_roundabout_small_easy, lower_medium_roundabout_difficult, lower_medium_roundabout_easy


def get_route_type(current_entry_idx, current_exit_idx):
    found = False
    easy = False
    difficult = False
    for entry_easy in (spawn_points_2_lane_roundabout_small_easy+lower_medium_roundabout_easy):
        entry_idx = entry_easy[0]
        if current_entry_idx == entry_idx:
            if current_exit_idx in entry_easy[1]:
                found = True
                easy = True
                break

    if not found:
        for entry_difficult in (spawn_points_2_lane_roundabout_small_difficult+lower_medium_roundabout_difficult):
            entry_idx = entry_difficult[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_difficult[1]:
                    difficult = True
                    break

    if easy:
        return 'easy'
    elif difficult:
        return 'difficult'
    else:
        raise Exception('No path type found')


class DDPGCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["angle_with_center"] = []
        episode.user_data["forward_velocity"] = []
        episode.user_data["custom_done_arrived"] = -1
        episode.user_data["reward_proportional_to_length"] = []
        episode.user_data["total_reward"] = []
        episode.user_data["entry_idx"] = -1

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        # Angle with center
        last_angle_with_center = worker.env.experiment.last_angle_with_center
        # if last_angle_with_center >= 0:
        episode.user_data["angle_with_center"].append(last_angle_with_center)

        # Forward Velocity
        last_forward_velocity = worker.env.experiment.last_forward_velocity
        episode.user_data["forward_velocity"].append(last_forward_velocity)

        # Reward Proportional to length
        reward = worker.env.experiment.reward_metric
        episode.user_data["reward_proportional_to_length"].append(reward)

        # Total Reward
        episode.user_data["total_reward"].append(reward)

        # Entry Idx
        episode.user_data["entry_idx"] = worker.env.experiment.entry_idx
        episode.user_data["exit_idx"] = worker.env.experiment.exit_idx

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


        # Proportional Reward
        collision_reward_proportional_to_length = 0
        done_reward_proportional_to_length = 0
        both_reward_proportional_to_length = 0

        done_without_reward_proportional_to_length = 0
        collision_without_reward_proportional_to_length = 0
        both_without_reward_proportional_to_length = 0


        reward_proportional_to_length = episode.user_data["reward_proportional_to_length"]
        if len(reward_proportional_to_length) > 0:
            if reward_proportional_to_length[-1] > 9500:
                # Done Episode
                done_reward_proportional_to_length = np.mean(episode.user_data["reward_proportional_to_length"])
                done_without_reward_proportional_to_length = np.mean(episode.user_data["reward_proportional_to_length"][:-1])

                episode.custom_metrics["done_reward_proportional_to_length"] = done_reward_proportional_to_length
                episode.custom_metrics[
                    "done_without_reward_proportional_to_length"] = done_without_reward_proportional_to_length

            elif reward_proportional_to_length[-1] < -700:
                # Collision episode
                collision_reward_proportional_to_length = np.mean(episode.user_data["reward_proportional_to_length"])
                collision_without_reward_proportional_to_length = np.mean(episode.user_data["reward_proportional_to_length"][:-1])

                episode.custom_metrics[
                    "collision_reward_proportional_to_length"] = collision_reward_proportional_to_length
                episode.custom_metrics[
                    "collision_without_reward_proportional_to_length"] = collision_without_reward_proportional_to_length
            else:
                # print_error_message(reward_proportional_to_length[-1])
                pass

            both_reward_proportional_to_length = np.mean(episode.user_data["reward_proportional_to_length"])
            both_without_reward_proportional_to_length = np.mean(episode.user_data["reward_proportional_to_length"][:-1])

            episode.custom_metrics["both_reward_proportional_to_length"] = both_reward_proportional_to_length
            episode.custom_metrics[
                "both_without_reward_proportional_to_length"] = both_without_reward_proportional_to_length


        # Reward per roundabout
        path_type = get_route_type(current_entry_idx=episode.user_data["entry_idx"], current_exit_idx=episode.user_data["exit_idx"])

        if path_type == 'easy':
            episode.custom_metrics["easy_episode_reward"] = sum(episode.user_data["total_reward"])
            if not worker.env.experiment.custom_done_arrived:
                episode.custom_metrics["easy_custom_done_arrived"] = 0

            elif worker.env.experiment.custom_done_arrived:
                episode.custom_metrics["easy_custom_done_arrived"] = 1
        elif path_type == 'difficult':
            episode.custom_metrics["difficult_episode_reward"] = sum(episode.user_data["total_reward"])
            if not worker.env.experiment.custom_done_arrived:
                episode.custom_metrics["difficult_custom_done_arrived"] = 0

            elif worker.env.experiment.custom_done_arrived:
                episode.custom_metrics["difficult_custom_done_arrived"] = 1
        else:
            print(f"Entry {episode.user_data['entry_idx']}")
            print(f"Exit {episode.user_data['exit_idx']}")
            raise Exception('Something when wrong here')



def print_error_message(reward):
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")
    print("THIS CANT BE TRUE")
    print(f"Last value was {reward} ")
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")