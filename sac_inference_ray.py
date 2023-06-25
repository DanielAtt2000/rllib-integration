#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import argparse
import pickle
from statistics import mean

import yaml

import ray
from ray.rllib.algorithms.sac import SAC

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from sac.sac_experiment_basic import SACExperimentBasic
import csv
from git import Repo
# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = SACExperimentBasic

# RUN FUNCTION
# python3 ./sac_inference_ray.py sac/sac_config.yaml
# /home/daniel/ray_results/carla_rllib/ppo_77e99cc55c/CustomPPOTrainer_CarlaEnv_e71d2_00000_0_2023-01-21_15-14-24/checkpoint_000571
# /home/daniel/ray_results/carla_rllib/dqn_8138f8582f/CustomDQNTrainer_CarlaEnv_93bce_00000_0_2023-01-30_18-07-29/checkpoint_000750
# /home/daniel/ray_results/carla_rllib/dqn_ea8eefa922/CustomDQNTrainer_CarlaEnv_16957_00000_0_2023-02-13_23-52-27/checkpoint_000305
# /home/daniel/ray_results/carla_rllib/dqn_53bd0b14d1/CustomDQNTrainer_CarlaEnv_e07a8_00000_0_2023-03-01_21-08-48/checkpoint_000419
# /home/daniel/ray_results/carla_rllib/good/dqn_00a7d0841e_smallroundaboutOnly_depthCamera/CustomDQNTrainer_CarlaEnv_80d2d_00000_0_2023-02-24_19-31-50/checkpoint_000358
# /home/daniel/ray_results/carla_rllib/good/dqn_66b0e183bf_truck_lidar_240x320/CustomDQNTrainer_CarlaEnv_5b56a_00000_0_2023-03-05_18-01-24/checkpoint_000260
# /home/daniel/ray_results/carla_rllib/good/dqn_6d5fa11796_3_lidar_images128x128/CustomDQNTrainer_CarlaEnv_9a3de_00000_0_2023-03-12_17-54-50/checkpoint_000412
# /home/daniel/ray_results/carla_rllib/good/dqn_8847e01844_2_lidar_images_84x84/CustomDQNTrainer_CarlaEnv_97af5_00000_0_2023-03-13_22-39-54/checkpoint_000313
# /home/daniel/ray_results/carla_rllib/good/dqn_8847e01844_2_lidar_images_84x84/CustomDQNTrainer_CarlaEnv_97af5_00000_0_2023-03-13_22-39-54/checkpoint_000205
# /home/daniel/ray_results/carla_rllib/good/dqn_6f31211ee7_2_lidar_images_with_hypreward_+angle_reward/CustomDQNTrainer_CarlaEnv_9f625_00000_0_2023-03-15_22-59-13/checkpoint_000250
# /home/daniel/ray_results/carla_rllib/dqn_53b9a7ee09/CustomDQNTrainer_CarlaEnv_8a877_00000_0_2023-03-16_08-52-46/checkpoint_000549
# /home/daniel/ray_results/carla_rllib/good/dqn_0a9d414623_no_lidar_2_ahead_waypoints/CustomDQNTrainer_CarlaEnv_34b01_00000_0_2023-03-17_08-27-43/checkpoint_000099
# /home/daniel/ray_results/carla_rllib/sac_933af966a6/CustomSACTrainer_CarlaEnv_44868_00000_0_2023-05-06_18-15-50/checkpoint_045000
# /home/daniel/ray_results/carla_rllib/sac_9496eb1e82/CustomSACTrainer_CarlaEnv_7ea37_00000_0_2023-06-21_01-45-10/checkpoint_019000
# /home/daniel/ray_results/carla_rllib/sac_725b71c70c/CustomSACTrainer_CarlaEnv_92ed2_00000_0_2023-06-22_07-56-47/checkpoint_019000
# /home/daniel/ray_results/carla_rllib/sac_564062f528/CustomSACTrainer_CarlaEnv_527af_00000_0_2023-06-22_19-15-01/checkpoint_020000
# /home/daniel/ray_results/carla_rllib/sac_e6a772bf3f/CustomSACTrainer_CarlaEnv_0d727_00000_0_2023-06-24_00-48-21/checkpoint_024000
# /home/daniel/ray_results/carla_rllib/sac_4c0293c613/CustomSACTrainer_CarlaEnv_b1f1d_00000_0_2023-06-24_10-54-14/checkpoint_027000
def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS
        config["num_workers"] = 0
        config["explore"] = False
        del config["num_cpus_per_worker"]
        # del config["num_gpus_per_worker"]

    return config

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument(
        "checkpoint",
        type=str,
        help="Checkpoint from which to roll out.")

    args = argparser.parse_args()
    args.config = parse_config(args)

    try:
        ray.init()

        previous_routes_files = open('testing_routes.txt', 'w')
        previous_routes_files.write(f"roundabout_idx:{0}\n")
        previous_routes_files.write(f"entry_idx:{0}\n")
        previous_routes_files.write(f"exit_idx:{0}\n")
        previous_routes_files.close()


        repo = Repo('.')
        remote = repo.remote('origin')
        remote.fetch()

        save_to_pickle('waiting_times', [0, 20, 30, 40, 50])


        # Restore agent
        agent = SAC(env=CarlaEnv, config=args.config)
        agent.restore(args.checkpoint)

        # Initalize the CARLA environment
        env = agent.workers.local_worker().env

        results_file = open(f'inference_results/latest/medium/easy/{args.checkpoint.replace("/","_")}.csv', mode='a')
        employee_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        employee_writer.writerow(['route','timesteps','collision_truck','collision_trailer','timeout','truck_lidar_collision','trailer_lidar_collision','distance_to_center_of_lane','completed'])


        while True:
            observation = env.reset()
            done = False
            info = None
            counter = 0
            distance_to_center_of_lane = []
            while not done:
                action = agent.compute_single_action(observation)
                observation, reward, done, info = env.step(action)
                distance_to_center_of_lane.append(info['distance_to_center_of_lane'])
                counter +=1

            # ['route', 'timesteps', 'collision_truck', 'collision_trailer', 'timeout','lidar_collision_truck','lidar_collision_trailer','distance_to_center_of_lane', 'completed']
            employee_writer.writerow([f'{env.env_start_spawn_point}|{env.env_stop_spawn_point}', counter, info['done_collision_truck'],info['done_collision_trailer'],info['done_time_idle'] or info['done_time_episode'],info['truck_lidar_collision'],info['trailer_lidar_collision'], mean(distance_to_center_of_lane),info['done_arrived']])
            results_file.flush()
            # Resetting Variables
            env.done_collision_truck = False
            env.done_collision_trailer = False
            env.done_time = False
            env.done_arrived = False
            env.env_start_spawn_point = -1
            env.env_stop_spawn_point = -1
            distance_to_center_of_lane = []

    except KeyboardInterrupt:
        print("\nshutdown by user")
    finally:
        ray.shutdown()
        kill_all_servers()

if __name__ == "__main__":

    main()
