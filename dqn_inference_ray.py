#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import argparse
import yaml

import ray
from ray.rllib.algorithms.dqn import DQN

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from dqn.dqn_experiment_basic import DQNExperimentBasic
import csv
from git import Repo
# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperimentBasic

# RUN FUNCTION
# python3 ./dqn_inference_ray.py dqn/dqn_config.yaml "/home/daniel/ray_results/carla_rllib/dqn_9b664eb1e1/CustomDQNTrainer_CarlaEnv_fc10a_00000_0_2023-01-16_19-09-57/checkpoint_000219"
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


        # Restore agent
        agent = DQN(env=CarlaEnv, config=args.config)
        agent.restore(args.checkpoint)

        # Initalize the CARLA environment
        env = agent.workers.local_worker().env

        results_file = open(f'inference_results/{str(remote.refs[repo.active_branch.name].commit)[:11]}_{args.checkpoint.replace("/","_")}.csv', mode='a')
        employee_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        employee_writer.writerow(['route','timesteps','collision_truck','collision_trailer','timeout','completed'])


        while True:
            observation = env.reset()
            done = False
            counter = 0
            while not done:
                action = agent.compute_single_action(observation)
                observation, reward, done, info = env.step(action)
                counter +=1

            # ['route', 'timesteps', 'collision_truck', 'collision_trailer', 'timeout', 'completed']
            employee_writer.writerow([f'{env.env_start_spawn_point}|{env.env_stop_spawn_point}', counter, env.done_collision_truck,env.done_collision_trailer,env.done_time,env.done_arrived])
            results_file.flush()
            # Resetting Variables
            env.done_collision_truck = False
            env.done_collision_trailer = False
            env.done_time = False
            env.done_arrived = False
            env.env_start_spawn_point = -1
            env.env_stop_spawn_point = -1

    except KeyboardInterrupt:
        print("\nshutdown by user")
    finally:
        ray.shutdown()
        kill_all_servers()

if __name__ == "__main__":

    main()
