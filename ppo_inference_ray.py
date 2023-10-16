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
from ray.rllib.algorithms.ppo import PPO

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from ppo.ppo_experiment_basic import PPOExperimentBasic
import csv
from git import Repo
# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = PPOExperimentBasic

# RUN FUNCTION
# python3 ./ppo_inference_ray.py ppo/ppo_config.yaml
# /home/daniel/ray_results/carla_rllib/ppo_02152f79ae_DO_NOT_DELETE/CustomPPOTrainer_CarlaEnv_b14f2_00000_0_2023-09-30_07-53-34/checkpoint_005000
# /home/daniel/ray_results/carla_rllib/ppo_30e97a9090_DO_NOT_DELETE/CustomPPOTrainer_CarlaEnv_5fe5f_00000_0_2023-10-07_19-03-02/checkpoint_002500
# /home/daniel/ray_results/carla_rllib/ppo_f0a9aeaa3d_DO_NOT_DELETE/CustomPPOTrainer_CarlaEnv_4bfb9_00000_0_2023-10-14_10-54-34/checkpoint_001500
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

def main(auto=False,commit_hash='temp',inference_run=[]):

    argparser = argparse.ArgumentParser()
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument(
        "checkpoint",
        type=str,
        help="Checkpoint from which to roll out.")

    args = argparser.parse_args()
    args.config = parse_config(args)
    if not auto:
        save_dir = f"inference_results/run/"
        x = input(f'Please confirm save directory {save_dir}: (y/no)')
        if x != 'y':
            raise Exception('Cancelled')

        town1 = args.config["env_config"]["experiment"]["town1"]
        save_to_pickle('server_maps', [town1])

        x = input(f'Confrim using map {town1}? (y/n): ')
        if x != 'y':
            raise Exception('Failed')

        print('Medium Roundabout TRAINING 13 routes')
        print('Double Roundabout Training 39 ')
        print('Medium Roundabout Testing 7 routes ----> CHANGE IN GetStartStopLocation <----')
        print('20m Roundabout Testing 16 ')
        x = input('What are the total number of routes being tested?')
        numbers_of_times_per_route = 2
        total_episodes = (numbers_of_times_per_route + 2 ) * int(x)
    else:
        save_dir = f"inference_results/final/{commit_hash}/{inference_run[0]}/"
        args.config["env_config"]["experiment"]["town1"] = inference_run[1]
        save_to_pickle('server_maps', [inference_run[1]])

        if inference_run[1] == 'mediumRoundabout4':
            save_to_pickle('mediumRoundabout4Type', inference_run[0])

        numbers_of_times_per_route = 2
        total_episodes = (numbers_of_times_per_route + 2) * int(inference_run[2])
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
        agent = PPO(env=CarlaEnv, config=args.config)
        agent.restore(args.checkpoint)

        # Initalize the CARLA environment
        env = agent.workers.local_worker().env

        results_file = open(f'{save_dir}{args.checkpoint.replace("/","_")}.csv', mode='a')
        employee_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        employee_writer.writerow(['route','timesteps','collision_truck','collision_trailer','timeout','truck_lidar_collision','trailer_lidar_collision','distance_to_center_of_lane','completed'])


        while True:
            observation, _ = env.reset()
            done = False
            info = None
            counter = 0
            distance_to_center_of_lane = []
            if total_episodes == 0:
                print('All episodes completed')
                break
            total_episodes -= 1

            while not done:
                action = agent.compute_single_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
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
        save_to_pickle('mediumRoundabout4Type', '')
        ray.shutdown()
        # kill_all_servers()

if __name__ == "__main__":
    x = input('have you made one after the other true? (y/n) ')
    if x != 'y':
        raise Exception()

    run_all = True
    if run_all:
        commit_hash = "d5efe1c5"
        x = input(f'Confirm saving to commit hash {commit_hash}? (y/n): ')
        if x != 'y':
            raise Exception()
        runs = [
            ['training','mediumRoundabout4',13],
            ['training','doubleRoundabout37',39],
            ['testing','mediumRoundabout4',7],
            ['testing','20m',16]
        ]
        for run in runs:
            main(auto=True, commit_hash=commit_hash, inference_run=run)
    else:
        main(auto=False)
