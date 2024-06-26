#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import argparse
import pickle
import time
from statistics import mean

import yaml

import ray
from ray.rllib.algorithms.ppo import PPO

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from ppo.ppo_paper import PPOExperimentBasic
import csv
from git import Repo
# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = PPOExperimentBasic

# RUN FUNCTION
# python3 ./ppo_paper_inference_ray.py ppo/ppo_config_paper.yaml
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
    if auto:
        argparser.add_argument(
            "run_type",
            type=str,
            help="run_type")
        argparser.add_argument(
            "map",
            type=str,
            help="map")


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
        inference_run = []
        inference_run.append(args.run_type)
        inference_run.append(args.map)

        if args.run_type == 'training':
            if args.map == 'mediumRoundabout4':
                inference_run.append(13)
            elif args.map == 'doubleRoundabout37':
                inference_run.append(39)
            else:
                raise Exception()
        elif args.run_type == 'testing':
            if args.map == 'mediumRoundabout4':
                inference_run.append(7)
            elif args.map == '20m':
                inference_run.append(16)
            else:
                raise Exception()
        else:
            raise Exception()

        save_dir = f"inference_results/final/{commit_hash}/{inference_run[0]}/"
        args.config["env_config"]["experiment"]["town1"] = inference_run[1]
        save_to_pickle('server_maps', [inference_run[1]])

        if inference_run[1] == 'mediumRoundabout4':
            save_to_pickle('mediumRoundabout4Type', inference_run[0])
            time.sleep(2)

        numbers_of_times_per_route = 3
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

        employee_writer.writerow(['route','timesteps','collision_truck','collision_trailer','timeout','high_angle','high_distance','distance_to_center_of_lane','trailer_distance_to_center_of_lane','completed'])


        while True:
            observation, _ = env.reset()
            done = False
            info = None
            counter = 0
            distance_to_center_of_lane = []
            trailer_distance_to_center_of_lane = []
            if total_episodes == 0:
                print('All episodes completed')
                break
            total_episodes -= 1

            while not done:
                action = agent.compute_single_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                distance_to_center_of_lane.append(info['distance_to_center_of_lane'])
                trailer_distance_to_center_of_lane.append(info['trailer_distance_to_center_of_lane'])
                counter +=1

            # ['route', 'timesteps', 'collision_truck', 'collision_trailer', 'timeout','lidar_collision_truck','lidar_collision_trailer','distance_to_center_of_lane', 'completed']
            employee_writer.writerow([f'{env.env_start_spawn_point}|{env.env_stop_spawn_point}', counter, info['done_collision_truck'],info['done_collision_trailer'],info['done_time_idle'] or info['done_time_episode'],info['done_angle'],info['done_distance'], mean(distance_to_center_of_lane), mean(trailer_distance_to_center_of_lane),info['done_arrived']])
            results_file.flush()
            # Resetting Variables
            env.done_collision_truck = False
            env.done_collision_trailer = False
            env.done_time = False
            env.done_arrived = False
            env.env_start_spawn_point = -1
            env.env_stop_spawn_point = -1
            distance_to_center_of_lane = []
            trailer_distance_to_center_of_lane = []

    except KeyboardInterrupt:
        print("\nshutdown by user")
    finally:
        save_to_pickle('mediumRoundabout4Type', '')
        print(f'\n\n Done running inference for {inference_run[0]} {inference_run[1]} {inference_run[2]}\n\n')
        ray.shutdown()
        # kill_all_servers()

if __name__ == "__main__":
    x = input('have you made one after the other true? (y/n) ')
    if x != 'y':
        raise Exception()

    run = 1
    run_all = False

    if run_all and run != 2:
        commit_hash = "257465bf"
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
    elif run == 2:
        commit_hash = ("98bf3e6e_2a98820b")
        checkpoint = '/home/daniel/ray_results/carla_rllib/ppo_98bf3e6ed7_DO_NOT_DELETE/CustomPPOTrainer_CarlaEnv_21da6_00000_0_2023-10-22_21-56-47/checkpoint_005750'
        x = input(f'Confirm saving to commit hash {commit_hash}? (y/n): ')
        if x != 'y':
            raise Exception()
        x = input(f'Ensure that medium is not run twice? (y/n): ')
        if x != 'y':
            raise Exception()

        print(f'python3 ./ppo_paper_inference_ray.py ppo/ppo_config_paper.yaml "{checkpoint}" "training" "mediumRoundabout4"')
        print(f'python3 ./ppo_paper_inference_ray.py ppo/ppo_config_paper.yaml "{checkpoint}" "training" "doubleRoundabout37"')
        print(f'python3 ./ppo_paper_inference_ray.py ppo/ppo_config_paper.yaml "{checkpoint}" "testing" "20m"')
        print(f'AFTERAFTERAFTER')
        print(f'python3 ./ppo_paper_inference_ray.py ppo/ppo_config_paper.yaml "{checkpoint}" "testing" "mediumRoundabout4"')
        print(f'AFTERAFTERAFTER')
        main(auto=True,commit_hash=commit_hash)
    else:
        main(auto=False)
