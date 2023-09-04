#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""DDPG Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from __future__ import print_function

import argparse
import math
import os
import random
import pickle
import yaml

import ray
from ray import tune

from checker import check_with_user, commit_hash
from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from rllib_integration.helper import get_checkpoint, launch_tensorboard

from ddpg.ddpg_experiment_basic import DDPGExperimentBasic
from ddpg.ddpg_callbacks import DDPGCallbacks
from ddpg.ddpg_trainer import CustomDDPGTrainer

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DDPGExperimentBasic

def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run(args):
    try:
        os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
        ray.init( num_gpus=1,include_dashboard=True,_temp_dir="/home/daniel/data-rllib-integration/ray_logs")
        analysis = tune.run(CustomDDPGTrainer,
                 name=args.name,
                 local_dir=args.directory,
                 # stop={"perf/ram_util_percent": 85.0},
                 checkpoint_freq=3000,
                 # checkpoint_at_end=True,
                 restore=get_checkpoint(args.name, args.directory, args.restore, args.overwrite),
                 config=args.config,
                 # queue_trials=True,
                 resume=False,
                 reuse_actors=True,
                 )
        print("----------------HERE")
        print(analysis.__dict__)
        # print(analysis.get_all_configs())
        # print(analysis.get_best_trial())
        print("----------------HERE")
    finally:
        kill_all_servers()
        ray.shutdown()

# Memory usage on this node: 15.5/15.8 GiB: ***LOW MEMORY***
# less than 10% of the memory on this node is available for use. This can cause unexpected crashes.
# Consider reducing the memory used by your application or reducing the Ray object store size by setting `object_store_memory` when calling `ray.init`.

def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS
        config["callbacks"] = DDPGCallbacks

    return config
def get_server_maps_dist(config):
    num_workers = config['num_workers']
    town1 = config["env_config"]["experiment"]["town1"]
    town1Ratio = config["env_config"]["experiment"]["town1Ratio"]
    town2 = config["env_config"]["experiment"]["town2"]
    town2Ratio = config["env_config"]["experiment"]["town2Ratio"]


    assert town1Ratio+town2Ratio == 1

    if town1 == 'None':
        raise Exception('No town 1 entered')
    if town2 == 'None':
        inp = input('No town 2 entered confirm? (y/n): ')
        if inp != 'y':
            raise Exception('No town 2 entered')
    print('---------------------------------------')


    output = []

    if town2 == 'None':
        for i in range(num_workers):
            output.append(town1)
    else:
        if town1Ratio < town2Ratio:
            num_of_workers_for_town1 = math.floor(num_workers*town1Ratio)
            num_of_workers_for_town1 = 1 if num_of_workers_for_town1 == 0 else num_of_workers_for_town1

            num_of_workers_for_town2 = num_workers - num_of_workers_for_town1

        else:
            num_of_workers_for_town2 = math.floor(num_workers*town2Ratio)
            num_of_workers_for_town2 = 1 if num_of_workers_for_town2 == 0 else num_of_workers_for_town2

            num_of_workers_for_town1 = num_workers - num_of_workers_for_town2


        for i in range(num_of_workers_for_town1):
            output.append(town1)
        for j in range(num_of_workers_for_town2):
            output.append(town2)
    return output


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument("-d", "--directory",
                           metavar='D',
                           default=os.path.expanduser("~") + "/ray_results/carla_rllib",
                           help="Specified directory to save results (default: ~/ray_results/carla_rllib")
    argparser.add_argument("-n", "--name",
                           metavar="N",
                           default="ddpg_example",
                           help="Name of the experiment (default: ddpg_example)")
    argparser.add_argument("--restore",
                           action="store_true",
                           default=False,
                           help="Flag to restore from the specified directory")
    argparser.add_argument("--overwrite",
                           action="store_true",
                           default=False,
                           help="Flag to overwrite a specific directory (warning: all content of the folder will be lost.)")
    argparser.add_argument("--tboff",
                           action="store_true",
                           default=False,
                           help="Flag to deactivate Tensorboard")
    argparser.add_argument("--auto",
                           action="store_true",
                           default=False,
                           help="Flag to use auto address")


    args = argparser.parse_args()
    args.config = parse_config(args)

    path = os.path.join(args.directory, args.name + '_' + str(commit_hash()))


    launch_tensorboard(logdir= path,
                       host="localhost", port="6010")


    specific_version = False
    check_commit = True

    output = get_server_maps_dist(config=args.config)
    print(output)
    save_to_pickle('server_maps',output)
    save_to_pickle('waiting_times',[0,20,40,60,85,105,125,145,165,185,0,20,40,60,80,100,120,140,160,180])

    if check_with_user(check_commit):
        args.name = args.name + '_' + str(commit_hash())

        if specific_version:
            args.name = ""
            x = random.randint(0,100)
            inp = input(f'SPECIFIC NAME APPLIED  ENTER {x} to confirm:')

            if int(x) == int(inp):
                run(args)
        else:
            run(args)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
