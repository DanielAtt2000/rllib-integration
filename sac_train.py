#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""SAC Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from __future__ import print_function

import argparse
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

from sac.sac_experiment_basic import SACExperimentBasic
from sac.sac_callbacks import SACCallbacks
from sac.sac_trainer import CustomSACTrainer

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = SACExperimentBasic

def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run(args):
    try:
        os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
        ray.init( num_gpus=1,include_dashboard=True,_temp_dir="/home/daniel/data-rllib-integration/ray_logs")
        analysis = tune.run(CustomSACTrainer,
                 name=args.name,
                 local_dir=args.directory,
                 # stop={"perf/ram_util_percent": 85.0},
                 checkpoint_freq=1000,
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

def save_commit_hash_to_file(args, commit_hash):
    random_file_no = 0

    while os.path.exists(f'commit_hashes/{random_file_no}.txt'):
        random_file_no += 1

    args.config['env_config']['experiment']['hero']['random_file_no'] = random_file_no

    with open(f'commit_hashes/{random_file_no}.txt','w') as file:
        file.write(commit_hash)

def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS
        config["callbacks"] = SACCallbacks

    return config


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
                           default="sac_example",
                           help="Name of the experiment (default: sac_example)")
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

    save_commit_hash_to_file(args,str(commit_hash()))

    path = os.path.join(args.directory, args.name + '_' + str(commit_hash()))


    launch_tensorboard(logdir= path,
                       host="localhost", port="6010")


    specific_version = False
    check_commit = True

    save_to_pickle('waiting_times',[0,20,60,80,100, 120,140,160,180])

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
