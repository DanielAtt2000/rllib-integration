#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""DQN Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from __future__ import print_function

import argparse
import os
import yaml

import ray
from ray import tune

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from rllib_integration.helper import get_checkpoint, launch_tensorboard

from ppo_example.ppo_experiment import PPOExperiment
from ppo_example.ppo_callbacks import PPOCallbacks
from ppo_example.ppo_trainer import CustomPPOTrainer

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = PPOExperiment


def run(args):
    try:
        ray.init( num_gpus=1)
        tune.run('SAC',
                 stop={"timesteps_total": 1000000},
                 name=args.name,
                 local_dir=args.directory,
                 # stop={"perf/ram_util_percent": 85.0},
                 checkpoint_freq=1,
                 # checkpoint_at_end=True,
                 # restore=get_checkpoint(args.name, args.directory,
                 #                        args.restore, args.overwrite),
                 config=args.config,
                 # queue_trials=True
                 resume=False,
                 reuse_actors=True
        )

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
        config["callbacks"] = PPOCallbacks

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
                           default="ppo_example",
                           help="Name of the experiment (default: ppo_example)")
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

    if not args.tboff:
        launch_tensorboard(logdir=os.path.join(args.directory, args.name),
                           host="0.0.0.0" if args.auto else "localhost")

    run(args)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
