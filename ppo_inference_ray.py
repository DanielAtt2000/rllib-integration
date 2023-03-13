self.lidar_max_points#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import argparse
import yaml

import ray
from ray.rllib.algorithms.ppo import PPO

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from ppo_example.ppo_experiment import PPOExperiment

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = PPOExperiment

# RUN FUNCTION
# python3 ./ppo_inference_ray.py ppo_example/ppo_config.yaml "/home/daniel/ray_results/carla_rllib/ppo_77e99cc55c/CustomPPOTrainer_CarlaEnv_e71d2_00000_0_2023-01-21_15-14-24/checkpoint_000571"
#
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

        # Restore agent
        agent = PPO(env=CarlaEnv, config=args.config)
        agent.restore(args.checkpoint)

        # Initalize the CARLA environment
        env = agent.workers.local_worker().env
        obs = env.reset()

        while True:
            action = agent.compute_single_action(obs)
            obs, _, _, _ = env.step(action)

    except KeyboardInterrupt:
        print("\nshutdown by user")
    finally:
        ray.shutdown()
        kill_all_servers()

if __name__ == "__main__":

    main()
