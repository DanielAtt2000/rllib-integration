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

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperimentBasic

# RUN FUNCTION
# python3 ./dqn_inference_ray.py dqn/dqn_config.yaml "/home/daniel/ray_results/carla_rllib/dqn_9b664eb1e1/CustomDQNTrainer_CarlaEnv_fc10a_00000_0_2023-01-16_19-09-57/checkpoint_000219"
# /home/daniel/ray_results/carla_rllib/ppo_77e99cc55c/CustomPPOTrainer_CarlaEnv_e71d2_00000_0_2023-01-21_15-14-24/checkpoint_000571
# /home/daniel/ray_results/carla_rllib/dqn_8138f8582f/CustomDQNTrainer_CarlaEnv_93bce_00000_0_2023-01-30_18-07-29/checkpoint_000750
# /home/daniel/ray_results/carla_rllib/dqn_ea8eefa922/CustomDQNTrainer_CarlaEnv_16957_00000_0_2023-02-13_23-52-27/checkpoint_000305
# /home/daniel/ray_results/carla_rllib/dqn_53bd0b14d1/CustomDQNTrainer_CarlaEnv_e07a8_00000_0_2023-03-01_21-08-48/checkpoint_000419
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
        agent = DQN(env=CarlaEnv, config=args.config)
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
