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
import random

import yaml
import ray
from ray import tune, air
import numpy as np
import gymnasium as gym
import math

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from checker import check_with_user, commit_hash
from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from rllib_integration.helper import get_checkpoint, launch_tensorboard

from dqn.dqn_experiment_basic import DQNExperimentBasic
from dqn.dqn_callbacks import DQNCallbacks
from dqn.dqn_trainer import CustomDQNTrainer

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperimentBasic
from ray.tune.schedulers import AsyncHyperBandScheduler,HyperBandScheduler,PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch

def run(args):
    try:
        os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
        ray.init( num_gpus=1,include_dashboard=True,_temp_dir="/home/daniel/rllib-integration/ray_logs")


        # Stop when we've either reached 100 training iterations or reward=300
        stopping_criteria = {"training_iteration": 60}

        # initial_params = [
        #     {"lr": 1, "height": 2, "activation": "relu"},
        #     {"width": 4, "height": 2, "activation": "relu"},
        # ]

        algo = OptunaSearch()
        algo = ConcurrencyLimiter(algo,max_concurrent=1)

        sch = AsyncHyperBandScheduler(
            # metric = "episode_reward_mean", mode="max"
        )

        tuner = tune.Tuner(
            CustomDQNTrainer,
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                search_alg = algo,
                scheduler=sch,
                num_samples=15,
                max_failures=-1,
            ),
            max_failures=-1,
            param_space={
                 "name": args.name,
                 "local_dir":args.directory,
                # To see the complete list of configurable parameters see:
                # https://github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
                "framework": "torch",
                "observation_space": gym.spaces.Box(
                low=np.array([0,0,-math.pi,-math.pi,-math.pi,-math.pi,-math.pi,-math.pi,-math.pi,0,-1,0,0,0,0,0,0,0,0,0,0,0]),
                high=np.array([100,100,math.pi,math.pi,math.pi,math.pi,math.pi,math.pi,math.pi,1,1,1,1,1,1,1,1,1,1,1,1,1]),
                dtype=np.float32
                ),
                "action_space": gym.spaces.Discrete(29),

                "num_workers": 1,
                "num_gpus": 1,
                "num_cpus_per_worker": 11,
                "recreate_failed_workers": True,
                "horizon": 6500,


                # "rollout_fragment_length": 4,
                "target_network_update_freq": 8000,
                "normalize_actions": False,

                # "batch_mode": "complete_episodes"
                "train_batch_size": tune.choice([32, 64,128]),
                # "num_steps_sampled_before_learning_starts": 10000,
                "n_step": tune.randint(1,11),
                "num_atoms": tune.choice([5,15,25]),
                "noisy": tune.choice([True, False]),
                "gamma": 0.99,
                "exploration_config": {
                  "type": "EpsilonGreedy",
                  "initial_epsilon": 1.0,
                   "final_epsilon": 0.01,
                   "epsilon_timesteps": 500000,
                },
                "replay_buffer_config": {
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 400000,
                    # How many steps of the model to sample before learning starts.
                    # If True prioritized replay buffer will be used.
                    # "prioritized_replay" : tune.choice([True, False]),
                    "prioritized_replay_alpha": 0.6,
                    "prioritized_replay_beta": 0.4,
                    "prioritized_replay_eps": 0.000001 ,

                },
                "model": {
                    "fcnet_hiddens":[128,256,512,1024]
                },
                "lr": tune.choice([0.0005, 0.00005,0.000005]),
                # "adam_epsilon": .00015,
                "min_sample_timesteps_per_iteration": 10000,
                "num_steps_sampled_before_learning_starts": 10000,

                "double_q": True,
                "dueling": True,

                "env" :CarlaEnv,
                "callbacks":DQNCallbacks,
                "env_config": {
                  "carla": {
                      "host": "192.168.1.113",
                      #    host: "172.17.0.1"
                      #   host: "127.0.0.1"
                      "programPort": "5418",
                      "timeout": 30.0,
                      # IF YOU ARE GOING TO CHANGE THE TIMESTEP CHANGE rotation_frequency of LIDAR
                      "timestep": 0.1, # IMP IMP
                      # IMP
                      "retries_on_error": 25,
                      "resolution_x": 300,
                      "resolution_y": 300,
                      "quality_level": "Low",
                      "enable_map_assets": True,
                      "enable_rendering": False,
                      "show_display": True,
                      "map_buffer": 1.2,
                      "truckTrailerCombo": True,
                  },



                    "experiment": {
                        "type":EXPERIMENT_CLASS,
                      "hero": {
                        "truckTrailerCombo": True,
                        "blueprintTruck": "vehicle.daf.dafxf",
                        #      blueprintTruck: "vehicle.audi.a2"
                        "blueprintTrailer": "vehicle.trailer.trailer",
                        "lidar_max_points": 3000,
                        "sensors": {
                          #        obstacle:
                          #          type: "sensor.other.obstacle"
                          "collision": {
                            "type": "sensor.other.collision",
                          }

                            #        depth_camera:
                            #          type: "sensor.camera.depth"
                            #          image_size_x: 84
                            #          image_size_y: 84
                            #          transform: '2.3,0.0,1.7,0.0,0.0,0.0' # x,y,z,pitch, yaw, roll
                            #        lidar:
                            #          type: "sensor.lidar.ray_cast_semantic"
                            #          channels : "32"
                            #          range : "50.0"
                            #          points_per_second : "50000"
                            #          rotation_frequency : "10" #  IMP THIS IS 1 / delta (timestep)
                            #          upper_fov : '5.0'
                            #          lower_fov : '-90.0'
                            #          horizontal_fov : '360.0'
                            #          sensor_tick : '0'
                            #          transform : '2,0.21,8,0.0,0.0,0.0' # x,y,z,pitch, yaw, roll
                            #        semantic_camera:
                            #          type: "sensor.camera.semantic_segmentation"
                            #          transform: '0.0,0.0,1.7,0.0,0.0,0.0' # x,y,z,pitch, yaw, roll
                            #        laneInvasion:
                            #          type: "sensor.other.lane_invasion"
                        }

                      },

                      "background_activity": {
                          "n_vehicles": 0,
                          "n_walkers": 0,
                          "tm_hybrid_mode": True,
                      },

                      #    town: "Town03_Opt"
                      "town": 'doubleRoundabout37',
                      "others": {
                          "framestack": 1,
                          "max_time_idle": 600,
                          "max_time_episode": 6400,
                      }

                    }

                }

            },
            run_config=air.RunConfig(
                 name=args.name,
                 local_dir=args.directory,
                stop=stopping_criteria,
                max_failures=-1,
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1)),
        )
        results = tuner.fit()

        import pprint

        best_result = results.get_best_result()
        print("Best hyperparameters found were: ", results.get_best_result().config)

        df = results.get_dataframe()
        print(df)


        print("\nBest performing trial's final reported metrics:\n")

        metrics_to_print = [
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
        ]
        pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

        file = open("results_dataframes/" + args.name + '_' + str(commit_hash()) + '.md','w')
        file.write(df.to_markdown())
        file.close()
        print(df.to_markdown())
        # tune.run(CustomDQNTrainer,
        #          name=args.name,
        #          local_dir=args.directory,
        #          # stop={"perf/ram_util_percent": 85.0},
        #          checkpoint_freq=1,
        #          # checkpoint_at_end=True,
        #          restore=get_checkpoint(args.name, args.directory, args.restore, args.overwrite),
        #          config=args.config,
        #          # queue_trials=True,
        #          resume=False,
        #          reuse_actors=True,
        #          scheduler=hyperband,
        #
        # )

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
        config["callbacks"] = DQNCallbacks

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
                           default="dqn_example",
                           help="Name of the experiment (default: dqn_example)")
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
                       host="localhost")

    specific_version = False
    check_commit = False

    if check_with_user(check_commit):
        args.name = args.name + '_' + str(commit_hash())

        if specific_version:
            args.name = "dqn_8f844bf22a"
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
