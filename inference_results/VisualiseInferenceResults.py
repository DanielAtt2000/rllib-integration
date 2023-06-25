import random

import numpy as np
import pandas as pd

results = pd.read_csv(
    'latest/combined/easy/_home_daniel_ray_results_carla_rllib_sac_4c0293c613_CustomSACTrainer_CarlaEnv_b1f1d_00000_0_2023-06-24_10-54-14_checkpoint_027000.csv')
results = results[results.route != 'route']
routes = results['route'].unique()

print('HAVE YOU ENSURED THAT IF YOU MEREGED TO ANALYSIS TOGETHER THAT THEY DO NOT CONTAIN THE SAME ROUTES?')
x = random.randint(0,10)
if x < 5:
    input('Yes?')

results = results.replace(['False'],0)
results = results.replace(['True'],1)


results['distance_to_center_of_lane'] = results['distance_to_center_of_lane'].astype('float32')
results['timesteps'] = results['timesteps'].astype('int')

final_results = pd.DataFrame(columns=['route', 'timesteps', 'collision_truck', 'collision_trailer', 'timeout', 'truck_lidar_collision',
     'trailer_lidar_collision', 'distance_to_center_of_lane', 'completed'])
total_runs = 0
total_success = 0
total_unsuccessful = 0
for route in routes:
    try:
        indicies = results.loc[results['route'] == route]
        no_of_runs = len(indicies)
        total_runs += no_of_runs
        x = indicies.sum()
        no_of_runs_successful = x['completed']
        total_success += no_of_runs_successful
        no_of_runs_unsuccessful = x['collision_truck'] + x['collision_trailer'] + x['timeout'] + x['truck_lidar_collision'] + x['trailer_lidar_collision']
        total_unsuccessful += no_of_runs_unsuccessful
        x['route'] = route
        x['timesteps'] = x['timesteps']/no_of_runs
        x['distance_to_center_of_lane'] = x['distance_to_center_of_lane']/no_of_runs
        # if int(x['route'].split('|')[0]) in [35,69,27,4,64]:
        # if no_of_runs_successful < 15:
        print('----------------')
        print(f'Route {x["route"]} tested {no_of_runs} times')
        print(x)
        print(f'Success rate for route {x["route"]} = {no_of_runs_successful/no_of_runs}')
        print(f'Unsuccessful rate for route {x["route"]} = {no_of_runs_unsuccessful/no_of_runs}')
    except:
        raise Exception('wtf happened')
print('OVERLALL RESULTS')
print(f'Total Runs {total_runs}')
print(f'Success rate overall = {total_success / total_runs}')
print(f'Unsuccessful rate overall = {total_unsuccessful / total_runs}')

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
print('HAVE YOU ENSURED THAT IF YOU MEREGED TO ANALYSIS TOGETHER THAT THEY DO NOT CONTAIN THE SAME ROUTES?')
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')