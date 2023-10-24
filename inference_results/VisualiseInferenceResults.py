import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rllib_integration.GetStartStopLocation import roundabout20m, lower_medium_roundabout_all, upper_medium_roundabout, \
    spawn_points_2_lane_roundabout_small_easy, spawn_points_2_lane_roundabout_small_difficult, \
    lower_medium_roundabout_easy, lower_medium_roundabout_difficult


def find_all(path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            result.append(os.path.join(root, file))
    return result

path = 'final/fe92e6d3_final/testing'
found_files = find_all(path)
assert len(found_files) == 1
results = pd.read_csv(found_files[0])
results = results[results.route != 'route']
routes = results['route'].unique()

print('HAVE YOU ENSURED THAT IF YOU MEREGED TO ANALYSIS TOGETHER THAT THEY DO NOT CONTAIN THE SAME ROUTES?')
x = random.randint(0,10)
if x < 5:
    input('Yes?')

results = results.replace(['False'],0)
results = results.replace(['True'],1)


results['distance_to_center_of_lane'] = results['distance_to_center_of_lane'].astype('float32')
results['trailer_distance_to_center_of_lane'] = results['trailer_distance_to_center_of_lane'].astype('float32')
results['timesteps'] = results['timesteps'].astype('int')

final_results = pd.DataFrame(columns=['route','roundabout', 'timesteps', 'collision_truck', 'collision_trailer', 'timeout', 'truck_lidar_collision',
     'trailer_lidar_collision', 'distance_to_center_of_lane','trailer_distance_to_center_of_lane','distance_to_center_of_lane_completed_only', 'completed'])
total_runs = 0
total_success = 0
total_unsuccessful = 0
total_unsuccessful_truck = 0
total_unsuccessful_trailer = 0
total_routes = 0
truck_total_distance_to_center_of_lane_completed_routes = 0
trailer_total_distance_to_center_of_lane_completed_routes = 0
total_completed_routes_at_least_one = 0

def get_route_type(current_entry_idx, current_exit_idx):
    found = False
    ifroundabout20m = False
    ifupper_medium_roundabout = False
    iflower_medium_roundabout_all = False
    ifdoubleRoundabout_roundabout_all = False
    for entry_easy in roundabout20m:
        entry_idx = entry_easy[0]
        if current_entry_idx == entry_idx:
            if current_exit_idx in entry_easy[1]:
                found = True
                ifroundabout20m = True
                break

    if not found:
        for entry_easy in upper_medium_roundabout:
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    ifupper_medium_roundabout = True
                    break
    if not found:
        for entry_easy in (lower_medium_roundabout_easy+lower_medium_roundabout_difficult):
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    iflower_medium_roundabout_all = True
                    break
    if not found:
        for entry_easy in (spawn_points_2_lane_roundabout_small_easy+spawn_points_2_lane_roundabout_small_difficult):
            entry_idx = entry_easy[0]
            if current_entry_idx == entry_idx:
                if current_exit_idx in entry_easy[1]:
                    found = True
                    ifdoubleRoundabout_roundabout_all = True
                    break

    if iflower_medium_roundabout_all:
        return '32m'
    elif ifupper_medium_roundabout:
        return '40m'
    elif ifroundabout20m:
        return '20m'
    elif ifdoubleRoundabout_roundabout_all:
        return 'double'
    else:
        raise Exception('fuc')



for route in routes:
    total_routes +=1
    try:
        indicies = results.loc[results['route'] == route]
        no_of_runs = len(indicies)
        total_runs += no_of_runs

        completed_routes = indicies.loc[indicies['completed'] == 1]
        combined_completed_routes = completed_routes.sum()
        if len(completed_routes) > 0:
            truck_average_distance_away_from_lane_for_completed_routes = combined_completed_routes['distance_to_center_of_lane']/len(completed_routes)
            trailer_average_distance_away_from_lane_for_completed_routes = combined_completed_routes['trailer_distance_to_center_of_lane']/len(completed_routes)
        else:
            truck_average_distance_away_from_lane_for_completed_routes = 0
            trailer_average_distance_away_from_lane_for_completed_routes = 0

        x = indicies.sum()
        no_of_runs_successful = x['completed']
        total_success += no_of_runs_successful
        no_of_runs_unsuccessful = x['collision_truck'] + x['collision_trailer'] + x['timeout'] + x['truck_lidar_collision'] + x['trailer_lidar_collision']
        total_unsuccessful += no_of_runs_unsuccessful
        total_unsuccessful_truck += x['truck_lidar_collision']
        total_unsuccessful_trailer += x['trailer_lidar_collision']
        truck_total_distance_to_center_of_lane_completed_routes +=truck_average_distance_away_from_lane_for_completed_routes
        trailer_total_distance_to_center_of_lane_completed_routes +=trailer_average_distance_away_from_lane_for_completed_routes
        total_completed_routes_at_least_one += 1 if len(completed_routes) > 0 else 0

        x['route'] = str(route)
        x['completed'] = x['completed']
        x['timesteps'] = x['timesteps']/no_of_runs
        x['distance_to_center_of_lane'] = x['distance_to_center_of_lane']/no_of_runs
        x['trailer_distance_to_center_of_lane'] = x['trailer_distance_to_center_of_lane']/no_of_runs
        x['roundabout'] = get_route_type(int(x['route'].split('|')[0]),int(x['route'].split('|')[1]))

        # x['route'] = total_routes
        # if no_of_runs_successful < 15:
        print('----------------')
        print(f'Route {x["route"]} tested {no_of_runs} times')
        print(x)
        final_results = final_results.append(x,ignore_index=True)
        print(f'Success rate for route {x["route"]} = {no_of_runs_successful/no_of_runs}')
        print(f'Unsuccessful rate for route {x["route"]} = {no_of_runs_unsuccessful/no_of_runs}')
        print(f'Truck collision rate for route {x["route"]} = {x["truck_lidar_collision"]/no_of_runs}')
        print(f'Trailer collision rate for route {x["route"]} = {x["trailer_lidar_collision"]/no_of_runs}')
        if no_of_runs_successful/no_of_runs != 1 and x['trailer_lidar_collision'] != no_of_runs_unsuccessful:
            pass
            # input(',HERE')
    except:
        raise Exception('Failure')
split_path = path.split('/')
final_results.to_csv(f'{split_path[0]}/{split_path[1]}/output/{split_path[1]}_{split_path[2]}_output.csv')
print('OVERLALL RESULTS')
print(f'Total Runs {total_runs}')
print(f'Success rate overall = {total_success / total_runs}')
print(f'Unsuccessful rate overall = {total_unsuccessful / total_runs}')
print(f'Truck collision rate overall = {total_unsuccessful_truck / total_runs}')
print(f'Trailer collision rate overall = {total_unsuccessful_trailer / total_runs}')
print(f'Mean TRUCK distance to center of lane for completed routes = {truck_total_distance_to_center_of_lane_completed_routes / total_completed_routes_at_least_one}')
print(f'Mean TRAILER distance to center of lane for completed routes = {trailer_total_distance_to_center_of_lane_completed_routes / total_completed_routes_at_least_one}')
print(f'Total Routes = {total_routes}')

import seaborn as sns
final_results = final_results.sort_values(by=['completed'])
final_results = final_results
# final_results=final_results[['completed','route','roundabout']]
# sns.barplot(data=final_results, x="route", y="completed")
final_results = final_results.loc[final_results['roundabout'] == '20m']
# Draw a nested barplot by species and sex

g = sns.catplot(
    data=final_results, kind="bar",
    x="route", y="completed", color="blue", alpha=0.9
)

g.set(xlabel='20m roundabout routes', ylabel='Number of successfully completed runs')
g.tick_params(axis='x', which='major', labelsize=7)
# g.despine(left=True)
# g.set_axis_labels("", "Body mass (g)")
# g.legend.set_title("")
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.show()
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
# print('HAVE YOU ENSURED THAT IF YOU MEREGED TO ANALYSIS TOGETHER THAT THEY DO NOT CONTAIN THE SAME ROUTES?')
# print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')