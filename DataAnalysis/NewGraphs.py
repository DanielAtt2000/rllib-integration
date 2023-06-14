from copy import deepcopy
from datetime import datetime
from math import ceil
# from statistics import mean
from numpy import mean

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import pandas as pd
import numpy as np

from DataAnalysis.NewProcessData2 import data_file, for_graphs
from rllib_integration.GetStartStopLocation import spawn_points_2_lane_roundabout_small_easy, \
    spawn_points_2_lane_roundabout_small_difficult


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Lidar:
    def __init__(self, time, front, front_left, front_right, left, right, trailer_0_left, trailer_1_left,
                 trailer_2_left,
                 trailer_3_left, trailer_4_left, trailer_5_left, trailer_0_right, trailer_1_right, trailer_2_right,
                 trailer_3_right, trailer_4_right, trailer_5_right):
        self.time = time
        self.front = front
        self.front_left = front_left
        self.front_right = front_right
        self.left = left
        self.right = right
        self.trailer_0_left = trailer_0_left
        self.trailer_1_left = trailer_1_left
        self.trailer_2_left = trailer_2_left
        self.trailer_3_left = trailer_3_left
        self.trailer_4_left = trailer_4_left
        self.trailer_5_left = trailer_5_left
        self.trailer_0_right = trailer_0_right
        self.trailer_1_right = trailer_1_right
        self.trailer_2_right = trailer_2_right
        self.trailer_3_right = trailer_3_right
        self.trailer_4_right = trailer_4_right
        self.trailer_5_right = trailer_5_right


def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(f'{os.path.join(directory, filename)}', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


directory = data_file
df = pd.DataFrame()
string = ''
largest_filename = "angle_to_center_of_lane_degrees.pkl"
with open(os.path.join(directory, largest_filename), 'rb') as handle:
    data = pickle.load(handle)
    df = data




for filename in os.listdir(directory):
    if largest_filename in filename:
        continue
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file) and file.endswith('.pkl'):
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            df = pd.merge(df,data, on='Time')
#             df = df.drop_duplicates(subset=['Time'])
#
# df = df.reset_index()

def plot_route(route_points_all, truck_points_all):
    x_route = []
    y_route = []

    x_truck = []
    y_truck = []

    try:
        x_truck = open_pickle(os.path.join(directory, 'x_truck.pickle'))
        y_truck = open_pickle(os.path.join(directory, 'y_truck.pickle'))
        x_route = open_pickle(os.path.join(directory, 'x_route.pickle'))
        y_route = open_pickle(os.path.join(directory, 'y_route.pickle'))

    except:
        route_points_all = route_points_all[route_points_all != 0]
        for route_id, route in enumerate(route_points_all):
            temp_x_route = []
            temp_y_route = []
            for point in route:
                temp_x_route.append(point[0])
                temp_y_route.append(point[1])

            x_route.append(temp_x_route)
            y_route.append(temp_y_route)
            print(f"Route {route_id}/{len(route_points_all)} ready")

        truck_points_all = truck_points_all[truck_points_all != 0]
        for truck_idx, truck_points in enumerate(truck_points_all):
            temp_x_truck = []
            temp_y_truck = []
            for truck_point in truck_points:
                temp_x_truck.append(truck_point[0])
                temp_y_truck.append(truck_point[1])

            x_truck.append(temp_x_truck)
            y_truck.append(temp_y_truck)
            print(f"Truck Path {truck_idx}/{len(truck_points_all)} ready")

        # Removing the first truck position since this is extra when starting
        # x_truck.pop(0)
        # y_truck.pop(0)
        save_to_pickle(f'x_truck', x_truck)
        save_to_pickle(f'y_truck', y_truck)
        save_to_pickle(f'x_route', x_route)
        save_to_pickle(f'y_route', y_route)

    # # Hack to remove
    # temp_x_route = deepcopy(x_route)
    # temp_y_route = deepcopy(y_route)

    x_min = min(min(min(x_route), min(x_truck)))
    x_max = max(max(max(x_route), max(x_truck)))

    y_min = min(min(min(y_route), min(y_truck)))
    y_max = max(max(max(y_route), max(y_truck)))

    print(f'Number of episodes {len(x_route)}')
    all_episodes_difficult_sum = []
    all_line_difficult_rewards = []
    all_point_difficult_rewards = []

    all_episodes_easy_sum = []
    all_line_easy_rewards = []
    all_point_easy_rewards = []

    easy_x_indices = []
    difficiult_x_indices = []

    done_data_diff = -1
    current_difficulty = ""
    for idx in range(len(x_route)):

        if idx > 5 and idx + 10 < len(x_route):
            entry = int(df['EntryExit'].loc[idx].split(',')[0])
            exit = int(df['EntryExit'].loc[idx].split(',')[1])

            for easy in spawn_points_2_lane_roundabout_small_easy:
                if easy[0] == entry and easy[1][0] == exit:
                    current_difficulty = "easy"
                    break
            for difficult in spawn_points_2_lane_roundabout_small_difficult:
                if difficult[0] == entry and difficult[1][0] == exit:
                    current_difficulty = "difficult"
                    break

            if current_difficulty == "easy":
                all_line_easy_rewards.append(sum(df.loc[idx, "line_reward"]))
                all_point_easy_rewards.append(sum(df.loc[idx, "point_reward"]))
                all_episodes_easy_sum.append(
                    sum(df.loc[idx, "line_reward"]) + sum(df.loc[idx, "point_reward"]))
                easy_x_indices.append(idx)
            elif current_difficulty == "difficult":
                all_line_difficult_rewards.append(sum(df.loc[idx, "line_reward"]))
                all_point_difficult_rewards.append(sum(df.loc[idx, "point_reward"]))
                all_episodes_difficult_sum.append(
                    sum(df.loc[idx, "line_reward"]) + sum(df.loc[idx, "point_reward"]))
                difficiult_x_indices.append(idx)
            else:
                raise Exception('wtf')
    plt.figure(figsize=(80, 5))
    plt.ylim(-20000)
    plt.plot(difficiult_x_indices, all_episodes_difficult_sum, label='All DIFFICULT episode rewards')
    plt.plot(easy_x_indices, all_episodes_easy_sum, label='All EASY episode rewards')

    window = 10
    average_difficult_y = []
    average_easy_y = []
    for ind in range(len(all_episodes_difficult_sum) - window + 1):
        average_difficult_y.append(np.mean(all_episodes_difficult_sum[ind:ind + window]))
    for ind in range(len(all_episodes_easy_sum) - window + 1):
        average_easy_y.append(np.mean(all_episodes_easy_sum[ind:ind + window]))
    # plt.plot(average_difficult_y,label='average_DIFFICULT _y')
    # plt.plot(average_easy_y,label='average_easy_y ')

    # plt.plot(all_point_difficult_rewards,label='all_point_difficult_rewards')
    # plt.plot(all_line_difficult_rewards,label='all_line_difficult_rewards')
    plt.legend(loc='upper center')
    plt.show()



    for idx in range(len(x_route)):
        after_this_time = "2023_06_04__18_20_0"



        # if len(x_truck[idx]) > 0:

        if idx > 0:
            # # Hack to remove
            # if idx != 0:
            #     x_route[idx] = temp_x_route[idx][len(temp_x_route[idx-1]):]
            #     y_route[idx] = temp_y_route[idx][len(temp_y_route[idx-1]):]

            if len(x_route[idx]) != 0:
                print('----------')
                print(f'Showing Episode {idx}/{len(x_route)}')
                # print(df_done.iloc[[idx]])
                lidar = False
                if lidar:
                    for lidar_point in df.loc[idx + 1, "lidar_data"]:

                        print(f"truck FRONT \t\t\t{round(lidar_point.front, 2)}")
                        print(f"truck 45 \t\t{round(lidar_point.front_left, 2)}\t\t{round(lidar_point.front_right, 2)}")
                        print(f"truck sides \t\t{round(lidar_point.left, 2)}\t\t{round(lidar_point.right, 2)}")
                        print(f"")
                        print(
                            f"trailer_0 \t\t{round(lidar_point.trailer_0_left, 2)}\t\t{round(lidar_point.trailer_0_right, 2)}")
                        print(
                            f"trailer_1 \t\t{round(lidar_point.trailer_1_left, 2)}\t\t{round(lidar_point.trailer_1_right, 2)}")
                        print(
                            f"trailer_2 \t\t{round(lidar_point.trailer_2_left, 2)}\t\t{round(lidar_point.trailer_2_right, 2)}")
                        print(
                            f"trailer_3 \t\t{round(lidar_point.trailer_3_left, 2)}\t\t{round(lidar_point.trailer_3_right, 2)}")
                        print(
                            f"trailer_4 \t\t{round(lidar_point.trailer_4_left, 2)}\t\t{round(lidar_point.trailer_4_right, 2)}")
                        print(
                            f"trailer_5 \t\t{round(lidar_point.trailer_5_left, 2)}\t\t{round(lidar_point.trailer_5_right, 2)}")
                        print(f"------------------------------------------------------")

                        if df.loc[idx, "Collisions"][1] != 0:
                            if lidar_point.time != df.loc[idx, "Collisions"][1]:
                                print('------------------------')
                                print(f"Lidar time \t\t\t{lidar_point.time}")
                                print(f'Collision time \t\t{df.loc[idx, "Collisions"][1]}')
                                print('------------------------')

                print()
                print()
                print('----------')

                buffer = 20

                fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
                a1 = axes[0]
                a2 = axes[1]
                # a3 = axes[2]
                # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
                a1.plot(x_route[idx][0], y_route[idx][0], 'bo', label='Route Starting waypoint')
                a1.plot(x_truck[idx][0], y_truck[idx][0], 'kd', label='Truck Starting waypoint')
                a1.plot(x_route[idx][2:], y_route[idx][2:], 'y^')
                a1.plot(x_truck[idx][2:], y_truck[idx][2:], "ro")
                a1.plot(df.loc[idx, "Collisions"].x,
                        df.loc[idx, "Collisions"].y, 'b*')

                a1.axis([x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer])
                # plt.axis([0, 1, 0, 1])
                a1.set_title(
                    f'Collision with {df.loc[idx, "Vehicle"]}. Episode {idx}/{len(x_route)}')
                a1.invert_yaxis()
                a1.legend(loc='upper center')

                # if df.loc[idx, "Vehicle"] != "None":
                #     assert df.loc[idx, "Collisions"][0] in df.loc[idx,'Done']
                # else:
                #     assert "arrived" in df.loc[idx,'Done']

                assert len(df.loc[idx, "point_reward"]) == len(x_truck[idx][2:])
                assert len(df.loc[idx, "line_reward"]) == len(x_truck[idx][2:])
                assert len(df.loc[idx, "total_episode_reward"]) == len(x_truck[idx][2:])

                a2.plot(np.array(df.loc[idx, "point_reward"]), label='Waypoint reward')
                a2.plot(np.array(df.loc[idx, "line_reward"]), label='Line reward')
                # a2.plot(np.array(df.loc[idx, "total_episode_reward"]),
                #         label='Total Episode Reward')

                combined_rewards = []
                for line_reward, point_reward in zip(df.loc[idx, "line_reward"],
                                                     df.loc[idx, "point_reward"]):
                    combined_rewards.append(line_reward + point_reward)
                # a2.plot(combined_rewards, label='Combined reward')

                print(f'line Reward total {sum(df.loc[idx, "line_reward"])}')
                print(f'Point Reward total {sum(df.loc[idx, "point_reward"])}')
                print(f'Sums Reward total {sum(df.loc[idx, "point_reward"]) + sum(df.loc[idx, "line_reward"])}')
                print(f'Total Episode Reward total {sum(df.loc[idx, "total_episode_reward"])}')
                print(f'Total Episode Reward WITHOUT LAST total {sum(df.loc[idx, "total_episode_reward"][:-1])}')

                last_waypoint = 0
                temp_arry = []
                for r,reward in enumerate(df.loc[idx, "total_episode_reward"]):
                    if reward > 40:
                        temp_arry.append(sum(df.loc[idx, "total_episode_reward"][last_waypoint:r]))
                        print(sum(df.loc[idx, "total_episode_reward"][last_waypoint:r]))
                        print(f'Average {mean(df.loc[idx, "total_episode_reward"][last_waypoint:r])}')

                        last_waypoint = r +1
                print(f'Average overall {mean(temp_arry[2:])}')
                print(df.loc[idx, "Time"])

                point_reward = list(df.loc[idx, "point_reward"])
                line_reward = list(df.loc[idx, "line_reward"])

                def divide(list, dividend):
                    new_list = []
                    for item in list:
                        new_list.append(item / dividend)
                    return new_list


                # print(f'New sum {sum(divide(point_reward,50)) + sum(divide(line_reward,100))}')





                assert len(
                    df.loc[idx, "closest_distance_to_next_waypoint_line"]) - 2 == len(
                    x_truck[idx][2:])
                assert len(df.loc[
                               idx, "closest_distance_to_next_plus_1_waypoint_line"]) - 2 == len(
                    x_truck[idx][2:])

                a2.plot(df.loc[idx, "closest_distance_to_next_waypoint_line"],
                        label='Distance to line')
                a2.plot(df.loc[idx, "closest_distance_to_next_plus_1_waypoint_line"],
                        label='+1 Distance to line')

                temp = []

                for i, item in enumerate(
                        df.loc[idx, "closest_distance_to_next_waypoint_line"]):
                    if i != 0:
                        line_reward_multiply_factor = 100
                        hyp_reward = \
                            df.loc[idx, "closest_distance_to_next_waypoint_line"][
                                i - 1] - item

                        hyp_reward = np.clip(hyp_reward, None, 0.5)
                        hyp_reward = hyp_reward - 0.5
                        temp.append(hyp_reward * line_reward_multiply_factor)

                # a2.plot(temp, label='Custom previous-current')
                a2.axis([0, 1000, -100, 10])
                a2.legend(loc='upper center')
                a2.set_title(df.loc[idx, "Time"])

                assert len(
                    df.loc[idx, "closest_distance_to_next_waypoint_line"]) - 2 == len(
                    x_truck[idx][2:])
                assert len(df.loc[
                               idx, "closest_distance_to_next_plus_1_waypoint_line"]) - 2 == len(
                    x_truck[idx][2:])

                # a3.plot(df.loc[idx, "closest_distance_to_next_waypoint_line"], label='Distance to line')
                # a3.plot(df.loc[idx, "closest_distance_to_next_plus_1_waypoint_line"], label='+1 Distance to line')
                #
                # a3.axis([0, 500, -1, 25])
                # a3.legend(loc='upper center')
                items_to_plot = []
                items_not_to_plot = ['Collisions', 'Done', 'lidar_data', 'line_reward_location',
                                     'path', 'index',
                                     'point_reward_location', 'radii', 'route','EntryExit', 'Time','Vehicle']
                for col in df.columns:
                    if col not in items_not_to_plot:
                        items_to_plot.append(col)

                items_to_plot = sorted(items_to_plot)
                num_of_cols = 4
                num_of_rows = ceil(len(items_to_plot) / num_of_cols)
                fig, axes = plt.subplots(ncols=num_of_cols, nrows=num_of_rows, figsize=(15, 10))

                for ax, col_name in zip(axes.ravel(), items_to_plot):
                    assert abs(len(df.loc[idx, col_name]) - len(x_truck[idx][2:])) < 3
                    ax.plot(df.loc[idx, col_name])
                    ax.set_title(str(col_name), fontsize=10)

                plt.show()


if for_graphs:
    for filename in os.listdir(directory):
        if "forward_velocity_z" not in filename:
            if "done" in filename:

                x = sns.catplot(df[['EntryExit', 'Done']], x="output", y="points")
                print("------------------------------")
                print("DONE DATA")
                print(df[['EntryExit', 'Done']].output.value_counts())
                print(df[['EntryExit', 'Done']].points.value_counts())
                print("------------------------------")

                # x = sns.swarmplot(df_done,x="output",y="points")
                # sns.scatterplot(data)
                plt.show()
                x.savefig(os.path.join(directory, filename + string + '.png'))
            elif "route" in filename or "path" in filename or "lidar_data" in filename or "Collisions" in filename:
                continue
            else:
                # for col in df.columns:
                #     print(df[col].describe(include='all'))
                #     print('--------------')
                file = os.path.join(directory, filename)
                # checking if it is a file
                # print(file)
                if os.path.isfile(file) and file.endswith('.pkl'):
                    # plt.xlabel(filename)
                    # sns.distplot(data)
                    if "hyp_distance" in filename or "closest_distance" in filename:
                        print(filename)
                        print(df[filename + string].describe(include='all'))
                        # sns.distplot(df[filename+ string], bins=10, kde=False)
                        values_greater_than = [len(np.where(df[filename + string] > 10)[0]),
                                               len(np.where(df[filename + string] > 20)[0]),
                                               len(np.where(df[filename + string] > 30)[0]),
                                               len(np.where(df[filename + string] > 40)[0]),
                                               len(np.where(df[filename + string] > 50)[0])]

                        temp = [10, 20, 30, 40, 50]
                        for value in range(5):
                            print(f"Values greater than {temp[value]}")
                            print(values_greater_than[value])
                            print(values_greater_than[value] / len(df[filename + string]) * 100)
                            print('---------------')

                        difference = []
                        for i, item in enumerate(df.loc[:, filename + string]):
                            if i == 0:
                                continue

                            difference.append(df.loc[i - 1, filename + string] - item)

                        series_difference = pd.Series(difference)
                        print('----------')
                        values_greater_than = []
                        temp = [0.5]
                        for value in temp:
                            values_greater_than.append(len(np.where(series_difference > 0.8)[0]))

                        for value in range(len(temp)):
                            print(f"Values greater than {temp[value]}")
                            print(values_greater_than[value])
                            print(values_greater_than[value] / len(series_difference) * 100)
                            print('---------------')

                        print(series_difference.describe())
                        a = sns.distplot(series_difference,bins=10000)
                        a.set(xlim=(-0.5,1))

                        plt.show()
                        print('----------')

                        plt.show()
                    else:
                        print(filename)
                        print(df[filename + string].describe(include='all'))
                        x = sns.displot(df[filename + string])
                        # sns.pointplot(x=df['forward_velocity'],y=df['forward_velocity_x'])
                        # sns.scatterplot(data)
                        plt.show()
                        x.savefig(os.path.join(directory, filename + string + '.png'))

plot_route(df["route"], df["path"])

# y_dist_to_waypoints 0-> 0.023
# x_dist_to_waypoints
# acceleration
# forward_velocity
# forward_velocity_x
# forward_velocity_z
# angle_with_center
