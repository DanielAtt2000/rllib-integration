from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import pandas as pd
import numpy as np

from DataAnalysis.ProcessData import data_file
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
df_done = pd.DataFrame()
string = ''
largest_filename = "angle_to_center_of_lane_degrees.pkl"
with open(os.path.join(directory, largest_filename), 'rb') as handle:
    data = pickle.load(handle)
    df[str(largest_filename)] = pd.Series(data)
    # df = df.fillna(0)

for filename in os.listdir(directory):
    if largest_filename in filename:
        continue
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file) and file.endswith('.pkl'):
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            if "done" not in filename:
                df[str(filename + string)] = pd.Series(data)
                # df = df.fillna(N)
            else:
                df_done['points'] = pd.Series(data[0][0])
                df_done['output'] = pd.Series(data[0][1])

graphs = False


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
            if "[]\n" in route:
                continue
            temp_x_route = []
            temp_y_route = []
            for route_point in route.split('),'):
                route_point = route_point.replace('[', '')
                route_point = route_point.replace(']', '')
                route_point = route_point.replace('(', '')
                route_point = route_point.replace(')', '')
                route_point = route_point.replace(' ', '')
                route_point = route_point.replace('\n', '')

                points = route_point.split(',')

                temp_x_route.append(float(points[0]))
                temp_y_route.append(float(points[1]))

            x_route.append(temp_x_route)
            y_route.append(temp_y_route)
            print(f"Route {route_id}/{len(route_points_all)} ready")

        truck_points_all = truck_points_all[truck_points_all != 0]
        for truck_idx, truck_points in enumerate(truck_points_all):
            if "[]\n" in truck_points:
                continue
            temp_x_truck = []
            temp_y_truck = []
            for truck_point in truck_points.split('),'):
                truck_point = truck_point.replace('[', '')
                truck_point = truck_point.replace(']', '')
                truck_point = truck_point.replace('(', '')
                truck_point = truck_point.replace(')', '')
                truck_point = truck_point.replace('\n', '')
                truck_point = truck_point.replace(' ', '')

                truck_point = truck_point.split(',')

                temp_x_truck.append(float(truck_point[0]))
                temp_y_truck.append(float(truck_point[1]))

            x_truck.append(temp_x_truck)
            y_truck.append(temp_y_truck)
            print(f"Truck Path {truck_idx}/{len(truck_points_all)} ready")

        # Removing the first truck position since this is extra when starting
        x_truck.pop(0)
        y_truck.pop(0)
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
            entry = int(df_done.loc[idx + done_data_diff][0].split(',')[0])
            exit = int(df_done.loc[idx + done_data_diff][0].split(',')[1])

            for easy in spawn_points_2_lane_roundabout_small_easy:
                if easy[0] == entry and easy[1][0] == exit:
                    current_difficulty = "easy"
            for difficult in spawn_points_2_lane_roundabout_small_difficult:
                if difficult[0] == entry and difficult[1][0] == exit:
                    current_difficulty = "difficult"

            if current_difficulty == "easy":
                all_line_easy_rewards.append(sum(df.loc[idx + 2, "line_reward.pkl"]))
                all_point_easy_rewards.append(sum(df.loc[idx + 2, "point_reward.pkl"]))
                all_episodes_easy_sum.append(
                    sum(df.loc[idx + 2, "line_reward.pkl"]) + sum(df.loc[idx + 2, "point_reward.pkl"]))
                easy_x_indices.append(idx + 2)
            elif current_difficulty == "difficult":
                all_line_difficult_rewards.append(sum(df.loc[idx + 2, "line_reward.pkl"]))
                all_point_difficult_rewards.append(sum(df.loc[idx + 2, "point_reward.pkl"]))
                all_episodes_difficult_sum.append(
                    sum(df.loc[idx + 2, "line_reward.pkl"]) + sum(df.loc[idx + 2, "point_reward.pkl"]))
                difficiult_x_indices.append(idx + 2)
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

    temp_data_diff = 2
    counter = 0
    for idx in range(len(x_route)):
        try:
            assert len(df.loc[idx + temp_data_diff, "point_reward.pkl"]) == len(x_truck[idx][2:])
        except:
            temp_data_diff +=1
            # print(len(df.loc[idx + 2, "point_reward.pkl"]))
            # print(len(x_truck[idx][2:]))
            counter +=1
            print(idx)
    print(counter)
    # input('Have you check the data diff?')
    for idx in range(len(x_route)):

        # if len(x_truck[idx]) > 0:

        if idx > 5700:

            for i in range(100):
                if abs(len(df.loc[idx + i, 'bearing_to_waypoint.pkl']) - len(x_truck[idx][2:])) < 3:
                    data_diff = i
                    break

            # # Hack to remove
            # if idx != 0:
            #     x_route[idx] = temp_x_route[idx][len(temp_x_route[idx-1]):]
            #     y_route[idx] = temp_y_route[idx][len(temp_y_route[idx-1]):]

            if len(x_route[idx]) != 0:
                print('----------')
                print(f'Showing Episode {idx}/{len(x_route)}')
                print(df_done.iloc[[idx - 1]])
                lidar = False
                if lidar:
                    for lidar_point in df.loc[idx + 1, "lidar_data.pkl"]:

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

                        if df.loc[idx + data_diff, "collisions.pkl"][1] != 0:
                            if lidar_point.time != df.loc[idx + data_diff, "collisions.pkl"][1]:
                                print('------------------------')
                                print(f"Lidar time \t\t\t{lidar_point.time}")
                                print(f'Collision time \t\t{df.loc[idx + data_diff, "collisions.pkl"][1]}')
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
                a1.plot(x_route[idx + 1][0], y_route[idx + 1][0], 'bo', label='Route Starting waypoint')
                a1.plot(x_truck[idx][0], y_truck[idx][0], 'kd', label='Truck Starting waypoint')
                a1.plot(x_route[idx + 1][2:], y_route[idx + 1][2:], 'y^')
                a1.plot(x_truck[idx][2:], y_truck[idx][2:], "ro")
                a1.plot(df.loc[idx + data_diff, "collisions.pkl"][2].x,
                        df.loc[idx + data_diff, "collisions.pkl"][2].y, 'b*')

                a1.axis([x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer])
                # plt.axis([0, 1, 0, 1])
                a1.set_title(
                    f'Collision with {df.loc[idx + data_diff, "collisions.pkl"][0]}. Episode {idx}/{len(x_route)} Route {df_done.loc[idx + done_data_diff]}')
                a1.invert_yaxis()
                a1.legend(loc='upper center')

                if df.loc[idx + data_diff, "collisions.pkl"][0] != "None":
                    assert df.loc[idx + data_diff, "collisions.pkl"][0] in df_done.loc[idx + done_data_diff][
                        1]
                else:
                    assert "arrived" in df_done.loc[idx + done_data_diff][1]

                assert len(df.loc[idx + data_diff, "point_reward.pkl"]) == len(x_truck[idx][2:])
                assert len(df.loc[idx + data_diff, "line_reward.pkl"]) == len(x_truck[idx][2:])
                assert len(df.loc[idx + data_diff, "total_episode_reward.pkl"]) == len(x_truck[idx][2:])

                a2.plot(np.array(df.loc[idx + data_diff, "point_reward.pkl"]), label='Waypoint reward')
                a2.plot(np.array(df.loc[idx + data_diff, "line_reward.pkl"]), label='Line reward')
                # a2.plot(np.array(df.loc[idx + data_diff, "total_episode_reward.pkl"]),
                #         label='Total Episode Reward')

                combined_rewards = []
                for line_reward, point_reward in zip(df.loc[idx + data_diff, "line_reward.pkl"],
                                                     df.loc[idx + data_diff, "point_reward.pkl"]):
                    combined_rewards.append(line_reward + point_reward)
                # a2.plot(combined_rewards, label='Combined reward')

                assert len(
                    df.loc[idx + data_diff, "closest_distance_to_next_waypoint_line.pkl"]) - 2 == len(
                    x_truck[idx][2:])
                assert len(df.loc[
                               idx + data_diff, "closest_distance_to_next_plus_1_waypoint_line.pkl"]) - 2 == len(
                    x_truck[idx][2:])

                a2.plot(df.loc[idx + data_diff, "closest_distance_to_next_waypoint_line.pkl"],
                        label='Distance to line')
                a2.plot(df.loc[idx + data_diff, "closest_distance_to_next_plus_1_waypoint_line.pkl"],
                        label='+1 Distance to line')

                temp = []

                for i, item in enumerate(
                        df.loc[idx + data_diff, "closest_distance_to_next_waypoint_line.pkl"]):
                    if i != 0:
                        line_reward_multiply_factor = 100
                        hyp_reward = \
                            df.loc[idx + data_diff, "closest_distance_to_next_waypoint_line.pkl"][
                                i - 1] - item

                        hyp_reward = np.clip(hyp_reward, None, 0.5)
                        hyp_reward = hyp_reward - 0.5
                        temp.append(hyp_reward * line_reward_multiply_factor)

                # a2.plot(temp, label='Custom previous-current')
                a2.axis([0, 1000, -55, 70])
                a2.legend(loc='upper center')

                assert len(
                    df.loc[idx + data_diff, "closest_distance_to_next_waypoint_line.pkl"]) - 2 == len(
                    x_truck[idx][2:])
                assert len(df.loc[
                               idx + data_diff, "closest_distance_to_next_plus_1_waypoint_line.pkl"]) - 2 == len(
                    x_truck[idx][2:])

                # a3.plot(df.loc[idx+data_diff, "closest_distance_to_next_waypoint_line.pkl"], label='Distance to line')
                # a3.plot(df.loc[idx+data_diff, "closest_distance_to_next_plus_1_waypoint_line.pkl"], label='+1 Distance to line')
                #
                # a3.axis([0, 500, -1, 25])
                # a3.legend(loc='upper center')
                items_to_plot = []
                items_not_to_plot = ['collisions.pkl', 'done.pkl', 'lidar_data.pkl', 'line_reward_location.pkl',
                                     'path.pkl',
                                     'point_reward_location.pkl', 'radii.pkl', 'route.pkl']
                for col in df.columns:
                    if col not in items_not_to_plot:
                        items_to_plot.append(col)

                items_to_plot = sorted(items_to_plot)
                num_of_cols = 4
                num_of_rows = len(items_to_plot) // num_of_cols
                fig, axes = plt.subplots(ncols=num_of_cols, nrows=num_of_rows, figsize=(15, 10))

                for ax, col_name in zip(axes.ravel(), items_to_plot):
                    assert abs(len(df.loc[idx + data_diff, col_name]) - len(x_truck[idx][2:])) < 3
                    ax.plot(df.loc[idx + data_diff, col_name])
                    ax.set_title(str(col_name), fontsize=10)

                plt.show()


if graphs:
    for filename in os.listdir(directory):
        if "forward_velocity_z" not in filename:
            if "done" in filename:

                x = sns.catplot(df_done, x="output", y="points")
                print("------------------------------")
                print("DONE DATA")
                print(df_done.output.value_counts())
                print(df_done.points.value_counts())
                print("------------------------------")

                # x = sns.swarmplot(df_done,x="output",y="points")
                # sns.scatterplot(data)
                plt.show()
                x.savefig(os.path.join(directory, filename + string + '.png'))
            elif "route" in filename or "path" in filename or "lidar_data" in filename or "collisions" in filename:
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
                            values_greater_than.append(len(np.where(series_difference > 0.4)[0]))

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
                        # sns.pointplot(x=df['forward_velocity.pkl'],y=df['forward_velocity_x.pkl'])
                        # sns.scatterplot(data)
                        plt.show()
                        x.savefig(os.path.join(directory, filename + string + '.png'))

plot_route(df["route.pkl" + string], df["path.pkl" + string])

# y_dist_to_waypoints 0-> 0.023
# x_dist_to_waypoints
# acceleration
# forward_velocity
# forward_velocity_x
# forward_velocity_z
# angle_with_center
