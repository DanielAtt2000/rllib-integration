import math
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

from DataAnalysis.NewProcessData2WithTrailer import data_file, for_graphs
from newHelper import get_route_type


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Lidar:
    def __init__(self, time, front, right15, right30, right45, right60, right75, left15, left30,left45,left60,left75, left, right, trailer_0_left, trailer_1_left, trailer_2_left,
                 trailer_3_left, trailer_4_left, trailer_5_left, trailer_0_right, trailer_1_right, trailer_2_right,
                 trailer_3_right, trailer_4_right, trailer_5_right):
        self.time = time
        self.front = front
        self.right15 = right15
        self.right30 = right30
        self.right45 = right45
        self.right60 = right60
        self.right75 = right75
        self.left15 = left15
        self.left30 = left30
        self.left45 = left45
        self.left60 = left60
        self.left75 = left75

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
largest_filename = "done.pkl"
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
            if len(data) == 0:
                print(f'{file} has no data')
                # a = random.randint(0,100)
                # x = input(f'Enter {a}:')
                continue

            df = pd.merge(df,data, on='Time')
#             df = df.drop_duplicates(subset=['Time'])
#
# df = df.reset_index()

def plot_route(route_points_all, truck_points_all,trailer_points_all=[]):
    x_route = []
    y_route = []

    x_truck = []
    y_truck = []

    x_trailer = []
    y_trailer = []

    try:
        x_truck = open_pickle(os.path.join(directory, 'x_truck.pickle'))
        y_truck = open_pickle(os.path.join(directory, 'y_truck.pickle'))
        x_route = open_pickle(os.path.join(directory, 'x_route.pickle'))
        y_route = open_pickle(os.path.join(directory, 'y_route.pickle'))
        x_trailer = open_pickle(os.path.join(directory, 'x_trailer.pickle'))
        y_trailer = open_pickle(os.path.join(directory, 'y_trailer.pickle'))

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
            temp_x_trailer = []
            temp_y_trailer = []
            for t,truck_point in enumerate(truck_points):
                temp_x_truck.append(truck_point[0])
                temp_y_truck.append(truck_point[1])

                if len(trailer_points_all) != 0:
                    temp_x_trailer.append(trailer_points_all[truck_idx][t][0])
                    temp_y_trailer.append(trailer_points_all[truck_idx][t][1])

            x_truck.append(temp_x_truck)
            y_truck.append(temp_y_truck)
            x_trailer.append(temp_x_trailer)
            y_trailer.append(temp_y_trailer)
            print(f"Truck Path {truck_idx}/{len(truck_points_all)} ready")

        # Removing the first truck position since this is extra when starting
        # x_truck.pop(0)
        # y_truck.pop(0)
        save_to_pickle(f'x_truck', x_truck)
        save_to_pickle(f'y_truck', y_truck)
        save_to_pickle(f'x_route', x_route)
        save_to_pickle(f'y_route', y_route)
        save_to_pickle(f'x_trailer', x_trailer)
        save_to_pickle(f'y_trailer', y_trailer)

    # # Hack to remove
    # temp_x_route = deepcopy(x_route)
    # temp_y_route = deepcopy(y_route)

    x_min = min(min(min(x_route), min(x_truck)))
    x_max = max(max(max(x_route), max(x_truck)))

    y_min = min(min(min(y_route), min(y_truck)))
    y_max = max(max(max(y_route), max(y_truck)))

    x_min_upper =[]
    x_max_upper =[]

    y_min_upper=[]
    y_max_upper= []

    x_min_lower = []
    x_max_lower = []

    y_min_lower = []
    y_max_lower= []

    for idx in range(len(y_route)):
        if min(y_route[idx]) > 0:
            x_min_upper.append(min(x_route[idx]))
            x_max_upper.append(max(x_route[idx]))

            y_min_upper.append(min(y_route[idx]))
            y_max_upper.append(max(y_route[idx]))
        else:
            x_min_lower.append(min(x_route[idx]))
            x_max_lower.append(max(x_route[idx]))

            y_min_lower.append(min(y_route[idx]))
            y_max_lower.append(max(y_route[idx]))

    if len(x_min_upper) != 0:
        x_min_upper = min(x_min_upper)
        y_min_upper = min(y_min_upper)
        x_max_upper = max(x_max_upper)
        y_max_upper = max(y_max_upper)

    if len(x_min_lower) != 0:
        x_min_lower = min(x_min_lower)
        y_min_lower = min(y_min_lower)
        x_max_lower = max(x_max_lower)
        y_max_lower = max(y_max_lower)






    print(f'Number of episodes {len(x_route)}')
    all_episodes_difficult_sum = []
    all_line_difficult_rewards = []
    all_point_difficult_rewards = []

    all_episodes_easy_sum = []
    all_line_easy_rewards = []
    all_point_easy_rewards = []

    easy_x_indices = []
    difficiult_x_indices = []

    easy_dict ={}
    diff_dict = {}

    x = input('skip initial visual? (y/n)')
    if x == 'n':
        rewards = []
        done_array = []
        for idx in range(len(x_route)):
            if idx < 10000000:
                if (df.loc[idx, "Done"]) == 'done_arrived':
                    done_array.append(1)
                else:
                    done_array.append(0)
                rewards.append(sum(df.loc[idx, "total_episode_reward"]))
        done_array_cum_sum = np.cumsum(done_array)
        stat = "count"  # or proportion
        # sns.histplot(done_array, stat=stat, cumulative=False, alpha=.4)
        # sns.ecdfplot(done_array, stat=stat)
        sns.lineplot(done_array_cum_sum)
        plt.show()

        x = sns.histplot(df['EntryExit'], binwidth=10)
        print(df.groupby(df['EntryExit'].tolist(), as_index=False).size())

        print("------------------------------")
        print("DONE DATA")
        # print(df[['EntryExit', 'Done']].output.value_counts())
        # print(df[['EntryExit', 'Done']].points.value_counts())
        print("------------------------------")

        # x = sns.swarmplot(df_done,x="output",y="points")
        # sns.scatterplot(data)
        plt.show()



    # done_data_diff = -1
    # current_difficulty = ""
    # for idx in range(len(x_route)):
    #
    #     if idx > 5 and idx + 10 < len(x_route):
    #         entry = int(df['EntryExit'].loc[idx].split(',')[0])
    #         exit = int(df['EntryExit'].loc[idx].split(',')[1])
    #
    #         for easy in spawn_points_2_lane_roundabout_small_easy:
    #             if easy[0] == entry and easy[1][0] == exit:
    #                 current_difficulty = "easy"
    #                 break
    #         for difficult in spawn_points_2_lane_roundabout_small_difficult:
    #             if difficult[0] == entry and difficult[1][0] == exit:
    #                 current_difficulty = "difficult"
    #                 break
    #
    #         if current_difficulty == "easy":
    #             # all_line_easy_rewards.append(sum(df.loc[idx, "line_reward"]))
    #             # all_point_easy_rewards.append(sum(df.loc[idx, "point_reward"]))
    #             all_episodes_easy_sum.append(
    #                 sum(df.loc[idx, "total_episode_reward"]))
    #
    #             value_type = 'dsad'
    #             arrived = 'noen'
    #             if mean(df.loc[idx, "angle_to_center_of_lane_degrees_ahead_waypoints"
    #                                 ""]) > 0:
    #                 value_type = 'positive'
    #             else:
    #                 value_type = 'negative'
    #
    #             if (df.loc[idx, "Done"]) == 'done_arrived':
    #                 arrived = 'yes'
    #             else:
    #                 arrived = 'no'
    #
    #             key = f'{value_type}|{arrived}'
    #             if easy_dict.get(key) == None:
    #                 easy_dict[key] = 1
    #             else:
    #                 easy_dict[key] += 1
    #
    #             easy_x_indices.append(idx)
    #         elif current_difficulty == "difficult":
    #             # all_line_difficult_rewards.append(sum(df.loc[idx, "line_reward"]))
    #             # all_point_difficult_rewards.append(sum(df.loc[idx, "point_reward"]))
    #             all_episodes_difficult_sum.append(sum(df.loc[idx, "total_episode_reward"]))
    #
    #             value_type = 'dsad'
    #             arrived = 'noen'
    #             if mean(df.loc[idx, "angle_to_center_of_lane_degrees_ahead_waypoints"]) > 0:
    #                 value_type = 'positive'
    #             else:
    #                 value_type = 'negative'
    #
    #             if (df.loc[idx, "Done"]) == 'done_arrived':
    #                 arrived = 'yes'
    #             else:
    #                 arrived = 'no'
    #
    #             key = f'{value_type}|{arrived}'
    #
    #             if diff_dict.get(key) == None:
    #                 diff_dict[key] = 1
    #             else:
    #                 diff_dict[key] += 1
    #
    #             difficiult_x_indices.append(idx)
    #         else:
    #             raise Exception('wtf')
    # plt.figure(figsize=(80, 5))
    # plt.ylim()
    # plt.plot( done_array, label='All DIFFICULT episode rewards')
    # plt.plot(easy_x_indices, all_episodes_easy_sum, label='All EASY episode rewards')
        save_reward = True
        filename = data_file.split('_')[1]
        if save_reward:
            save_to_pickle('../rewards/' + filename + '_total_reward',rewards)
            save_to_pickle('../done/' + filename + '_done',done_array)



        window = 250
        average_difficult_y = []
        average_easy_y = []
        for ind in range(len(done_array) - window + 1):
            x = np.mean(rewards[ind:ind + window])
            average_difficult_y.append(x)
        # for ind in range(len(all_episodes_easy_sum) - window + 1):
        #     average_easy_y.append(np.mean(all_episodes_easy_sum[ind:ind + window]))
        plt.plot(average_difficult_y,label='average_DIFFICULT _y')
        # plt.plot(average_easy_y,label='average_easy_y ')

        # plt.plot(all_point_difficult_rewards,label='all_point_difficult_rewards')
        # plt.plot(all_line_difficult_rewards,label='all_line_difficult_rewards')
        plt.ylabel('Total episode reward \n averaged over a 250 episode window')
        plt.xlabel('Number of episodes')
        # plt.xticks(np.arange(0, 8000 + 1, 500))
        # plt.legend(loc='upper center')
        plt.show()

        window = 250
        average_difficult_y = []
        average_easy_y = []
        for ind in range(len(done_array) - window + 1):
            x = np.mean(done_array[ind:ind + window])
            average_difficult_y.append(x)
        # for ind in range(len(all_episodes_easy_sum) - window + 1):
        #     average_easy_y.append(np.mean(all_episodes_easy_sum[ind:ind + window]))
        plt.plot(average_difficult_y,label='average_DIFFICULT _y')
        # plt.plot(average_easy_y,label='average_easy_y ')

        # plt.plot(all_point_difficult_rewards,label='all_point_difficult_rewards')
        # plt.plot(all_line_difficult_rewards,label='all_line_difficult_rewards')
        plt.ylabel('Ratio of successful vs unsuccessful episodes \n averaged over a 250 episode window')
        plt.xlabel('Number of episodes')
        # plt.xticks(np.arange(0, 8000 + 1, 500))
        # plt.legend(loc='upper center')
        plt.show()

        print('EASY DICT')
        print(easy_dict)
        total = 0
        for key, value in easy_dict.items():
            total += value
        for key, value in easy_dict.items():
            print(f'{key} = {value/total}')

        total_positive = 0
        total_negative = 0
        for key, value in easy_dict.items():
            if 'positive' in key:
                total_positive += value
            if 'negative' in key:
                total_negative += value

        for key, value in easy_dict.items():
            if 'positive' in key:
                print(f'positive: {total_positive/(total_negative+total_positive)}')
            if 'negative' in key:
                print(f'negative: {total_negative/(total_negative+total_positive)}')



        print('DIFF DICT')
        print(diff_dict)
        for key, value in diff_dict.items():
            total += value
        for key, value in diff_dict.items():
            print(f'{key} = {value/total}')

        total_positive = 0
        total_negative = 0
        for key, value in diff_dict.items():
            if 'positive' in key:
                total_positive += value
            if 'negative' in key:
                total_negative += value

        for key, value in diff_dict.items():
            if 'positive' in key:
                print(f'positive: {total_positive/(total_negative+total_positive)}')
            if 'negative' in key:
                print(f'negative: {total_negative/(total_negative+total_positive)}')



    for idx in range(len(x_route)):
        after_this_time = "2023_10_15__18_45_42_237838"
        current_index_time = datetime.strptime(df.loc[idx, "Time"],"%Y_%m_%d__%H_%M_%S_%f")
        date_obj = datetime.strptime(after_this_time, "%Y_%m_%d__%H_%M_%S_%f")
        # if len(x_truck[idx]) > 0:
        # if current_index_time > date_obj:
        if idx > 0:

            entry_exit = df.loc[idx, 'EntryExit']
            entry = int(entry_exit.split(',')[0])
            exit = int(entry_exit.split(',')[1])
            roundabout, lane = get_route_type(entry, exit)


            # plt.rcParams.update({'font.size': 13})
            # plt.ylabel(f'Velocity (km/h)')
            # plt.xlabel(f'Timesteps')
            # plt.plot([7.048597, 10.56761, 2.889151, 0.14211477, 0.7640619, 0.74261886, 0.5545511, 0.36898744, 0.22958417, 0.13651977, 0.07854896, 0.044162855, 0.023562279, 0.0124610765, 0.0063865897, 0.003162705, 0.0015765068, 0.0008729219, 0.0004574737, 0.0003370581, 0.00035383835, 0.00033868154, 0.00034476002, 0.00034258523, 0.91197264, 2.0345352, 2.9443657, 3.8451521, 4.677823, 5.4231534, 6.099134, 6.71643, 7.158859, 7.5530663, 7.8954506, 8.309437, 8.615425, 8.748457, 8.951688, 8.829759, 8.721211, 8.60064, 8.432978, 8.154224, 8.443003, 8.990368, 9.419192, 9.669604, 10.104006, 10.487111, 10.769341, 10.9648, 11.247985, 11.495302, 11.56789, 11.49541, 11.411024, 11.250843, 11.070036, 10.961799, 11.000701, 10.845775, 10.522882, 10.2881775, 10.020085, 9.955016, 9.733338, 9.565581, 9.533154, 9.475601, 9.334275, 9.286343, 9.476143, 9.692063, 9.741028, 9.573834, 9.619084, 9.599404, 9.5211, 9.275028, 9.221838, 9.198604, 9.110917, 8.974067, 8.91928, 8.815614, 8.571563, 8.476254, 8.457135, 8.449491, 8.444022, 8.510987, 8.588728, 8.727587, 8.725164, 8.6886, 8.909105, 9.1230755, 9.328306, 9.258016, 9.066631, 8.745373, 8.465128, 8.615762, 8.582039, 8.737809, 8.779495, 8.688319, 8.5960245, 8.702222, 8.667752, 8.518334, 8.466722, 8.632658, 8.711213, 8.763383, 8.615609, 8.379261, 8.341181, 8.311021, 8.229082, 7.9698696, 7.747595, 8.0107155, 8.331076, 8.5376625, 8.535216, 8.460433, 8.606991, 8.748678, 8.879641, 8.768273, 8.58356, 8.2910795, 8.089192, 8.230902, 8.353651, 8.341567, 8.333678, 8.326349, 8.351539, 8.317289, 8.382137, 8.490773, 8.456789, 8.479561, 8.595629, 8.601483, 8.55862, 8.380951, 8.423912, 8.301952, 8.476604, 8.557357, 8.547234, 8.373418, 8.177293, 8.167895, 8.118682, 7.9992976, 7.743457, 7.4703913, 7.5058284, 7.9175224, 8.277655, 8.502052, 8.505167, 8.434984, 8.589162, 8.735778, 8.825508, 8.8453045, 8.806322, 8.734568, 8.656042, 8.583993, 8.357876, 8.193671, 8.043627, 7.995025, 7.6269617, 7.5453744, 7.417404, 7.5760894, 7.825951, 8.309033, 8.554157, 8.876444, 8.848886, 8.835186, 8.726134, 8.565836, 8.311316, 8.213033, 8.366435, 8.555072, 8.776782, 8.756332, 8.546193, 8.287936, 8.132274, 7.9822145, 7.7471485, 7.514975, 7.6145716, 8.021211, 8.326136, 8.605952, 8.65581, 8.674132, 8.57877, 8.572946, 8.56056, 8.438242, 8.255873, 8.346629, 8.555389, 8.611102, 8.550036, 8.446084, 8.460878, 8.406544, 8.250754, 8.022155, 8.253003, 8.467415, 8.651664, 8.791179, 8.650636, 8.653001, 8.573582, 8.426924, 8.199453, 8.257879, 8.357671, 8.475611, 8.45023, 8.284695, 8.368952, 8.442022, 8.252644, 8.245638, 8.348218, 8.533696, 8.499529, 8.497407, 8.31592, 8.18823, 7.954877, 7.6600623, 7.602822, 7.775473, 8.014996, 8.291341, 8.461933, 8.614119, 8.563841, 8.502203, 8.489486, 8.424352, 8.4418, 8.339896, 8.291, 8.33784, 8.220264, 8.231761, 8.087805, 7.897888, 7.6744614, 7.6305127, 7.867796, 8.257362, 8.537281, 8.583826, 8.467774, 8.581996, 8.771011, 8.790608, 8.69273, 8.595645, 8.452981, 8.269219, 8.376096, 8.383047, 8.298221, 8.477817, 8.639, 8.592079, 8.360307, 8.215161, 8.065758, 7.89479, 7.7809057, 7.727299, 7.9375553, 8.129679, 8.204938, 8.428123, 8.572058, 8.626315, 8.487496, 8.380957, 8.4907875, 8.667465, 8.666145, 8.603009, 8.496069, 8.332578, 8.288984, 8.287336, 8.278281, 8.269602, 8.1067915, 8.247673, 8.521312, 8.6124525, 8.734848, 8.602205, 8.513441, 8.328106, 8.288156, 8.3570795, 8.460904, 8.58786, 8.678896, 8.714709, 8.559846, 8.323169, 8.220854, 8.25799, 8.132107, 7.9571404, 7.688031, 7.5383153, 7.6484766, 7.931009, 8.28734, 8.603909, 8.667628, 8.568552, 8.524581, 8.378444, 8.33036, 8.364291, 8.3644905, 8.436335, 8.575673, 8.50038],label='constant.velocity')
            # plt.plot([7.048597, 10.56761, 2.8894029, 0.1303027, 0.7639415, 0.743069, 0.55469537, 0.36922708, 0.2294855, 0.13626085, 0.078133, 0.044292558, 0.023576425, 0.012465032, 0.006394192, 0.0031920609, 0.0015649032, 0.0008070228, 0.000450514, 0.00036939653, 0.00033063776, 0.00032745383, 0.00033180832, 0.0003353188, 0.84743136, 1.9483229, 2.8042457, 3.571376, 4.3380294, 4.9329214, 5.5020847, 6.1003413, 6.6487947, 7.1438813, 7.529135, 7.9668407, 8.283432, 8.648624, 8.883807, 8.930475, 8.919418, 8.798904, 8.5519285, 8.381675, 8.391377, 8.9199295, 9.35985, 9.664747, 9.867468, 9.954627, 10.134285, 10.24786, 10.465135, 10.595837, 10.692003, 10.766941, 10.792122, 10.830938, 10.966867, 11.090008, 11.0534, 10.86543, 10.858971, 10.713006, 10.715623, 10.9895525, 11.15062, 11.26129, 11.348247, 11.357041, 11.436083, 11.580178, 11.679253, 11.759625, 11.830837, 11.822505, 11.910329, 12.241093, 12.615984, 12.638153, 12.433086, 12.157883, 12.101512, 12.0767145, 12.106647, 12.090721, 12.129439, 12.085421, 12.149147, 12.427945, 12.57384, 12.558767, 12.61424, 12.867999, 13.170203, 13.477762, 13.789954, 13.810083, 13.802103, 13.674397, 13.252184, 12.90442, 12.643396, 12.554944, 12.520116, 12.709463, 12.610569, 12.47153, 12.225934, 12.235007, 12.252922, 12.158192, 12.08363, 12.1793785, 12.337244, 12.505993, 12.680082, 12.857066, 13.03533, 13.214096, 13.037779, 12.838786, 12.562438, 12.560182, 12.578229, 12.467501, 12.377204, 12.465873, 12.622903, 12.792706, 12.968102, 13.146268, 13.325445, 13.504872, 13.311613, 13.087707, 12.836174, 12.681453, 12.703695, 12.844787, 13.014569, 12.865221, 12.693099, 12.43116, 12.430617, 12.459357, 12.359195, 12.269402, 12.357672, 12.51606, 12.688196, 12.865959, 13.046232, 13.227243, 13.408256, 13.588977, 13.769264, 13.553959, 13.300401, 13.055337, 12.887354, 12.9020815, 13.039969, 12.861077, 12.691057, 12.41288, 12.241681, 12.008973, 11.931004, 11.661327, 11.720347, 11.759873, 11.677062, 11.654991, 11.498783, 11.616817, 11.584138, 11.609381, 11.647708, 11.52667, 11.537859], label='variable.velocity')
            # plt.legend(loc='lower right')

            roundabout16MDifficultEntryPoints = [31, 117, 34, 5]
            roundabout50MDifficultEntryPoints = [73, 15, 55, 96, 58, 55]
            #
            # if roundabout == 'double' and entry in roundabout16MDifficultEntryPoints and 'done_arrived' in df.loc[idx, "Done"]:
            #     pass
            # else:
            #     continue



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
                    for lidar_point in df.loc[idx, "lidar_data"]:

                        print(f"truck FRONT \t\t\t{round(lidar_point.front, 2)}")
                        print(f"truck 15 \t\t{round(lidar_point.left15, 2)}\t\t{round(lidar_point.right15, 2)}")
                        print(f"truck 30 \t\t{round(lidar_point.left30, 2)}\t\t{round(lidar_point.right30, 2)}")
                        print(f"truck 45 \t\t{round(lidar_point.left45, 2)}\t\t{round(lidar_point.right45, 2)}")
                        print(f"truck 60 \t\t{round(lidar_point.left60, 2)}\t\t{round(lidar_point.right60, 2)}")
                        print(f"truck 75 \t\t{round(lidar_point.left75, 2)}\t\t{round(lidar_point.right75, 2)}")
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
                        print(
                            f"trailer_6 \t\t{round(lidar_point.trailer_6_left, 2)}\t\t{round(lidar_point.trailer_6_right, 2)}")
                        print(
                            f"trailer_7 \t\t{round(lidar_point.trailer_7_left, 2)}\t\t{round(lidar_point.trailer_7_right, 2)}")
                        print(f"------------------------------------------------------")

                        # if df.loc[idx, "Collisions"] != 0:
                        #     if lidar_point.time != df.loc[idx, "Collisions"][1]:
                        #         print('------------------------')
                        #         print(f"Lidar time \t\t\t{lidar_point.time}")
                        #         print(f'Collision time \t\t{df.loc[idx, "Collisions"][1]}')
                        #         print('------------------------')

                print()
                print()
                print('----------')



                fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
                a1 = axes[0]
                a2 = axes[1]

                # x_min = min(x_route[idx])
                # x_max = max(x_route[idx])
                #
                # y_min = min(y_route[idx])
                # y_max = max(y_route[idx])
                buffer = 5
                # a3 = axes[2]
                # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
                a1.plot(x_route[idx][0], y_route[idx][0], 'bo', label='Route Starting waypoint')

                # a1.plot(x_truck[idx][0], y_truck[idx][0], 'kd', label='Truck Starting waypoint')
                a1.plot(x_trailer[idx][2:], y_trailer[idx][2:], "m*")
                a1.plot(x_route[idx][2:], y_route[idx][2:], 'g^')
                a1.plot(x_truck[idx][2:], y_truck[idx][2:], "ro")
                a1.plot(x_route[idx][-20], y_route[idx][-20], 'y^', label='Route End waypoint')
                # a1.plot(df.loc[idx, "Collisions"].x, df.loc[idx, "Collisions"].y, 'b*')

                # roundabout = '32m'
                new_x_min = 0
                new_x_max = 0
                new_y_min = 0
                new_y_max = 0

                if roundabout == '32m':
                    new_x_min = 20
                    new_x_max = 120
                    new_y_min = -50
                    new_y_max = 75
                elif roundabout == '40m':
                    new_x_min = -100
                    new_x_max = -20
                    new_y_min = -180
                    new_y_max = -100
                elif roundabout == '20m':
                    new_x_min = -40
                    new_x_max = 40
                    new_y_min = -70
                    new_y_max = 10
                elif roundabout == '16m':
                    new_x_min = 110
                    new_x_max = 200
                    new_y_min = 70
                    new_y_max = 150
                elif roundabout == '50m':
                    new_x_min = -113
                    new_x_max = 107
                    new_y_min = -393
                    new_y_max = -193
                else:
                    raise Exception()


                # testing = True
                # if testing and roundabout != 'double':
                # a1.axis([new_x_min - buffer, new_x_max + buffer, new_y_min - buffer, new_y_max + buffer])
                #
                # elif testing and roundabout == 'double':
                #     if y_route[idx][0] > 0:
                #         a1.axis(
                #             [110 - buffer, 200 + buffer, 70 - buffer, 150 + buffer])
                #     else:
                #         a1.axis(
                #             [-113 - buffer, 107 + buffer, -393 - buffer, -193 + buffer])

                # elif y_route[idx][0] > 0:
                #     a1.axis([x_min_upper - buffer, x_max_upper + buffer, y_min_upper - buffer, y_max_upper + buffer])
                # else:
                #     a1.axis([x_min_lower - buffer, x_max_lower + buffer, y_min_lower - buffer, y_max_lower + buffer])

                # a1.axis([-80, 70,-85,85])

                # plt.axis([0, 1, 0, 1])
                a1.set_title(
                    f'{df.loc[idx,"EntryExit"]} {roundabout} {df.loc[idx, "Done"]}. Episode {idx}/{len(x_route)}. Total Episode Reward total {sum(df.loc[idx, "total_episode_reward"])}. W/O Last value {sum(df.loc[idx, "total_episode_reward"][:-1])}')
                a1.invert_yaxis()
                # a1.legend(loc='upper center')

                # if df.loc[idx, "Vehicle"] != "None":
                #     assert df.loc[idx, "Collisions"][0] in df.loc[idx,'Done']
                # else:
                #     assert "arrived" in df.loc[idx,'Done']

                # assert len(df.loc[idx, "point_reward"]) == len(x_truck[idx][2:])
                # assert len(df.loc[idx, "line_reward"]) == len(x_truck[idx][2:])
                assert len(df.loc[idx, "total_episode_reward"]) == len(x_truck[idx][2:])

                # a2.plot(np.array(df.loc[idx, "point_reward"]), label='Waypoint reward')
                # a2.plot(np.array(df.loc[idx, "line_reward"]), label='Line reward')
                # a2.plot(np.array(df.loc[idx, "total_episode_reward"]),
                #         label='Total Episode Reward')

                combined_rewards = []
                # for line_reward, point_reward in zip(df.loc[idx, "line_reward"],
                #                                      df.loc[idx, "point_reward"]):
                #     combined_rewards.append(line_reward + point_reward)
                # a2.plot(combined_rewards, label='Combined reward')

                # print(f'line Reward total {sum(df.loc[idx, "line_reward"])}')
                # print(f'Point Reward total {sum(df.loc[idx, "point_reward"])}')
                # print(f'Sums Reward total {sum(df.loc[idx, "point_reward"]) + sum(df.loc[idx, "line_reward"])}')
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

                print('-----VELOCITY------')
                last_velocity = 0
                diff_velocities = []
                for v, velocity in enumerate(df.loc[idx, "forward_velocity"]):
                    diff = velocity - last_velocity
                    if abs(diff) < 2:
                        print(diff,end='|')
                        diff_velocities.append(abs(diff))
                    last_velocity= velocity
                print('')
                print(f'Mean: {mean(diff_velocities)}')
                print(f'Max: {max(diff_velocities)}')
                print(f'Min: {min(diff_velocities)}')
                print('-----VELOCITY------')
                # point_reward = list(df.loc[idx, "point_reward"])
                # line_reward = list(df.loc[idx, "line_reward"])

                def divide(list, dividend):
                    new_list = []
                    for item in list:
                        new_list.append(item / dividend)
                    return new_list


                # print(f'New sum {sum(divide(point_reward,50)) + sum(divide(line_reward,100))}')




                #
                # assert len(
                #     df.loc[idx, "closest_distance_to_next_waypoint_line"]) - 2 == len(
                #     x_truck[idx][2:])
                # assert len(df.loc[
                #                idx, "closest_distance_to_next_plus_1_waypoint_line"]) - 2 == len(
                #     x_truck[idx][2:])
                #
                # a2.plot(df.loc[idx, "closest_distance_to_next_waypoint_line"],
                #         label='Distance to line')
                # a2.plot(df.loc[idx, "closest_distance_to_next_plus_1_waypoint_line"],
                #         label='+1 Distance to line')

                # temp = []
                #
                # for i, item in enumerate(
                #         df.loc[idx, "closest_distance_to_next_waypoint_line"]):
                #     if i != 0:
                #         line_reward_multiply_factor = 100
                #         hyp_reward = \
                #             df.loc[idx, "closest_distance_to_next_waypoint_line"][
                #                 i - 1] - item
                #
                #         hyp_reward = np.clip(hyp_reward, None, 0.5)
                #         hyp_reward = hyp_reward - 0.5
                #         temp.append(hyp_reward * line_reward_multiply_factor)
                #
                # # a2.plot(temp, label='Custom previous-current')
                # a2.axis([0, 1000, -100, 10])
                # a2.legend(loc='upper center')
                # a2.set_title(df.loc[idx, "Time"])

                # assert len(
                #     df.loc[idx, "closest_distance_to_next_waypoint_line"]) - 2 == len(
                #     x_truck[idx][2:])
                # assert len(df.loc[
                #                idx, "closest_distance_to_next_plus_1_waypoint_line"]) - 2 == len(
                #     x_truck[idx][2:])

                # a3.plot(df.loc[idx, "closest_distance_to_next_waypoint_line"], label='Distance to line')
                # a3.plot(df.loc[idx, "closest_distance_to_next_plus_1_waypoint_line"], label='+1 Distance to line')
                #
                # a3.axis([0, 500, -1, 25])
                # a3.legend(loc='upper center')
                items_to_plot_derived = []
                items_not_to_plot = ['Collisions', 'Done',
                                     'path', 'index', 'route','EntryExit', 'Time','Vehicle', 'trailer_path']


                for col in df.columns:
                    if col not in items_not_to_plot:
                        items_to_plot_derived.append(col)

                items_to_plot = [
                                 # ('trailer_bearing_to_waypoint', -math.pi, math.pi),
                                 ('forward_velocity',0,5),
                                 ('total_episode_reward',-2,2),
                                 ('truck_distance_to_center_of_lane',0,5),
                                 ('trailer_distance_to_center_of_lane',-3.5,3.5)
                                 ]

                if len(items_to_plot_derived) != len(items_to_plot):
                    items_not_found  =[]
                    for item in items_to_plot_derived:
                        found = False
                        for item_2 in items_to_plot  :
                            if item == item_2[0]:
                                found = True
                                break

                        if not found:
                            items_not_found.append(item)

                    if len(items_not_found) == 0:

                        for item in items_to_plot:
                            found = False
                            for item_2 in items_to_plot_derived:
                                if item[0] == item_2:
                                    found = True
                                    break

                            if not found:
                                items_not_found.append(item)



                    raise Exception(f'Mismatch in items to plot {items_not_found}')

                print(sorted(items_to_plot))
                items_to_plot = sorted(items_to_plot)
                num_of_cols = 4
                num_of_rows = ceil(len(items_to_plot) / num_of_cols)
                fig, axes = plt.subplots(ncols=num_of_cols, nrows=num_of_rows, figsize=(15, 10))

                for ax, col in zip(axes.ravel(), items_to_plot):
                    assert abs(len(df.loc[idx, col[0]]) - len(x_truck[idx][2:])) < 3
                    ax.plot(df.loc[idx, col[0]])
                    ax.set_title(str(col[0]), fontsize=10)
                    ax.set_ylim([col[1], col[2]])


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
if 'trailer_path' in df:
    plot_route(df["route"], df["path"], df['trailer_path'])
else:
    plot_route(df["route"], df["path"])

# y_dist_to_waypoints 0-> 0.023
# x_dist_to_waypoints
# acceleration
# forward_velocity
# forward_velocity_x
# forward_velocity_z
# angle_with_center
