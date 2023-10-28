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

from DataAnalysis.NewProcessData2 import data_file, for_graphs
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
        if idx > 5:

            entry_exit = df.loc[idx, 'EntryExit']
            entry = int(entry_exit.split(',')[0])
            exit = int(entry_exit.split(',')[1])
            roundabout, lane = get_route_type(entry, exit)

            if roundabout == '40m':
                pass
            else:
                continue
            plt.rcParams.update({'font.size': 13})
            plt.ylabel(f'Velocity (km/h)')
            plt.xlabel(f'Timesteps')
            plt.plot(df.loc[idx, 'forward_velocity'],color='blue', label='variable.velocity.no.smoothing')
            plt.plot(
[7.048597, 10.56761, 2.8894105, 0.13028622, 0.76398146, 0.74309766, 0.5547448, 0.36943007, 0.22950329, 0.13629064, 0.07813459, 0.044429272, 0.02353207, 0.01242082, 0.0063795387, 0.003202216, 0.0015451047, 0.0007884691, 0.00045761015, 0.00035542966, 0.00035784606, 0.0003337669, 0.00036608026, 0.0003514521, 0.7609375, 1.2557857, 1.9232389, 2.8117986, 3.4074733, 3.9440458, 4.662095, 5.2311215, 5.8137655, 6.449711, 6.81842, 7.214229, 7.5270786, 7.8425956, 8.0221195, 8.026988, 7.9551067, 7.794955, 7.612859, 7.3922276, 7.661725, 8.119041, 8.509185, 8.922516, 9.415055, 9.793848, 10.025795, 10.178207, 10.255969, 10.599771, 10.813776, 11.0079155, 11.131046, 11.2160635, 11.220092, 11.080876, 11.015356, 11.002345, 10.989648, 10.976787, 11.088489, 11.246982, 11.519895, 11.580537, 11.695482, 11.782156, 11.855986, 11.857571, 12.052724, 12.062824, 12.159181, 12.235053, 12.30161, 12.290485, 12.335499, 12.477765, 12.56598, 12.636573, 12.697907, 12.608938, 12.422831, 12.409967, 12.394804, 12.380002, 12.235142, 12.272382, 12.1737175, 12.313301, 12.382159, 12.415395, 12.318131, 12.406048, 12.331947, 12.361235, 12.367073, 12.359067, 12.223106, 12.127847, 12.1681795, 12.177938, 12.162794, 12.147447, 11.999206, 11.849718, 11.837273, 11.673249, 11.583261, 11.532908, 11.881532, 11.983759, 12.123509, 12.218469, 12.292818, 12.227444, 12.054469, 12.277004, 12.413467, 12.584062, 12.689962, 12.67788, 12.36311, 12.2993, 12.284468, 12.269916, 12.25515, 12.307262, 12.376756, 12.781875, 13.01702, 13.143696, 13.141869, 13.252413, 13.172521, 13.330026, 13.417154, 13.4689455, 13.391797, 13.425522, 13.516013, 13.565751, 13.565386, 13.35147, 13.304179, 13.1502, 13.135434, 13.030802, 12.847244, 12.955018, 12.906308, 12.979483, 13.039747, 13.094315, 13.005409, 12.802314, 13.000858, 13.115319, 13.277958, 13.33081, 13.683579, 13.766461, 13.899053, 13.980968, 14.040207, 14.037084, 14.019167, 14.001666, 13.984391, 13.967204, 13.741946, 13.179512, 13.018118, 12.984235, 12.977767, 12.903731, 12.868063, 13.079898, 13.176734, 13.330837, 13.413721, 13.3617935, 13.196561, 13.050998, 12.977574, 13.147963, 12.981316, 12.702715, 12.61624, 12.534212, 12.51403, 12.667803, 12.691708, 12.894584, 12.778414, 13.012826, 13.132894, 13.233721, 13.086057, 13.184572, 13.4979, 13.654485, 13.814584, 13.813007, 13.754161, 13.957088, 13.967748, 13.816873, 13.634201, 13.576382, 13.219325, 13.378133, 13.70112, 14.036087, 14.443091, 14.526725, 14.693709, 14.648506, 14.96966, 15.051695, 15.211417, 15.301935, 15.208198, 15.342077, 15.241535, 15.235809, 14.977288, 14.630124, 14.526743, 14.190495, 14.049052, 13.580304, 13.374144, 13.46543, 13.366556, 13.499463, 13.49506, 13.535459, 13.814086, 13.949763, 14.04919, 14.103405, 14.012145, 13.989981, 13.907218, 14.032215, 14.169611, 13.786763, 13.653783, 13.204685, 13.072389, 12.945894, 13.245305, 13.252008, 13.462728, 13.684663, 13.588413, 13.687488, 13.452009, 13.44728, 13.758927, 13.738771, 13.755375, 13.759204, 13.770025, 13.557314, 13.49703, 13.286494, 13.271438, 13.077597, 13.030877, 12.688227, 12.578667, 12.431455, 12.478483, 12.333433, 12.160073, 12.250086, 12.03044, 12.049815, 12.13059, 12.188913, 12.267697, 12.323555, 12.671736, 12.889176, 12.907594, 12.627526, 12.538511, 12.290577, 12.167122, 12.152996, 12.181263, 12.005877, 12.03558, 12.190068, 12.1798935, 12.228058, 12.172606, 12.081151, 12.081488, 11.96486, 12.059167, 12.066089, 12.059703, 12.042433, 12.003509, 11.983309, 11.968666, 11.813492, 11.499018, 11.407039, 11.236991, 11.101404, 11.031538, 10.947646, 11.027378, 11.115797, 11.43256, 11.659042, 11.836701, 11.877956, 11.992858, 12.32763, 12.374692, 12.570316, 12.685421, 12.606956, 12.356632, 12.287038, 12.259562, 12.241484, 11.9847975, 12.073581, 12.039269, 12.111421, 12.36627, 12.313459, 12.36058, 12.615129, 12.655082, 12.901678, 12.956537, 13.005778, 12.856143, 12.702291, 12.877789, 12.950198, 12.934632, 12.762302, 12.517594, 12.433022, 12.132016, 12.090426, 12.033921, 12.029129, 12.107503, 12.132596, 12.435941, 12.793203, 13.019924, 13.028226, 13.101392, 13.390084, 13.725489, 13.794785, 13.781728, 13.651023, 13.309063, 12.861333, 12.559075, 12.45163, 12.411514, 12.714403, 13.054235, 13.385832, 13.380429, 13.332512, 13.189355, 13.332285, 13.614032, 13.792962, 13.739195, 13.7434225, 13.974796, 14.271685, 14.577977, 14.889311, 14.9449625, 14.918648, 14.896178, 14.875534, 14.85569, 14.991344, 14.763873, 14.434895, 14.149334, 13.904355, 13.8958435, 13.555621, 13.309635, 13.021806, 12.826977, 12.8413315, 12.567961, 12.408056, 12.1811075, 12.101943, 11.984983, 11.921249, 12.017971, 12.179708, 12.087691, 12.06829, 11.971894, 12.11289, 12.28494, 12.143536, 12.013993, 11.791387, 11.69838, 11.647809, 11.223972, 10.944457, 10.594848, 10.373383, 10.238766, 10.3014765, 10.546896, 10.863668, 11.196869, 11.536114, 11.877085, 12.217752, 12.557298, 12.540805, 12.527996, 12.428972, 12.53012, 12.562916, 12.643375, 12.799917, 12.75326, 12.727989, 12.710257, 12.363063, 12.044253, 11.851404, 11.765474, 11.810855, 11.960694, 12.135262, 12.017354, 11.899864, 11.68534, 11.598372, 11.644136, 11.7942705, 11.673997, 11.586435, 11.393139, 11.4064865, 11.268614, 11.038239, 10.793839, 10.702515, 10.6741705, 10.862227, 11.145907, 11.46631, 11.799983, 12.137835, 12.476348, 12.814113, 12.781701, 12.746434, 12.6310215, 12.729068, 12.748756, 12.819379, 13.065802, 13.377819, 13.397201, 13.374798, 13.357479, 12.971524, 12.559644, 12.270277, 12.23123, 12.129841, 12.06173, 12.154126, 12.314807, 12.210171, 12.19591, 12.095265, 12.235063, 12.040875, 11.908848, 11.68867, 11.596958, 11.642902, 11.658243, 11.342758, 11.067972, 10.702361, 10.490176, 10.39765, 10.470257, 10.718355, 11.036471, 11.3703785, 11.70991, 12.050913, 12.3914175, 12.385242, 12.385165, 12.295157, 12.402208, 12.444982, 12.53035, 12.782338, 12.932817, 12.595629, 12.373978, 12.060633, 12.000031, 12.063855, 11.951585, 11.838761, 11.633229, 11.551387, 11.5982685, 11.379599, 11.297419, 11.127605, 11.161412, 11.227148, 11.205709, 11.160217, 11.266001, 11.432263, 11.556635, 11.538917, 11.525718, 11.514102, 11.503002, 11.517205, 11.727362, 11.690713, 11.846694, 11.937501, 12.239345, 12.185085, 12.197534, 12.125399, 12.237772, 12.290458, 12.379907, 12.633578, 12.6143875, 12.645435, 12.552751, 12.25885, 11.919325, 11.651006, 11.54859, 11.5163, 11.724555, 11.91993, 12.108054, 12.29338, 12.477522, 12.315773, 12.156522, 11.945421, 11.84918, 11.889945, 11.64957, 11.546551, 11.36378, 11.322404, 11.2649355, 11.049906, 10.954758, 10.9246, 10.907196, 10.8943815, 11.036509, 11.309979, 11.325906, 11.386531, 11.347263, 11.531306, 11.7504425, 11.863083, 11.960956, 12.220508, 12.539398, 12.868274, 13.200624, 13.533575, 13.865452, 13.965018, 13.543172, 13.339471, 13.005364, 12.936768, 12.6284, 12.4600935, 12.215466, 12.165604, 12.062047, 11.992942, 12.085124, 11.940962, 11.855323, 11.607076, 11.506569, 11.554051, 11.706388, 11.597692, 11.519725, 11.206408, 10.932277, 10.787865, 10.74541, 10.727139, 10.78099, 11.025191, 11.331228, 11.660128, 11.675556, 11.7182, 11.662415, 11.83919, 12.050659, 12.142828, 12.22717, 12.481269, 12.797926, 13.125372, 13.456435, 13.698795, 13.676508, 13.270885, 12.83391, 12.389188, 12.093682, 12.155757, 12.310898, 12.170922, 12.051493, 11.823181, 11.721526, 11.763519, 11.912536, 11.788306, 11.694046, 11.47656, 11.395888, 11.445782, 11.23842, 11.155104, 10.794172, 10.568275, 10.350923, 10.141248, 10.05972, 10.251151, 10.538345, 10.862426, 11.199972, 11.2536335, 11.336125, 11.307774, 11.447698, 11.558979, 11.6833935, 11.9502125, 12.2730255, 12.605221, 12.940153, 13.275207, 13.304871, 12.909077, 12.497662, 12.063531, 11.748072, 11.769059, 11.932494, 11.81072, 11.715243, 11.500076, 11.418509, 11.467256, 11.258501, 11.185883, 11.022394, 10.992916, 10.9576, 10.931772, 11.041763, 11.210242, 11.014736, 10.800235, 10.55166, 10.501669, 10.191182, 10.0804825, 10.034484, 10.218833, 10.471233, 10.660799, 10.80314, 11.081686, 11.409837, 11.746219, 11.8107195, 11.968228, 12.053213, 12.358893, 12.688365, 12.6867485, 12.775659, 12.761061, 12.594244, 12.347783, 11.94628, 11.92217, 11.905189, 11.874508, 11.958938, 11.884492, 12.065415, 12.178995, 12.261342, 12.327295, 12.386873, 12.337163, 12.390237, 12.3719635, 12.544185, 12.500857, 12.483607, 12.131447, 12.114002, 12.091105, 11.906917, 11.730177, 12.026058, 12.352594, 12.399144, 12.356688, 12.090858, 12.170313, 12.051982, 12.14379, 12.182532, 12.045854, 11.748347, 11.31618, 11.276311, 11.061959, 10.963903, 10.831015, 10.8162775, 10.805544, 10.7880335, 10.770546, 10.599062, 10.494775, 10.440687, 10.418939, 10.4187355, 10.43242, 10.4553385, 10.432114, 10.514268, 10.653173, 10.754773, 10.840286, 10.915336, 10.984171, 10.901976, 10.622584, 10.230204, 10.187082, 10.088614, 10.084538, 9.934799, 10.038437, 10.3823395, 10.7839365, 10.916893, 11.137923, 11.27048, 11.330673, 11.466861, 11.795236, 11.831954, 12.030519, 12.145205, 12.170669, 12.257259, 11.97838, 11.966482, 11.95303, 11.937591],color='red',label='variable.velocity.smoothing')

            plt.legend(loc='lower right')
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
                lidar = True
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
                    new_x_min = 0
                    new_x_max = 0
                    new_y_min = 0
                    new_y_max = 0
                elif roundabout == '20m':
                    new_x_min = 0
                    new_x_max = 0
                    new_y_min = 0
                    new_y_max = 0
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
                a1.axis([new_x_min - buffer, new_x_max + buffer, new_y_min - buffer, new_y_max + buffer])
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
                items_to_plot_derived = []
                items_not_to_plot = ['Collisions', 'Done', 'lidar_data', 'line_reward_location',
                                     'path', 'index',
                                     'point_reward_location', 'radii', 'route','EntryExit', 'Time','Vehicle', 'trailer_path']


                for col in df.columns:
                    if col not in items_not_to_plot:
                        items_to_plot_derived.append(col)

                items_to_plot = [('angle_between_truck_and_trailer',-1,1 ),

                                 ('angle_between_waypoints_5',0,1.1),
                                 ('angle_between_waypoints_7',0,1.1),
                                 ('angle_between_waypoints_10',0,1.1),
                                 ('angle_between_waypoints_12', 0, 1.1),

                                 ('angle_between_waypoints_minus5',0,1.1),
                                 ('angle_between_waypoints_minus7',0,1.1),
                                 ('angle_between_waypoints_minus10', 0, 1.1),
                                 ('angle_between_waypoints_minus12', 0, 1.1),


                                 ('angle_to_center_of_lane_degrees',-math.pi, math.pi ),
                                 ('angle_to_center_of_lane_degrees_2',-math.pi, math.pi ),
                                 ('angle_to_center_of_lane_degrees_5',-math.pi, math.pi ),
                                 ('angle_to_center_of_lane_degrees_7',-math.pi, math.pi ),
                                 ('angle_to_center_of_lane_degrees_ahead_waypoints',-math.pi, math.pi ),
                                 # ('angle_to_center_of_lane_degrees_ahead_waypoints_2',-math.pi, math.pi ),

                                 ('truck_bearing_to_waypoint', -math.pi, math.pi),
                                 ('truck_bearing_to_waypoint_2', -math.pi, math.pi),
                                 ('truck_bearing_to_waypoint_5', -math.pi, math.pi),
                                 ('truck_bearing_to_waypoint_7', -math.pi, math.pi),
                                 ('truck_bearing_to_waypoint_10', -math.pi, math.pi),

                                 ('trailer_bearing_to_waypoint', -math.pi, math.pi),
                                 ('trailer_bearing_to_waypoint_2', -math.pi, math.pi),
                                 ('trailer_bearing_to_waypoint_5', -math.pi, math.pi),
                                 ('trailer_bearing_to_waypoint_7', -math.pi, math.pi),
                                 ('trailer_bearing_to_waypoint_10', -math.pi, math.pi),

                                 # ('truck_bearing_to_ahead_waypoints_ahead_2',-math.pi, math.pi ),


                                 ('closest_distance_to_next_plus_1_waypoint_line',0,20),
                                 ('closest_distance_to_next_waypoint_line',0,20),
                                 ('forward_velocity',0,20),
                                 ('hyp_distance_to_next_plus_1_waypoint',0,20),
                                 ('hyp_distance_to_next_waypoint',0,20),
                                    # ('mean_radius',0,1.1),
                                 ('total_episode_reward',-10,10),
                                 ('distance_to_center_of_lane',0,5)
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
