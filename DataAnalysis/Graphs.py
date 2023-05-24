from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import pandas as pd
import numpy as np

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Lidar:
    def __init__(self,time,  front, front_left, front_right, left, right, trailer_0_left, trailer_1_left, trailer_2_left,
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

directory = 'data_d5885fe0125'
df= pd.DataFrame()
df_done = pd.DataFrame()
string= ''
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file) and file.endswith('.pkl'):
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            if "done" not in filename:
                df[str(filename+string)] = pd.Series(data)
                df = df.fillna(0)
            else:
                df_done['points'] = pd.Series(data[0][0])
                df_done['output'] = pd.Series(data[0][1])

graphs = False
get_from_file = True
def plot_route(route_points_all,truck_points_all):
    x_route = []
    y_route = []

    x_truck = []
    y_truck = []

    if get_from_file:
        x_truck = open_pickle(os.path.join(directory, 'x_truck.pickle'))
        y_truck = open_pickle(os.path.join(directory, 'y_truck.pickle'))
        x_route = open_pickle(os.path.join(directory, 'x_route.pickle'))
        y_route = open_pickle(os.path.join(directory, 'y_route.pickle'))

    else:
        route_points_all = route_points_all[route_points_all != 0]
        for route_id, route in enumerate(route_points_all):
            if "[]\n" in route:
                continue
            temp_x_route = []
            temp_y_route = []
            for route_point in route.split('),'):

                route_point = route_point.replace('[','')
                route_point = route_point.replace(']','')
                route_point = route_point.replace('(','')
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


                truck_point = truck_point.replace('[','')
                truck_point = truck_point.replace(']','')
                truck_point = truck_point.replace('(','')
                truck_point = truck_point.replace(')','')
                truck_point = truck_point.replace('\n','')
                truck_point = truck_point.replace(' ','')

                truck_point = truck_point.split(',')

                temp_x_truck.append(float(truck_point[0]))
                temp_y_truck.append(float(truck_point[1]))

            x_truck.append(temp_x_truck)
            y_truck.append(temp_y_truck)
            print(f"Truck Path {truck_idx}/{len(truck_points_all)} ready")

        # Removing the first truck position since this is extra when starting
        x_truck.pop(0)
        y_truck.pop(0)
        save_to_pickle(f'x_truck',x_truck)
        save_to_pickle(f'y_truck',y_truck)
        save_to_pickle(f'x_route',x_route)
        save_to_pickle(f'y_route',y_route)

    # # Hack to remove
    # temp_x_route = deepcopy(x_route)
    # temp_y_route = deepcopy(y_route)

    x_min = min(min(min(x_route), min(x_truck)))
    x_max = max(max(max(x_route), max(x_truck)))

    y_min = min(min(min(y_route), min(y_truck)))
    y_max = max(max(max(y_route), max(y_truck)))

    print(f'Number of episodes {len(x_route)}')
    for idx in range(len(x_route)):

        # if len(x_truck[idx]) > 0:

        if idx > 2050:
            # # Hack to remove
            # if idx != 0:
            #     x_route[idx] = temp_x_route[idx][len(temp_x_route[idx-1]):]
            #     y_route[idx] = temp_y_route[idx][len(temp_y_route[idx-1]):]

            if len(x_route[idx]) != 0:
                collision_data_diff = 2
                reward_data_diff = 2
                closest_distance_data_diff = 2
                print('----------')
                print(f'Showing Episode {idx}/{len(x_route)}')
                print(df_done.iloc[[idx-1]])
                for lidar_point in df.loc[idx+1,"lidar_data.pkl"]:

                    print(f"truck FRONT \t\t\t{round(lidar_point.front, 2)}")
                    print(f"truck 45 \t\t{round(lidar_point.front_left, 2)}\t\t{round(lidar_point.front_right, 2)}")
                    print(f"truck sides \t\t{round(lidar_point.left, 2)}\t\t{round(lidar_point.right, 2)}")
                    print(f"")
                    print(f"trailer_0 \t\t{round(lidar_point.trailer_0_left, 2)}\t\t{round(lidar_point.trailer_0_right, 2)}")
                    print(f"trailer_1 \t\t{round(lidar_point.trailer_1_left, 2)}\t\t{round(lidar_point.trailer_1_right, 2)}")
                    print(f"trailer_2 \t\t{round(lidar_point.trailer_2_left, 2)}\t\t{round(lidar_point.trailer_2_right, 2)}")
                    print(f"trailer_3 \t\t{round(lidar_point.trailer_3_left, 2)}\t\t{round(lidar_point.trailer_3_right, 2)}")
                    print(f"trailer_4 \t\t{round(lidar_point.trailer_4_left, 2)}\t\t{round(lidar_point.trailer_4_right, 2)}")
                    print(f"trailer_5 \t\t{round(lidar_point.trailer_5_left, 2)}\t\t{round(lidar_point.trailer_5_right, 2)}")
                    print(f"------------------------------------------------------")

                    if df.loc[idx+collision_data_diff,"collisions.pkl"][1] != 0:
                        if lidar_point.time != df.loc[idx+collision_data_diff,"collisions.pkl"][1]:
                            print('------------------------')
                            print(f"Lidar time \t\t\t{lidar_point.time}")
                            print(f'Collision time \t\t{df.loc[idx+collision_data_diff,"collisions.pkl"][1]}')
                            print('------------------------')




                print()
                print()
                print('----------')

                buffer = 10

                fig,axes = plt.subplots(ncols=2,nrows=1,figsize=(15, 10))
                a1 = axes[0]
                a2 = axes[1]
                # a3 = axes[2]
                # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
                a1.plot(x_route[idx+1][0], y_route[idx+1][0], 'bo',label='Route Starting waypoint')
                a1.plot(x_truck[idx][0], y_truck[idx][0], 'kd',label='Truck Starting waypoint')
                a1.plot(x_route[idx+1][2:], y_route[idx+1][2:], 'y^')
                a1.plot(x_truck[idx][2:], y_truck[idx][2:], "ro")
                a1.plot(df.loc[idx+collision_data_diff,"collisions.pkl"][2].x,df.loc[idx+collision_data_diff,"collisions.pkl"][2].y,'b*')


                a1.axis([x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer])
                # plt.axis([0, 1, 0, 1])
                a1.set_title(f'Collision with {df.loc[idx+2,"collisions.pkl"][0]}. Episode {idx}/{len(x_route)}')
                a1.invert_yaxis()
                a1.legend(loc='upper center')

                assert len(df.loc[idx+reward_data_diff,"point_reward.pkl"]) == len(x_truck[idx][2:])

                a2.plot(df.loc[idx+reward_data_diff,"point_reward.pkl"],label='Waypoint reward')
                a2.plot(df.loc[idx+reward_data_diff,"line_reward.pkl"],label='Line reward')
                combined_rewards = []
                for line_reward, point_reward in zip(df.loc[idx+reward_data_diff,"line_reward.pkl"],df.loc[idx+reward_data_diff,"point_reward.pkl"]):
                    combined_rewards.append(line_reward+point_reward)
                # a2.plot(combined_rewards, label='Combined reward')

                a2.plot(df.loc[idx + closest_distance_data_diff, "closest_distance_to_next_waypoint_line.pkl"],
                        label='Distance to line')
                a2.plot(df.loc[idx + closest_distance_data_diff, "closest_distance_to_next_plus_1_waypoint_line.pkl"],
                        label='+1 Distance to line')

                temp = []

                for i,item in enumerate(df.loc[idx + closest_distance_data_diff, "closest_distance_to_next_waypoint_line.pkl"]):
                    if i != 0:
                        line_reward_multiply_factor = 100
                        hyp_reward = df.loc[idx + closest_distance_data_diff, "closest_distance_to_next_waypoint_line.pkl"][i-1]-item

                        hyp_reward = np.clip(hyp_reward, None, 0.5)
                        hyp_reward = hyp_reward - 0.5
                        temp.append(hyp_reward * line_reward_multiply_factor)

                # a2.plot(temp, label='Custom previous-current')
                a2.axis([0,500,-55,25])
                a2.legend(loc='upper center')

                assert len(df.loc[idx+closest_distance_data_diff, "closest_distance_to_next_waypoint_line.pkl"])-2 == len(x_truck[idx][2:])
                assert len(df.loc[idx+closest_distance_data_diff, "closest_distance_to_next_plus_1_waypoint_line.pkl"])-2 == len(x_truck[idx][2:])

                # a3.plot(df.loc[idx+closest_distance_data_diff, "closest_distance_to_next_waypoint_line.pkl"], label='Distance to line')
                # a3.plot(df.loc[idx+closest_distance_data_diff, "closest_distance_to_next_plus_1_waypoint_line.pkl"], label='+1 Distance to line')
                #
                # a3.axis([0, 500, -1, 25])
                # a3.legend(loc='upper center')


                plt.show()


if graphs:
    for filename in os.listdir(directory):
        if "forward_velocity_z" not in filename:
            if "done" in filename:

                x= sns.catplot(df_done,x="output",y="points")
                print("------------------------------")
                print("DONE DATA")
                print(df_done.output.value_counts())
                print(df_done.points.value_counts())
                print("------------------------------")

                # x = sns.swarmplot(df_done,x="output",y="points")
                # sns.scatterplot(data)
                plt.show()
                x.savefig(os.path.join(directory, filename + string + '.png'))
            elif "route" in filename or "path" in filename:
                continue
            else:
                for col in df.columns:
                    print(df[col].describe(include='all'))
                    print('--------------')
                file = os.path.join(directory, filename)
                # checking if it is a file
                print(file)
                if os.path.isfile(file) and file.endswith('.pkl'):
                    # plt.xlabel(filename)
                    # sns.distplot(data)
                    if "hyp_distance" in filename or "closest_distance" in filename:
                        # sns.distplot(df[filename+ string], bins=10, kde=False)
                        values_greater_than = [len(np.where(df[filename+ string]>10)[0]),
                                               len(np.where(df[filename+ string]>20)[0]),
                                               len(np.where(df[filename+ string]>30)[0]),
                                               len(np.where(df[filename+ string]>40)[0]),
                                               len(np.where(df[filename+ string]>50)[0])]

                        temp = [10,20,30,40,50]
                        for value in range(5):
                            print(f"Values greater than {temp[value]}")
                            print(values_greater_than[value])
                            print(values_greater_than[value]/len(df[filename+ string]) * 100)
                            print('---------------')

                        difference = []
                        for i,item in enumerate(df.loc[:, filename+ string]):
                            if i == 0:
                                continue

                            difference.append(df.loc[i-1, filename+ string] - item )

                        series_difference = pd.Series(difference)
                        print('----------')
                        values_greater_than = []
                        temp = [100]
                        for value in temp:
                            values_greater_than.append(len(np.where(series_difference > 0.5)[0]))


                        for value in range(len(temp)):
                            print(f"Values greater than {temp[value]}")
                            print(values_greater_than[value])
                            print(values_greater_than[value] / len(series_difference) * 100)
                            print('---------------')

                        print(series_difference.describe())
                        sns.distplot(series_difference, bins=10, kde=False)

                        plt.show()
                        print('----------')

                        plt.show()
                    else:
                        x = sns.displot(df[filename + string])
                        # sns.pointplot(x=df['forward_velocity.pkl'],y=df['forward_velocity_x.pkl'])
                        # sns.scatterplot(data)
                        plt.show()
                        x.savefig(os.path.join(directory, filename + string + '.png'))




plot_route(df["route.pkl" + string],df["path.pkl" + string])

# y_dist_to_waypoints 0-> 0.023
# x_dist_to_waypoints
# acceleration
# forward_velocity
# forward_velocity_x
# forward_velocity_z
# angle_with_center

