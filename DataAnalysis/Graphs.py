from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import pandas as pd
import numpy as np

def save_to_pickle(filename, data):
    filename = filename + '.pickle'
    with open(f'{os.path.join(directory, filename)}', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def open_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

directory = 'data_0402b71f7bf'
df= pd.DataFrame()
df_done = pd.DataFrame()
string= '_beforeNormalisation'
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

def plot_route(route_points_all,truck_points_all):
    x_route = []
    y_route = []

    x_truck = []
    y_truck = []
    get_from_file = True
    if get_from_file:
        x_truck = open_pickle(os.path.join(directory, 'x_truck.pickle'))
        y_truck = open_pickle(os.path.join(directory, 'y_truck.pickle'))
        x_route = open_pickle(os.path.join(directory, 'x_route.pickle'))
        y_route = open_pickle(os.path.join(directory, 'y_route.pickle'))

    else:
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

    # Hack to remove
    temp_x_route = deepcopy(x_route)
    temp_y_route = deepcopy(y_route)

    for idx in range(len(x_route)):

        # if len(x_truck[idx]) > 0:
        if idx > 1000:
            # Hack to remove
            if idx != 0:
                x_route[idx] = temp_x_route[idx][len(temp_x_route[idx-1]):]
                y_route[idx] = temp_y_route[idx][len(temp_y_route[idx-1]):]

            x_min = min(min(x_route[idx]),min(x_truck[idx]))
            x_max = max(max(x_route[idx]),max(x_truck[idx]))

            y_min = min(min(y_route[idx]),min(y_truck[idx]))
            y_max = max(max(y_route[idx]),max(y_truck[idx]))
            buffer = 10

            # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
            plt.plot(x_route[idx][0], y_route[idx][0], 'bo',label='Route Starting waypoint')
            plt.plot(x_truck[idx][0], y_truck[idx][0], 'kd',label='Truck Starting waypoint')
            plt.plot(x_route[idx][2:], y_route[idx][2:], 'y^')
            plt.plot(x_truck[idx][2:], y_truck[idx][2:], "ro")
            plt.axis([x_min - buffer, x_max + buffer, y_min - buffer, y_max + buffer])
            # plt.axis([0, 1, 0, 1])
            # plt.title(f'{angle_to_center_of_lane_degrees * 180}')
            plt.gca().invert_yaxis()
            plt.legend(loc='upper center')
            plt.show()


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
            file = os.path.join(directory, filename)
            # checking if it is a file
            print(file)
            if os.path.isfile(file) and file.endswith('.pkl'):
                # plt.xlabel(filename)
                # sns.distplot(data)
                x = sns.displot(df[filename + string])
                # sns.pointplot(x=df['forward_velocity.pkl'],y=df['forward_velocity_x.pkl'])
                # sns.scatterplot(data)
                plt.show()
                x.savefig(os.path.join(directory, filename + string + '.png'))




# plot_route(df["route.pkl" + string],df["path.pkl" + string])

# y_dist_to_waypoints 0-> 0.023
# x_dist_to_waypoints
# acceleration
# forward_velocity
# forward_velocity_x
# forward_velocity_z
# angle_with_center

