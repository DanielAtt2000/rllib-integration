import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import pandas as pd
import numpy as np



directory = 'data_d8ccb284873'
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

def plot_route(route_points,truck_points):
    x_route = []
    y_route = []

    x_truck = []
    y_truck = []
    for route_point in route_points:
        # print(f"X: {point.location.x} Y:{point.location.y}")
        x_route.append(route_point[0])
        y_route.append(route_point[1])

    for truck_point in truck_points:
        x_truck.append(truck_point[0])
        y_truck.append(truck_point[1])

    x_min = min(x_route)
    x_max = max(x_route)

    y_min = min(y_route)
    y_max = max(y_route)
    buffer = 10

    # print(f"X_TRUCK: {truck_normalised_transform.location.x} Y_TRUCK {truck_normalised_transform.location.y}")
    plt.plot([x_route.pop(0)], y_route.pop(0), 'bo')
    plt.plot(x_route, y_route, 'y^')
    plt.plot(x_truck, y_truck, "'gs'")
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




plot_route(df["route.pkl" + string],df["path.pkl" + string])

# y_dist_to_waypoints 0-> 0.023
# x_dist_to_waypoints
# acceleration
# forward_velocity
# forward_velocity_x
# forward_velocity_z
# angle_with_center

