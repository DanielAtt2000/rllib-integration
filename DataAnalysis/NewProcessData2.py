import math
import os

import pandas as pd
from git import Repo
import pickle
import numpy as np
from numpy import exp
from scipy.stats import boxcox


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Lidar:
    def __init__(self, time, front, front_left, front_right, left, right, trailer_0_left, trailer_1_left, trailer_2_left,
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
def get_value(text, symbol):
    location_of_symbol = text.find(symbol)
    end_of_value_comma = text[location_of_symbol:].find(',')
    end_of_value_bracket = text[location_of_symbol:].find(')')

    if end_of_value_comma < end_of_value_bracket and end_of_value_comma != -1:
        end_of_value = end_of_value_comma
    else:
        end_of_value = end_of_value_bracket

    return float(text[location_of_symbol + len(symbol) + 1:location_of_symbol + end_of_value])


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def min_max_normalisation(name, value):
    min = float(min_max_values[name][0])
    max = float(min_max_values[name][1])
    return (value-min)/(max-min)

no_changes = True
log = False
data_file = 'data_4a6c85c6b62'
directory = '/home/daniel/data-rllib-integration/data/' + data_file
for_graphs = False
def main():
    assert no_changes == True and log == False


    new_file_dir = directory.split('/')[-1]

    if not os.path.exists(new_file_dir):
        os.mkdir(new_file_dir)
    else:
        raise Exception('Path already exists')

    def clip_custom(name, value):
        max = float(min_max_values[name][1])
        return np.clip(value, 0,max)
    min_max_values = {'acceleration':[0,10],
                      'angle_with_center':[0,100],
                      'forward_velocity':[0,10],
                      'forward_velocity_x':[0,1.5],
                      'forward_velocity_z':[0,0.5],
                      'x_dist_to_waypoint':[0,0.03],
                      'y_dist_to_waypoint':[0,0.015],}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(file):
            with open(file,'r') as open_file:
                lines = open_file.readlines()

                new_data = []
                temp_array_1 = []
                temp_array_2 = []

                if 'radii' in filename:
                    continue

                for line in lines:
                    if filename == "collisions":

                        locations_of_USD = find(line, '$')
                        time = line[locations_of_USD[0] + 1:locations_of_USD[1]]
                        line = line[locations_of_USD[1] + 1:]

                        if "[]" in line:
                            new_data.append((time, 'None',Vector(x=0,y=0)))
                            continue

                        splits = line.split('\'')
                        vehicle = splits[1]
                        # time = splits[7]
                        new_data.append([time,vehicle, Vector(x = get_value(line, 'x'),y = get_value(line, 'y') )])
                    elif filename == "lidar_data":
                        temp_list = []
                        if "deque([]" in line:
                            continue

                        locations_of_USD = find(line, '$')
                        time = line[locations_of_USD[0] + 1:locations_of_USD[1]]
                        line = line[locations_of_USD[1] + 1:]

                        open_brackets_indices = find(line, '[')
                        closed_brackets_indices = find(line, ']')

                        lidar_points = []
                        if len(open_brackets_indices) >= 2:
                            lidar_points.append(line[open_brackets_indices[1]:closed_brackets_indices[0]])
                        if len(open_brackets_indices) >= 3:
                            lidar_points.append(line[open_brackets_indices[2]:closed_brackets_indices[1]])
                        if len(open_brackets_indices) >= 4:
                            lidar_points.append(line[open_brackets_indices[3]:closed_brackets_indices[2]])
                        if len(open_brackets_indices) >= 5:
                            lidar_points.append(line[open_brackets_indices[4]:closed_brackets_indices[3]])

                        for lidar_point in lidar_points:
                            splits = lidar_point.split(',')

                            temp_list.append(Lidar(time = splits[0].strip('[').strip("\'"),
                                                    front=float(splits[13]),
                                                  front_right=float(splits[16]),
                                                  front_left=float(splits[17]),
                                                  right=float(splits[14]),
                                                  left=float(splits[15]),
                                                  trailer_0_left=float(splits[1]),
                                                  trailer_1_left=float(splits[3]),
                                                  trailer_2_left=float(splits[5]),
                                                  trailer_3_left=float(splits[7]),
                                                  trailer_4_left=float(splits[9]),
                                                  trailer_5_left=float(splits[11]),
                                                  trailer_0_right=float(splits[2]),
                                                  trailer_1_right=float(splits[4]),
                                                  trailer_2_right=float(splits[6]),
                                                  trailer_3_right=float(splits[8]),
                                                  trailer_4_right=float(splits[10]),
                                                  trailer_5_right=float(splits[12]),))
                        new_data.append([time,temp_list])
                    elif filename == "done":
                        if line != "\n":

                            locations_of_USD = find(line, '$')
                            time = line[locations_of_USD[0] + 1:locations_of_USD[1]]
                            line = line[locations_of_USD[1] + 1:]

                            entry_pos = line.find("ENTRY: ")
                            exit_pos = line.find("EXIT: ")
                            dash_pos = line.find("-")

                            output = line[dash_pos + len('-'):].strip()
                            if output != '':
                                entry_point = line[entry_pos+len("ENTRY: "):exit_pos].strip()
                                exit_point = line[exit_pos+len("EXIT: "):dash_pos].strip()


                                temp_array_1.append(f"{entry_point},{exit_point}")
                                temp_array_2.append(output)
                            new_data.append([time, f"{entry_point},{exit_point}",output])
                    elif filename == "trailer_collisions" or filename == "truck_collisions":
                        entry_pos = line.find("type=")
                        collision_object = str(line[entry_pos + len("type="):-3].strip())
                        new_data.append(collision_object)
                    elif filename == "path" or filename == "route":
                        temp_array = []
                        if "[]" not in line:
                            data_entry = line.strip()
                            locations_of_USD = find(data_entry, '$')
                            time = data_entry[locations_of_USD[0] + 1:locations_of_USD[1]]
                            data_entry = data_entry[locations_of_USD[1] + 1:]

                            for truck_point in data_entry.split('),'):
                                truck_point = truck_point.replace('[', '')
                                truck_point = truck_point.replace(']', '')
                                truck_point = truck_point.replace('(', '')
                                truck_point = truck_point.replace(')', '')
                                truck_point = truck_point.replace('\n', '')
                                truck_point = truck_point.replace(' ', '')

                                truck_point = truck_point.split(',')

                                truck_point = (float(truck_point[0]), float(truck_point[1]))
                                temp_array.append(truck_point)

                            new_data.append([time, temp_array])
                    else:
                        if for_graphs:
                            for data_entry in line.split(','):
                                data_entry = data_entry.strip()
                                for data in data_entry.split(','):
                                    if data != '[]':
                                        if data[0] == '[':
                                            data = data[1:]
                                        if data[-1] == ']':
                                            data = data[:-1]
                                        new_data.append(float(data))
                        else:
                            temp_array = []
                            if '[]' not in line:
                                data_entry = line.strip()
                                locations_of_USD = find(data_entry, '$')
                                time = data_entry[locations_of_USD[0] + 1:locations_of_USD[1]]
                                data_entry = data_entry[locations_of_USD[1] + 1:]
                                for data in data_entry.split(','):
                                    if data[0] == '[':
                                        data = data[1:]
                                    if data[-1] == ']':
                                        data = data[:-1]
                                    temp_array.append(float(data))

                                new_data.append([time,temp_array])

                if filename == "done":
                    with open(os.path.join(new_file_dir, filename + '.pkl'), 'wb') as handle:
                        df = pd.DataFrame(new_data, columns=['Time', 'EntryExit', 'Done'])
                        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

                elif filename == 'collisions':
                    with open(os.path.join(new_file_dir, filename + '.pkl'), 'wb') as handle:
                        df = pd.DataFrame(new_data, columns=['Time', 'Vehicle', 'Collisions'])
                        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(os.path.join(new_file_dir,filename + '.pkl'), 'wb') as handle:
                        print(filename)
                        df = pd.DataFrame(new_data,columns=['Time', filename])
                        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done')
if  __name__ == "__main__" :
    main()

