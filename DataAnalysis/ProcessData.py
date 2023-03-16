import math
import os
from git import Repo
import pickle
import numpy as np
from numpy import exp
from scipy.stats import boxcox


def min_max_normalisation(name, value):
    min = float(min_max_values[name][0])
    max = float(min_max_values[name][1])
    return (value-min)/(max-min)

no_changes = True
log = False
directory = '../data/data_53b9a7ee095'

assert no_changes == True and log == False


new_file_dir = directory.split('/')[2]

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

            for line in lines:
                if filename == "done":
                    if line != "\n":

                        entry_pos = line.find("ENTRY: ")
                        exit_pos = line.find("EXIT: ")
                        dash_pos = line.find("-")

                        entry_point = line[entry_pos+len("ENTRY: "):exit_pos].strip()
                        exit_point = line[exit_pos+len("EXIT: "):dash_pos].strip()
                        output = line[dash_pos+len('-'):].strip()

                        temp_array_1.append(f"{entry_point},{exit_point}")
                        temp_array_2.append(output)
                elif filename == "trailer_collisions" or filename == "truck_collisions":
                    entry_pos = line.find("type=")
                    collision_object = str(line[entry_pos + len("type="):-3].strip())
                    new_data.append(collision_object)
                elif filename == "path" or filename == "route":
                    if line != "[]":
                        new_data.append(line)
                else:
                    for data_entry in line.split(','):
                        data_entry = data_entry.strip()
                        for data in data_entry.split(','):
                            if data != '[]':
                                if data[0] == '[':
                                    data = data[1:]
                                if data[-1] == ']':
                                    data = data[:-1]

                                if no_changes:
                                    new_data.append(float(data))
                                else:
                                    if log:
                                        # try:
                                        #     x = math.log10(float(data)+1)
                                        #     if x >3:
                                        #         print(data)
                                        #     new_data.append(x)
                                        # except ValueError:
                                        #     print(data)
                                        #     new_data.append(-10)

                                        # transform to be exponential
                                        data = exp(float(data))
                                        # power transform
                                        new_data.append(boxcox(data,-1))
                                    else:
                                        data = clip_custom(filename,float(data))
                                        data = min_max_normalisation(filename,float(data))
                                        new_data.append(float(data))

            if filename == "done":
                new_data.append([temp_array_1,temp_array_2])
            with open(os.path.join(new_file_dir,filename + '.pkl'), 'wb') as handle:
                pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



