import math
import os
from git import Repo
import pickle
import numpy as np
def min_max_normalisation(name, value):
    min = float(min_max_values[name][0])
    max = float(min_max_values[name][1])
    return (value-min)/(max-min)

no_changes = True
log = False
directory = '../data'

assert no_changes == True and log == False

repo = Repo('../')
remote = repo.remote('origin')
commit_hash = repo.head.commit

new_file_dir = f"data_{str(commit_hash)[:11]}"



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
            for line in lines:
                for data_entry in line.split(','):
                    data_entry = data_entry.strip()
                    for data in data_entry.split(','):
                        if data != '[]':
                            if data[0] == '[':
                                data = data[1:]
                            if data[-1] == ']':
                                data = data[:-1]

                            from numpy import exp
                            from scipy.stats import boxcox



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

            with open(os.path.join(new_file_dir,filename + '.pkl'), 'wb') as handle:
                pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



