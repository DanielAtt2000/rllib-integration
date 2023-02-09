import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import pandas as pd
import numpy as np



directory = 'data_7ba3a56f814'
df= pd.DataFrame()
string= '_beforeNormalisation'
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file) and file.endswith('.pkl'):
        with open(file, 'rb') as handle:
            data = pickle.load(handle)

            df[str(filename+string)] = pd.Series(data)
            df = df.fillna(0)


for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file) and file.endswith('.pkl'):
        # plt.xlabel(filename)
        # sns.distplot(data)
        x = sns.displot(df[filename+string])
        # sns.pointplot(x=df['forward_velocity.pkl'],y=df['forward_velocity_x.pkl'])
        # sns.scatterplot(data)
        plt.show()
        x.savefig(os.path.join(directory,filename + string +'.png'))

# y_dist_to_waypoints 0-> 0.023
# x_dist_to_waypoints
# acceleration
# forward_velocity
# forward_velocity_x
# forward_velocity_z
# angle_with_center

