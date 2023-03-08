import pickle

import numpy as np
from PIL import Image
import time

from matplotlib import pyplot as plt


def read_data_from_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)



data = read_data_from_pickle('collision_data_new.pkl')
lidar = read_data_from_pickle('lidar/03072023_175354097293.pkl')

for index, row in data.iterrows():
    path = f"lidar/{row['filename']}.pkl"
    row['filename'] = read_data_from_pickle(path)


temp = read_data_from_pickle('collision_data_2.pkl')
print(temp.index[temp['done_collision'] == True].tolist())
lidar = read_data_from_pickle('lidar/03072023_175354097293.pkl')
plt.figure()
xy_res = np.array(lidar).shape
plt.imshow(lidar, cmap="PiYG_r")
# cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
plt.clim(-0.4, 1.4)
plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
# plt.gca().invert_yaxis()
plt.show()

img = Image.fromarray(read_data_from_pickle('depth/03072023_175354097293.pkl'), None)
img.show()
time.sleep(0.005)
img.close()



print('HElo ')