import pickle
import numpy as np
from matplotlib import pyplot as plt

from torchvision.io import read_image
#
# image = rea("../../image_data/collision_data_inner_roundabout.pkl.pkl")
#
# import matplotlib.image as mpimg
# image_path = "../../image_data/lidar/03072023_212450443245.pkl.png"
# image = mpimg.imread(image_path)
# plt.imshow(image)
# plt.show()

def read_data_from_pickle(filename):
    with open(filename, 'rb') as handle:
        print(filename)
        return pickle.load(handle)

def save_data( filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

collision_data = read_data_from_pickle('../../image_data/collision_data_inner_roundabout.pkl')


collision_true = collision_data.loc[collision_data['done_collision'] == True]

for index, row in collision_true.iterrows():
    lidar = read_data_from_pickle(f"../../image_data/lidar/{row['filename']}.pkl")
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
