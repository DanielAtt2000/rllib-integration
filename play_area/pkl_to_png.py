from PIL import Image

import os, pickle

def read_data_from_pickle(filename):
    with open(filename, 'rb') as handle:
        print(filename)
        return pickle.load(handle)
import matplotlib.image
import cv2
for filename in os.listdir("../image_data/lidar"):
    if not os.path.exists(f"../image_data/lidar/{filename}.png") and not filename.endswith(".png") and "inner" not in filename:
    # if filename.endswith(".pkl"):
        cv2.imwrite(f"../image_data/lidar/{filename}.png", read_data_from_pickle(f"../image_data/lidar/{filename}"))
    else:
        print('here')






