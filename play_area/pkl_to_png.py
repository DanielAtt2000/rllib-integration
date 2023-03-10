from PIL import Image
import os, pickle
import matplotlib.image
import cv2

def read_data_from_pickle(filename):
    with open(filename, 'rb') as handle:
        print(filename)
        return pickle.load(handle)



def convert_pickle_to_png():

    for filename in os.listdir("../image_data/lidar"):
        if not os.path.exists(f"../image_data/lidar/{filename}.png") and not filename.endswith(
                ".png") and "inner" not in filename:
            # if filename.endswith(".pkl"):
            cv2.imwrite(f"../image_data/lidar/{filename}.png", read_data_from_pickle(f"../image_data/lidar/{filename}"))
        else:
            print('here')


def remove_extra_name():
    for filename in os.listdir("../image_data/lidar"):
        if  filename.endswith(".pkl.png"):
            new_filename =filename.replace('.pkl','')
            new_filename =new_filename.replace('.png','')

            os.rename(f"../image_data/lidar/{filename}",f"../image_data/lidar/{new_filename}.png")
            print(filename)
            print(new_filename)

