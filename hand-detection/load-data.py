import os
import json
import math

import matplotlib.pyplot as plt
import pandas as pd


## testing code
outpath = "./test-outputs/"
if not os.path.isdir(outpath):
    os.makedirs(outpath)

# 1. get bounding boxes for hands
def calc_basic_bbox(coordinates):
    
    if not coordinates:
        return None
    
    min_x = min(coord[0] for coord in coordinates)
    max_x = max(coord[0] for coord in coordinates)
    min_y = min(coord[1] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)

    # make coordinates integers, keeping it inclusive of all coordinates
    min_x = int(min_x)
    max_x = int(math.ceil(max_x))
    min_y = int(min_y)
    max_y = int(math.ceil(max_y))
    
    # define the corners of the bounding box
    top_left = (min_x, max_y)
    top_right = (max_x, max_y)
    bottom_left = (min_x, min_y)
    bottom_right = (max_x, min_y)
    
    # bounding box defined by its corners
    bounding_box = [top_left, top_right, bottom_right, bottom_left]
    
    return bounding_box


# def resize_bbox(img, bbox, target_size=(128, 128)):



# load data
def load(dir_name): 

    df = pd.DataFrame(columns=["img_filepath", "img_arr", "bbox"])
    label_fnames = sorted([f for f in os.listdir(dir_name) if f.endswith('.json')])    

    for fname in label_fnames: 
        img_fname = fname[:fname.find(".json")] + ".jpg"
        img = plt.imread(dir_name+img_fname)

        drow = {
            "img_filepath": dir_name + img_fname, 
            "img_arr": img, 
            #TODO: calculate bbox coords
            "bbox": None 
        }
        break


if __name__ == "__main__": 
    load(dir_name="./dataset/train/", out=outpath)