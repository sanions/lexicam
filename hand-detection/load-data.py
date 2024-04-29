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
    '''
    Finds the smallest possible bounding box that encloses all the given coordinates.
    '''
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
    tl = (min_x, max_y)
    br = (max_x, min_y)
    
    # bounding box defined by its top-left and bottom-right corners
    bounding_box = [tl, br]
    
    return bounding_box


def resize_bbox(img, bbox, target_size=(128, 128)):
    '''
    Expand bounding box to the area around the hand until it reaches target_size, if possible.
    '''

    h, w = target_size
    tl, br = bbox
    min_x, max_y = tl
    max_x, min_y = br
    cropped = img[min_y:max_y, min_x:max_x]

    # resizing
    if cropped.shape[0] < w:
        mid_x = (min_x + max_x)/2
        min_x, max_x = mid_x - h/2, mid_x + h/2
    if cropped.shape[1] < h:
        mid_y = (min_y + max_y)/2
        min_y, max_y = mid_y - w/2, mid_y + w/2

    min_x = max(0, int(min_x))
    max_x = min(img.shape[1]-1, int(math.ceil(max_x)))
    min_y = max(0, int(min_y))
    max_y = min(img.shape[0]-1, int(math.ceil(max_y)))

    tl = (min_x, max_y)
    br = (max_x, min_y)

    return [tl, br]


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