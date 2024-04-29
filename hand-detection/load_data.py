import os
import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

testing = False
if testing: 
    outdir = "test-outputs"
else: 
    outdir = "outputs"

outpath = os.path.join(os.path.dirname(__file__), outdir)
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


def clean_data(dir_name, out_file, target_size=(128, 128)): 

    df = pd.DataFrame(columns=["img_filepath", "img_size", "bbox_tl_x", "bbox_tl_y", "bbox_br_x", "bbox_br_y"])
    label_fnames = sorted([f for f in os.listdir(dir_name) if f.endswith('.json')])    

    for fname in label_fnames: 
        # read label file & get hand landmarks coordinates
        with open(dir_name+fname, "r") as f: 
            labels = json.loads(f.read())
        hand_pts = labels["hand_pts"]

        # read corresponding image
        img_fname = fname[:fname.find(".json")] + ".jpg"
        img = plt.imread(dir_name+img_fname)

        if img.shape != (576, 720, 3): 
            continue

        tl, br = calc_basic_bbox(hand_pts)

        ### TESTING CODE! Saves bounding box image to outpath. ###
        # top_left, bottom_right = bbox
        # plt.imshow(img)
        # x_values = [top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]]
        # y_values = [top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]]
        # plt.plot(x_values, y_values, color='red')
        # plt.savefig(os.path.join(outpath, "out.jpg"))

        nrow = {
            "img_filepath": dir_name + img_fname, 
            "img_size": img.shape,
            "bbox_tl_x": tl[0], 
            "bbox_tl_y": tl[1], 
            "bbox_br_x": br[0], 
            "bbox_br_y": br[1], 
        }

        df.loc[len(df)] = nrow
    
    df.to_csv(os.path.join(outpath, out_file))
    return df


def load_images(df):
    '''
    Returns a (num_samples x (imgsize) x 3) normalized numpy array containing 
    the images in given df in the correct format for input to model. 
    '''
    images = []
    for ix, row in df.iterrows(): 
        img = plt.imread(row["img_filepath"])
        images.append(img)

    images = np.array(images)
    return images


def load_labels(df):
    '''
    Returns a (num_samples x 4) numpy array containing the (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    of each bounding box in given df. Prepares data to be input to model.
    '''
    labels = []
    for ix, row in df.iterrows(): 
        label = [int(row["bbox_tl_x"]), 
                int(row["bbox_tl_y"]), 
                int(row["bbox_br_x"]), 
                int(row["bbox_br_y"])] 
        labels.append(label)

    labels = np.array(labels)
    return labels
    

if __name__ == "__main__": 
    home_dir = os.path.expanduser("~")
    dataset_dir = os.path.join(home_dir, "Documents/classes/lexicam/dataset/test/")
    df = clean_data(dir_name=dataset_dir, out_file="hands_bbox_test.csv")
    print(df)