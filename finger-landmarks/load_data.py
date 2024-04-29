import os

import pandas as pd

home_dir = os.path.expanduser("~")
dataset_dir = os.path.join(home_dir, "Documents/classes/lexicam/dataset/")


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


def expand_bbox(img, bbox, target_size):
    '''
    Expands bounding box to the area around the hand until it reaches target_size, if possible.

    REMINDER: Resulting bounding box could be too large. Use cv2.resize() or similar to fix. 
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


def calc_bbox(img, hand_pts, target_size): 
    '''
    Calculates bounding box [bottom_left, top_right]. 

    REMINDER: Resulting bounding box could be too large. Use cv2.resize() or similar to fix. 
    '''
    bbox = calc_basic_bbox(hand_pts)
    return expand_bbox(img, bbox, target_size


def create_dataset(dir_name, target_size=(128, 128)):

    # initialize df
    # go through images in dataset -- for each filename: 

        # get img
        # get labels

        # calculate bounding box for img using calc_bbox
        # crop img to bounding box
        # resize images to target_size using cv2.resize

        # recalculate labels 
            # first use left, bottom of bbox to shift landmark coords
            # rescale labels based on resized scale factors

        # save cropped & resized image to new dir
        # add row containing new img filename and recalculated labels to df
    
    pass


    


if __name__ == "__main__": 
    create_dataset(os.path.join(dataset_dir, 'train'))