"""Image util functions to operate."""

import cv2


def resize_image_scale(path_to_image, new_path_to_image, scale):
    """
    Resizes image to new scale in both x and y.
    Keeps ratio.
    Saves image to new_path_to_image
    :path_to_image: str : Path to image to be resized.
    :new_path_to_image: str : Path to save new image.
    :scale: float : Scale to resize image to.
    """
    img = cv2.imread(path_to_image)
    new_x = img.shape[1] * scale
    new_y = img.shape[0] * scale
    new_img = cv2.resize(img, (int(new_x), int(new_y)))
    cv2.imwrite(new_path_to_image, new_img)


def resize_image_fixed(path_to_image, new_path_to_image, size):
    """
    Resizes image to fixed size.
    Does not keep ratio.
    :path_to_image: str : Path to image to be resized.
    :new_path_to_image: str : Path to save new image.
    :size: Tuple[int, int] : New image size in pixels, (x, y).
    """
    img = cv2.imread(path_to_image)
    new_img = cv2.resize(img, size)
    cv2.imwrite(new_path_to_image, new_img)

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:42:06 2019
@author: Kraken

Project: Humpback Whale Challenge
"""

# =============================================================================
# Image Duplication
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import imagehash

import os
path = '.'
print(os.listdir(path))

TRAIN_IMG_PATH = './train'

def getImageMetaData(file_path):
    with Image.open(file_path) as img:
        img_hash = imagehash.phash(img)
        return img.size, img.mode, img_hash

def get_train_input():
    train_input = pd.read_csv('./train.csv')
    
    m = train_input.Image.apply(lambda x: getImageMetaData(TRAIN_IMG_PATH + "/" + x))
    train_input["Hash"] = [str(i[2]) for i in m]
    train_input["Shape"] = [i[0] for i in m]
    train_input["Mode"] = [str(i[1]) for i in m]
    train_input["Length"] = train_input["Shape"].apply(lambda x: x[0]*x[1])
    train_input["Ratio"] = train_input["Shape"].apply(lambda x: x[0]/x[1])
    train_input["New_Whale"] = train_input.Id == "new_whale"
    
    
    img_counts = train_input.Id.value_counts().to_dict()
    train_input["Id_Count"] = train_input.Id.apply(lambda x: img_counts[x])
    return train_input

train_input = get_train_input()

t = train_input.Hash.value_counts()
t = t.loc[t>1]
print("There are {} duplicate images.".format(np.sum(t)-len(t)))
t.head(20)

import collections

def plot_images(path, imgs):
    assert(isinstance(imgs, collections.Iterable))
    imgs_list = list(imgs)
    nrows = len(imgs_list)
    if (nrows % 2 != 0):
        nrows = nrows + 1 

    plt.figure(figsize=(18, 6*nrows/2))
    for i, img_file in enumerate(imgs_list):
#        print(img_file) # name of image in train folder
        with Image.open(path + "/" + img_file) as img:
            ax = plt.subplot(nrows/2, 2, i+1)
            ax.set_title("#{}: '{}'".format(i+1, img_file))
            ax.imshow(img)
        
    plt.show()

plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[0]].Image)

# =============================================================================
# Remove Duplicate Images
# =============================================================================

def remove_images(path, imgs):
    assert(isinstance(imgs, collections.Iterable))
    imgs_list = list(imgs)
    for i, img_file in enumerate(imgs_list):
        if i!=0:
            os.remove(path + '/' + img_file)
for i in range(len(t)):
    remove_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[i]].Image)


if __name__ == '__main__':

    IMAGE = './boop.png'    # change here
    TMP_IMG = 'resized_image.png'

    resize_image_scale(IMAGE, TMP_IMG, 10)
    cv2.imshow('resized_image.png')
    cv2.waitKey(0)

    resize_image_scale(IMAGE, TMP_IMG, (100, 100))
    cv2.imshow('resized_image.png')
    cv2.waitKey(0)
