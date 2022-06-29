from skimage import color
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import skimage.morphology as morpho 
import skimage.draw as draw 
from scipy import ndimage
from skimage import measure
from skimage.transform import rescale
from skimage import img_as_bool
from skimage.transform import resize
from skimage.morphology import convex_hull_image
from scipy.ndimage import gaussian_filter
from skimage.segmentation import clear_border
import time
from skimage.color import rgb2hsv
from scipy.spatial.distance import dice
from os.path import exists
from fastai.vision import *
from fastai.vision.all import*
from fastai.vision.data import ImageDataLoaders
from Image_preprocessing import *




def segmentation_otsu(img):
    '''
        Return the otsu segmantation of a gray scale image 
    '''
    img_r = img.copy()
    up_left_widow = img_r[:20,:20]
    if np.all(up_left_widow < 10/255):
        img_r[img_r<50/255] = 0
        
        border = clear_border(1 - img_r, bgval=254)
        border_indices = border==254
        border[border!=254] = 1.
        border[border==254] = 0.
        img = border * img
    
    min_sigma = np.inf
    tau_out = 0
    min_level = 0
    max_level = 255
    n_levels = max_level - min_level  
    for tau in np.linspace(min_level+1, max_level, n_levels-1):
        # first, get weights of the two regions
        omega_0 = img[np.logical_and(img>min_level, img<=tau)].sum()
        omega_1 = img[img>tau].sum()
        
        # only analyse thresholds which give two non-empty regions
        if (omega_0!=0 and omega_1!=0):
            sigma_0 = np.var(img[np.logical_and(img>min_level, img<=tau)])
            sigma_1 = np.var(img[img>tau])
            sigma_total = omega_0*sigma_0 + omega_1*sigma_1
            if (sigma_total < min_sigma):
                tau_out = tau
                min_sigma = sigma_total
                
    img_out = np.zeros((img.shape[0],img.shape[1]))
    img_out[np.logical_and(img>min_level, img<=tau_out)] = 1
    #img_out[img>tau_out] = 1

    return img_out,tau_out

def segment(img, channel=2):
    '''
        A method that takes an img and a channel and return the otsu segmentation after some preprocessing
    '''
    # Remove the unwanted blue region from some images 
    hsv_img = color.rgb2hsv(img)
    
    lower_blue = np.array([0.45,40/255,40/255]) 
    upper_blue = np.array([0.8,1,1])
    
    # Threshold the HSV image to get only blue colors
    binary_img = cv2.inRange(hsv_img, lower_blue, upper_blue)
    
    # We only select the blue channel to start
    imbr = img[:, :, channel]
    
    imbr=imbr*(1-binary_img/255)
    # We rescale to speed up computations
    im_rescaled = np.uint8(255*rescale(imbr , 0.5, anti_aliasing=True,  order=1, preserve_range=True))
    
    # Get the otsu segmantation
    Otsu,tau = segmentation_otsu(im_rescaled)
    return post_processing(Otsu)