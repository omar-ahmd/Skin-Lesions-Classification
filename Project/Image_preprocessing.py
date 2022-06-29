from skimage.transform import warp, AffineTransform
import numpy as np
from skimage import measure
from skimage.segmentation import clear_border
from skimage.transform import resize
import skimage.morphology as morpho 
import cv2
from scipy import ndimage
from skimage import color
import numpy as np # linear algebra
import pandas as pd # data processing
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

def crop_imge(img, seg, shape):
    indices = np.where(seg==1)
    x = np.min(indices[0])
    y = np.min(indices[1])
    w = np.max(indices[0]) - x 
    h = np.max(indices[1]) - y
    if x==0 and y==0 and w==255 and h==255: return resize(img[10:-10, 10:-10], shape)
    cX = int(x + w/2)
    cY = int(y + h/2)
    tform = AffineTransform(translation=(-cY + int(seg.shape[1]/2), -cX + int(seg.shape[0]/2)))
    img_centered = warp(img, tform.inverse)#, mode='reflect')
    cX = int(img.shape[0]/2)
    cY = int(img.shape[1]/2)
   
    
    if seg[seg==1].size/seg.size < 0.007: 
        x = cX - int(0.15*seg.shape[0]/2)
        w = int(0.15*seg.shape[0])
        
        y = cY - int(0.15*img.shape[0]/2)
        h = int(0.15*seg.shape[1]) 
        new_img = img_centered[x : x + w, y : y + w]
        
    elif w > h :
        x = cX - int(w/2)
        y = cY - int(w/2)
        new_img = img_centered[x : x + w, y : y + w]
    else:
        x = cX - int(h/2)
        y = cY - int(h/2)
        new_img = img_centered[x : x + h, y : y + h]

    indices = np.where(new_img>0)
    x = np.min(indices[0])
    y = np.min(indices[1])
    w = np.max(indices[0]) - x 
    h = np.max(indices[1]) - y
    return resize(new_img[x : x + w, y : y + w], shape)


def clear_black_border(im):
    img = np.array(im)[:,:,0]
    img_r = img.copy()
    up_left_widow = img_r[:20,:20]
    if np.all(up_left_widow < 10):
        img_r[img_r<50] = 0
        border = clear_border(255 - img_r, bgval=254)
        border_indices = border==254
        border[border!=254] = 1.
        border[border==254] = 0.
        img = border * img
        contours = measure.find_contours(border,0)
        cnt = contours[0]
        x,y,w,h = cnt[:,0].min(), cnt[:,1].min(), cnt[:,0].max(), cnt[:,1].max()
        return x,y,w,h
    return 0,0,img.shape[0],img.shape[1]


def post_processing(mask, convex=True):
    # An opening and closing to remove small white and black regions
    se=morpho.disk(2)
    ot=morpho.opening(mask,se)
    se=morpho.disk(2)
    ot=morpho.closing(ot,se) 
    
    #Finding the best segmentation using a the size of the area and the distance from the center
    sim=[]
    masks=[]
    
    # Add black border to reduce the border effect on choice of the best segmentation
    ot = cv2.copyMakeBorder(
    ot, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    contours = measure.find_contours(ot,0)#cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    for contour in contours:
        # Create an empty image to store the masked array
        r_mask = np.zeros_like(ot, dtype='bool')
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        x = np.min(contour[:,0])
        y = np.min(contour[:,1])
        w = np.max(contour[:,0]) - x 
        h = np.max(contour[:,1]) - y
        cX = x+w/2
        cY = y+h/2
        dist_to_center = np.sqrt((cX - ot.shape[0]/2)**2 + (cY - ot.shape[1]/2)**2)
        
        r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1     
        # Fill in the hole created by the contour boundary
        r_mask = ndimage.binary_fill_holes(r_mask)
        x,y = np.where(r_mask==1)
        cX_neigh = np.arange(int(cX)-10, int(cX)+10, 1)
        cY_neigh = np.arange(int(cY)-10, int(cY)+10, 1)
        if np.any(r_mask[cX_neigh,cY_neigh]==1) :
            if convex:
                r_mask = convex_hull_image(r_mask, tolerance=1e-10)
            area = r_mask[r_mask==1].size/ot.size
            masks.append(r_mask)
                
                
            dist_to_center = dist_to_center/np.sqrt((ot.shape[0]/2)**2 + (ot.shape[1]/2)**2)
            func_to_min = dist_to_center - area
            sim.append(func_to_min)
    try:
        ind = np.argmin(sim)
        result = masks[ind]
        if result[10:-10,10:-10].sum()/result[10:-10,10:-10].size < 0.005:
            result[60:200,60:200] = 1
    except:
        result = np.ones(ot.shape)

    return result[10:-10,10:-10]





def strel(forme,taille,angle=45):
    """renvoie un element structurant de forme  
     'diamond'  boule de la norme 1 fermee de rayon taille
     'disk'     boule de la norme 2 fermee de rayon taille
     'square'   carre de cote taille (il vaut mieux utiliser taille=impair)
     'line'     segment de langueur taille et d'orientation angle (entre 0 et 180 en degres)
      (Cette fonction n'est pas standard dans python)
    """

    if forme == 'diamond':
        return morpho.diamond(taille)
    if forme == 'disk':
        return morpho.disk(taille)
    if forme == 'square':
        return morpho.square(taille)
    if forme == 'line':
        angle=int(-np.round(angle))
        angle=angle%180
        angle=float(angle)/180.0*np.pi
        x=int(np.round(np.cos(angle)*taille))
        y=int(np.round(np.sin(angle)*taille))
        if x**2+y**2 == 0:
            if abs(np.cos(angle))>abs(np.sin(angle)):
                x=int(np.sign(np.cos(angle)))
                y=0
            else:
                y=int(np.sign(np.sin(angle)))
                x=0
        rr,cc= draw.line(0,0,y,x)
        rr=rr-rr.min()
        cc=cc-cc.min()
        img=np.zeros((rr.max()+1,cc.max()+1) )
        img[rr,cc]=1
        return img
    raise RuntimeError('Erreur dans fonction strel: forme incomprise')

    

def dullRazor(imbr):
    #fermeture par un ligne
    sizeLine=4
    line_0=strel('line',sizeLine,angle = 0)
    closing_line_0=morpho.closing(imbr,line_0)
    line_15=strel('line',sizeLine,angle = 15)
    closing_line_15=morpho.closing(imbr,line_15)
    line_30=strel('line',sizeLine,angle = 30)
    closing_line_30=morpho.closing(imbr,line_30)
    line_45=strel('line',sizeLine,angle = 45)
    closing_line_45=morpho.closing(imbr,line_45)
    line_60=strel('line',sizeLine,angle = 60)
    closing_line_60=morpho.closing(imbr,line_60)
    line_75=strel('line',sizeLine,angle = 75)
    closing_line_75=morpho.closing(imbr,line_75)
    line_90=strel('line',sizeLine,angle = 90)
    closing_line_90=morpho.closing(imbr,line_90)

    
    stacked = np.dstack((closing_line_0, closing_line_15, closing_line_30, closing_line_45, closing_line_60, closing_line_75, closing_line_90))
    stacked = np.max(stacked,axis=2)


    mask = np.abs(stacked - imbr)
    if np.sum(mask) > 95000: # Apply the inpainting only when the size of the mask is greater than certain constant
        # inpaint the original image depending on the mask
        image_result = cv2.inpaint(imbr,mask,1,cv2.INPAINT_TELEA)
        image_result = ndimage.median_filter(image_result, size=1)
    else: 
        image_result = gaussian_filter(imbr, sigma=1)
    
    return image_result