#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:33:12 2022

@author: nastysushi
"""


import numpy as np
import cv2 
from lib.videoConf import CFEVideoConf

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('assets/footage.mp4')

fps = 50
#config = CFEVideoConf(cap, filepath=save_path, res='480p')
#out = cv2.VideoWriter(save_path, config.video_type, fps, config.dims)
curr_frame = 0


# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
    
    
# The code does output .avi because MPEG-4 codec gave troubles on mac, but the resulting video is
# converted and downsampled to a MPEG-4 format.
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

# =============================================================================
# def apply_invert(frame):
#     return cv2.bitwise_not(frame)
# 
# def verify_alpha_channel(frame):
#     try:
#         frame.shape[3] # 4th position
#     except IndexError:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
#     return frame
# 
# 
# def apply_color_overlay(frame, 
#             intensity=0.2, 
#             blue = 0,
#             green = 0,
#             red = 0):
#     frame = verify_alpha_channel(frame)
#     frame_h, frame_w, frame_c = frame.shape
#     color_bgra = (blue, green, red, 1)
#     overlay = np.full((frame_h, frame_w, 4), color_bgra, dtype='uint8')
#     cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
#     return frame
# 
# def apply_sepia(frame, intensity=0.5):
#     blue = 20
#     green = 66 
#     red = 112
#     frame = apply_color_overlay(frame, 
#             intensity=intensity, 
#             blue=blue, green=green, red=red)
#     return frame
# 
# 
# def alpha_blend(frame_1, frame_2, mask):
#     alpha = mask/255.0 
#     blended = cv2.convertScaleAbs(frame_1*(1-alpha) + frame_2*alpha)
#     return blended
# 
# 
# def apply_circle_focus_blur(frame, intensity=0.2):
#     frame           = verify_alpha_channel(frame)
#     frame_h, frame_w, frame_c = frame.shape
#     y = int(frame_h/2)
#     x = int(frame_w/2)
#     radius = int(x/2) # int(x/2)
#     center = (x,y)
#     mask    = np.zeros((frame_h, frame_w, 4), dtype='uint8')
#     cv2.circle(mask, center, radius, (255,255,255), -1, cv2.LINE_AA)
#     mask    = cv2.GaussianBlur(mask, (21,21),11 )
#     blured  = cv2.GaussianBlur(frame, (21,21), 11)
#     blended = alpha_blend(frame, blured, 255-mask)
#     frame   = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
#     return frame
# 
# def apply_portrait_mode(frame):
#     frame           = verify_alpha_channel(frame)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 120,255,cv2.THRESH_BINARY)
#     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
#     blured = cv2.GaussianBlur(frame, (21,21), 11)
#     blended = alpha_blend(frame, blured, mask)
#     frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
#     return frame
# =============================================================================
 



def grayscale():
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return mask


def gaussianBlur():
    val = round((curr_frame-200)/10, 1)
    mask = cv2.GaussianBlur(frame, (9,9), val)
    #print(val)
    return mask

def bilateralBlur(x):
    mask = cv2.bilateralFilter(frame,x,75,75)
    return mask    

 
def grabObjectHSV():
    """
    Grabs an object in RGB and HSV color space. 
        Show binary frames with the foreground object 
        in white and background in black.

    Returns
    -------
    mask : TYPE
        Yellow: #FFFF00, 
        Retuns a filter based on ....

    """
    # BGR: Blue, green, red 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    
    mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    return mask


def sobel():
    """
    

    Returns
    -------
    sobelx : TYPE
        Change in color intensity to detect edges by taking first derivative.

    """
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1,0, ksize=3)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0,1, ksize=3)
    
    return sobelx


     
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(curr_frame)

    curr_filter = frame
    
    ############### BASIC IMAGE PROCESSING ######################
    # grayscale
    curr_filter = grayscale()
    
    # blur
    ## gaussian filter
    curr_filter = gaussianBlur()
    
    # bi-lateral filter
    # TODO: every second, +7 waarde
    curr_filter = bilateralBlur(2)
    
    # mask RGB_HSV
    # TODO: Finetuning parameters
    curr_filter = grabObjectHSV()
    
    
    ############### OBJECT DETECTION ######################
    
    # sobel edge detection (5s)
    curr_filter = sobel()
    
    
    # hough transform 
    
    
    
    
    
    
    
    
    ## TODO LIST
    # 1. met vaste video van 1min beginnen werken
    # 2. Export video mogelijk .avi (?)
    # 3. output nu met filter (TODO)
    # 3. Grabobject afstemmen op kleur
    
    

    """
    #########################     0-4s: color <-> grayscale 2x     #########################

    if(fps < curr_frame <= fps*2 or fps*3 < curr_frame <= fps*4):
        curr_filter = grayscale()
    elif(fps*2 < curr_frame <= fps*3):
        curr_filter = frame
    
    #########################            4-12s: blur                #########################
    
    ## Gaussian filter
    # curr frame between fps 200-400
    if(fps*4 < curr_frame <= fps*8):
        # V1: loopje dat gaussian filter opent
        val = round((curr_frame-200)/10, 1)
        curr_filter = cv2.GaussianBlur(frame, (9,9), val)
        print(val)
        
        #val = round((curr_frame-200)/10, 1)
        #curr_filter = cv2.GaussianBlur(frame, (9,9), val)
        
    ## Bi-lateral filter
    # highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters.
    elif(fps*8 < curr_frame <= fps*9):
        curr_filter = cv2.bilateralFilter(frame,2,75,75)
    elif(fps*9 < curr_frame <= fps*10):
        curr_filter = cv2.bilateralFilter(frame,9,75,75)     
    elif(fps*10 < curr_frame <= fps*11):
        curr_filter = cv2.bilateralFilter(frame,15,75,75)  
    elif(fps*11 < curr_frame <= fps*12):
        curr_filter = cv2.bilateralFilter(frame,21,75,75)     
        
     """

    
# =============================================================================
    #curr_filter = cv2.bitwise_not(frame)
    
    
#     portrait_mode = apply_portrait_mode(frame)
#     cv2.imshow('portrait_modeS', portrait_mode)
# 
#     circle_blur = apply_circle_focus_blur(frame)
#     cv2.imshow('circle_blur', circle_blur)
# 
#     sepia = apply_sepia(frame.copy())
#     cv2.imshow('sepia', sepia)
# 
#     redish_color = apply_color_overlay(frame.copy(), intensity=.5, red=230, blue=10)
#     cv2.imshow('redish_color', redish_color)
#
#     invert = apply_invert(frame)
#     cv2.imshow('invert', invert)
# =============================================================================

    cv2.imshow('filter', curr_filter)
    
    curr_frame += 1
    
    # output video
    out.write(curr_filter) # VERANDEREN naar frame met effectjes
    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
