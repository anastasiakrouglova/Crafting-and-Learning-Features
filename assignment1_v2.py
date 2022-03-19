#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:33:12 2022

@author: nastysushi
"""


import numpy as np
import cv2 
from lib.videoConf import CFEVideoConf

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('assets/footage.mp4')

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


 # out = cv2.VideoWriter('grey.avi',fourcc, 30.0, (800,600), isColor=False)

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
    """
    Turns video to grayscale. Convertion back to BGR to make grayscale
    export possible.

    """
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return mask



def gaussianBlur():
    """
    Gaussian filter removes high-frequency components 
    from the image, images become more smooth.
    
    """
    val = round((curr_frame-200)/10, 1)
    mask = cv2.GaussianBlur(frame, (9,9), val)
    #print(val)
    return mask


def bilateralBlur(x):
    """
    Bilateral filter does not avarege across edges.
    Weighted by spatial distance and intensity difference.
    
    """
    mask = cv2.bilateralFilter(frame,x,75,75)
    return mask    

 
def grabObjectHSV(morphOp, spectrum):
    """
    Grabs an object in RGB and HSV color space. 
    Show binary frames with the foreground object 
    in white and background in black.

    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   

    
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    
       
    """
    MORPHOLOGICAL OPERATIONS
    
    ----
    
    Erosion: 
        It is useful for removing small white noises.
        Used to detach two connected objects etc.
        
    Dilation:
        In cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won’t come back, but our object area increases.
        It is also useful in joining broken parts of an object.
        
    Closing:
        A dilation followed by an erosion 
        (i.e. the reverse of the operations for an opening). 
        closing tends to close gaps in the image.
    """
    if(morphOp == 'erosion'):
        kernel = np.ones((10,10), np.uint8)
        morph_op = cv2.erode(hsv_frame, kernel, iterations=1) 
    elif(morphOp == 'dilation'):
        
        kernel = np.ones((10,10), np.uint8)
        morph_op = cv2.dilate(hsv_frame, kernel, iterations=3)
        
        
    elif(morphOp == 'closing'):
        kernel = np.ones((30,30), np.uint8)
        morph_op = cv2.morphologyEx(hsv_frame, cv2.MORPH_CLOSE, kernel)
    elif(morphOp == 'opening'):
        kernel = np.ones((30,30), np.uint8)
        morph_op = cv2.morphologyEx(hsv_frame, cv2.MORPH_OPEN, kernel)

    
    
    mask = cv2.inRange(morph_op, lower_yellow, upper_yellow)
    # coversion to make export to video possible
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
  # if(spectrum == 'hsv'):
  #     return hsv_frame
    return mask


def sobel(detection):
    """
    Sobel filter is a 1D discrete derivative filter for edge detection.
    Change in color intensity to detect edges by taking first derivative.

    """

	# Convert to graycsale
    img_gray = grayscale()

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

    
    # sobelx - base = not blurred img
    if(detection == 'sobelx_noblur'):
        return cv2.Sobel(frame, cv2.CV_8U, 1,0, ksize=3)
    elif(detection == 'sobelx'):
        return cv2.Sobel(img_blur, cv2.CV_8U, 1,0, ksize=3)
    elif(detection == 'sobely'):
        return cv2.Sobel(img_blur, cv2.CV_8U, 0,1, ksize=3)
    elif(detection == 'sobelxy_noblur'):
        return cv2.Sobel(frame, cv2.CV_8U, dx=1, dy=1, ksize=5)
    elif(detection == 'sobelxy'):
        return cv2.Sobel(img_blur, cv2.CV_8U, dx=1, dy=1, ksize=5)
    else:
        print('forgot to add a detection parameter')


def houghTransform(dp, mindst):
    """
    Hough transform is a feature extraction method for detecting simple 
    shapes such as circles, lines etc in an image.

    """
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Finite difference filters respond strongly to noise, so
    # Smoothing edges, by forcing pixels different
    # from their nieghbors to look more like neighbors, helps forecome the problem
    img = cv2.medianBlur(gray, 5)
    
    # convert gray back to BGR
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, mindst, param1=100, param2=30,minRadius=0, maxRadius=0) 
    
    
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        #draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2],(0,255,0), 3)
        #draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0,0,255), 5)
        
    return cimg

    
def videoTests(fps, curr_frame):
    # TODO: visualise detected edges
    # 10s
   # curr_filter = houghTransform(1, 120)
    
        
    if(curr_frame <= fps*2):
        curr_filter = houghTransform(1, 120)
    elif(fps*2 < curr_frame <= fps*4):
        curr_filter = houghTransform(1, 50)
        
    elif(fps*4 < curr_frame <= fps*6):
         curr_filter = houghTransform(1, 20)
    elif(fps*6 < curr_frame <= fps*8):
        curr_filter = houghTransform(2, 120)
    elif(fps*8 < curr_frame <= fps*10):
         curr_filter = houghTransform(3, 120)
    else:   
        curr_filter = frame

    
    return curr_filter


def objectDetection(minTreshold):
    """
    ...
     
    """
    # look at a smaller piece of video
    #belt = frame[102:342, 134:549]
    
    gray = grayscale()
    
    _, threshold = cv2.threshold(gray, minTreshold, 255, cv2.THRESH_BINARY)
    
    # Detect object
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        rectangle = cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 3)
        
    return frame


def objectDetectionTimed():
    # variable to change threshold
    #minThreshold = 88
    #FIXEN!!!! ZIE GRAYSCALE
    
    gray = grayscale()
    
    #if(curr_frame < 100):
       # return objectDetection(minThreshold)

    #elif(100 <= curr_frame < 250):
    # ideaal: onder -> zie je stuk van draaiding, erboven stuk van eend weggaat
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #_, threshold = cv2.threshold(gray, minThreshold, 255, cv2.THRESH_BINARY)

    return gray
    




    
def videoPartOne(fps, curr_frame):
    ## color to grayscale several times (0-4s) ------------------------------------------
    if(fps < curr_frame <= fps*2 or fps*3 < curr_frame <= fps*4):
        curr_filter = grayscale()
        labeled = cv2.putText(img=curr_filter, text='grayscale', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*2 < curr_frame <= fps*3):
        curr_filter = frame
        labeled = cv2.putText(img=curr_filter, text='normal frame', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    #start: 0, 200, 800   
    
    ## Blur (4-12s) ------------------------------------------
    # gaussian blur (frame 200-400)
    elif(fps*4 < curr_frame <= fps*8):
        val = round((curr_frame-200)/10, 1)
        curr_filter = cv2.GaussianBlur(frame, (9,9), val)
        labeled = cv2.putText(img=curr_filter, text='Gaussian blur', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    

    # Bi-lateral filter 
        #curr_frame: start: 400 - 800
        # highly effective in noise removal while keeping edges sharp. But the operation is slower compared to other filters.
        # Mus be an integer between 2 and 36
    elif(fps*8 < curr_frame <= fps*9):
        curr_filter = bilateralBlur(2)
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (2)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*9 < curr_frame <= fps*10):
        curr_filter = bilateralBlur(15)   
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (15)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    

    elif(fps*10 < curr_frame <= fps*11):
        curr_filter = bilateralBlur(21)
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (21)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*11 < curr_frame <= fps*12):
        curr_filter = bilateralBlur(31) # Time consuming
        labeled = cv2.putText(img=curr_filter, text='BilateralBlur (31)', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
               
    # Black and white ------------------------------------------
    # TODO: CHANGES IN ANOTHER COLOR
    elif(fps*12 < curr_frame <= fps*14): # 0-2s
        curr_filter = grabObjectHSV('dilation', 'binary')    
        labeled = cv2.putText(img=curr_filter, text='Dilation', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*14 < curr_frame <= fps*16): # 2-4s
        curr_filter = grabObjectHSV('erosion', 'binary')
        labeled = cv2.putText(img=curr_filter, text='Erosion', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*16 < curr_frame <= fps*18):
        curr_filter = grabObjectHSV('closing', 'binary')
        labeled = cv2.putText(img=curr_filter, text='Closing', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
    elif(fps*18 < curr_frame <= fps*20):
        curr_filter = grabObjectHSV('opening', 'binary')
        labeled = cv2.putText(img=curr_filter, text='Opening', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
        
    ## DEFAULT
    else:
        curr_filter = frame
        labeled = cv2.putText(img=curr_filter, text='normal frame', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
        
    return labeled



def videoPartTwo(offset, fps, curr_frame):
    nfps = fps + offset
    
    if(curr_frame <= nfps):
        curr_filter = sobel('sobelx_noblur')
        
    elif(nfps < curr_frame <= nfps*2):
        curr_filter = sobel('sobelx')
    elif(nfps*2 < curr_frame <= nfps*3):
        curr_filter = sobel('sobely')
    elif(nfps*3 < curr_frame <= nfps*4):
        curr_filter = sobel('sobelxy')
    elif(nfps*4 < curr_frame <= nfps*5):
        curr_filter = sobel('sobelxy_noblur')    
    else:
        curr_filter = frame
    
    # TODO: return label
    
    return curr_filter
        
        
    




# =============================================================================
#     if(curr_frame <= fps*2): # 0-2s
#         curr_filter = grabObjectHSV('dilation', 'binary')    
#         labeled = cv2.putText(img=curr_filter, text='Dilation', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
#     elif(fps*2 < curr_frame <= fps*4): # 2-4s
#         curr_filter = grabObjectHSV('erosion', 'binary')
#         labeled = cv2.putText(img=curr_filter, text='Erosion', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
#     elif(fps*4 < curr_frame <= fps*6):
#         curr_filter = grabObjectHSV('closing', 'binary')
#         labeled = cv2.putText(img=curr_filter, text='Closing', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
#     elif(fps*6 < curr_frame <= fps*8):
#         curr_filter = grabObjectHSV('opening', 'binary')
#         labeled = cv2.putText(img=curr_filter, text='Opening', org=(150, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),thickness=3)    
#     return labeled
#
# =============================================================================
    
    
# =============================================================================
#     if(curr_frame <= fps):
#         curr_filter = sobel('sobelx_noblur')
#         #curr_filter = cv2.drawContours(curr_filter, countours, -1, (0, 255, 0), 2)
#     elif(fps < curr_frame <= fps*2):
#         curr_filter = sobel('sobelx')
#     elif(fps*2 < curr_frame <= fps*3):       
#         curr_filter = sobel('sobely')
#     elif(fps*3 < curr_frame <= fps*4):       
#         curr_filter = sobel('sobelxy')
#     elif(fps*4 < curr_frame <= fps*5):
#         curr_filter = sobel('sobelxy_noblur')
# 
#     else:
#         curr_filter = frame
#     
#     return curr_filter
# =============================================================================
# =============================================================================
        
        
# =============================================================================
#         # TODO: CHANGES IN ANOTHER COLOR
#         if(curr_frame <= fps): # 0-2s
#             curr_filter = grabObjectHSV('dilation', 'hsv')
#         elif(fps < curr_frame <= fps*2): # 0-2s
#             curr_filter = grabObjectHSV('dilation', 'binary')
#         elif(fps*2 < curr_frame <= fps*3): # 2-4s
#         
#             curr_filter = grabObjectHSV('erosion', 'hsv')
#         elif(fps*3 < curr_frame <= fps*4): # 2-4s
#             curr_filter = grabObjectHSV('erosion', 'binary')
#             
#         elif(fps*4 < curr_frame <= fps*5):
#             curr_filter = grabObjectHSV('closing', 'hsv')
#         elif(fps*5 < curr_frame <= fps*6):
#             curr_filter = grabObjectHSV('closing', 'binary')
#         
#         elif(fps*6 < curr_frame <= fps*7):
#             curr_filter = grabObjectHSV('opening', 'hsv')
#         elif(fps*7 < curr_frame <= fps*8):
#             curr_filter = grabObjectHSV('opening', 'binary')
#         else:
#             curr_filter = frame
# =============================================================================
        
    


def video(fps, curr_frame):
    
    #vid = videoPartOne(fps, curr_frame)
    #vid = videoPartTwo(20, fps, curr_frame)
    
 
    vid = videoTests(fps, curr_frame)
    
    
    return vid

     
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(curr_frame)


    #curr_filter = frame
    
    ############### BASIC IMAGE PROCESSING ######################
    #curr_filter = grayscale()
    #curr_filter = gaussianBlur()
    #curr_filter = bilateralBlur(2) # TODO: every second, +7 waarde
    #curr_filter = grabObjectHSV() # mask RGB_HSV  # TODO: Finetuning parameters
    
    ############### OBJECT DETECTION ######################
    
    #curr_filter = sobel()     # sobel edge detection (5s)
    #curr_filter = houghTransform()
    #curr_filter = objectDetectionTimed()
    
    
    ############### FREEDOM ######################
    
    # Effect: Sepia ofzo    
    # Effect: portrait mode (achtergrond vaag, object scherp)
    # Afbeelding laten verdwijnen
    
    
    
    
    
    curr_filter = video(fps, curr_frame)
    
    #curr_filter = bilateralBlur(31)
    
    

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

    # Als meedere filters niet gaan werken; if-statement met im-show!
    # dus if(.. seconden ): imshow(filter1), else imshow(filter2)
    cv2.imshow('filter', curr_filter)
    
    curr_frame += 1
    
   
    
    # output video
    out.write(curr_filter) 
    

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
