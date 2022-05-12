# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:34:43 2022

@author: Dell
"""

import numpy as np
import math
import cv2
#%%
def Hough(image):
    # Theta 0 - 180 degree
    theta = np.arange(0, 180, 1)
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))

    # initialize an H matrix 
    rho_range = round(math.sqrt(image.shape[0]**2 + image.shape[1]**2))  # Max value of rho is the diagonal
    H = np.zeros((2 * rho_range, len(theta)), dtype=np.uint8)

    # Get edge pixel location (x,y)
    edge_pixels = np.where(image == 255)
    coord = list(zip(edge_pixels[0], edge_pixels[1])) # 

    # Calculate rho value for each image location (x,y) with all the theta range
    for p in range(len(coord)):
        for t in range(len(theta)):
            rho = int(round(coord[p][1] * cos[t] + coord[p][0] * sin[t]))
            H[rho, t] += 1 

    return H , theta , rho

def Hough_peaks(H, numpeaks):
    edge_pixels = np.where(H >80)
    
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))
    #coordinates.sort()
    #len_coordinates = len(coordinates)
    #selected = coordinates[len_coordinates-numpeaks: len_coordinates]
    selected = coordinates[0:numpeaks]
    
    return selected

def Houghlines(img,peaks):
    #H,th,roh = Hough(edges)
    #peaks = Hough_peaks(H,numpeaks)
    # Draw detected line on an original image
    for i in range(0, len(peaks)):
        a = np.cos(np.deg2rad(peaks[i][1]))
        b = np.sin(np.deg2rad(peaks[i][1]))
        x0 = a*peaks[i][0]
        y0 = b*peaks[i][0]
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
       
        p1 = (x1,y1)
        p2 = (x2,y2)
        
        x = cv2.line(img,p1,p2,(0,255,0),2)
        
    return x
        



#%%

image = cv2.imread("WhatsApp Image 2022-04-12 at 12.11.01 AM.JPEG")
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grayscale,50,150)
# We could also try sobel
#img_blur = cv2.GaussianBlur(grayscale, (3,3), 0)
#edges = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)

H, theta, rho = Hough(edges)
cv2.imshow("H", H)
H_BGR = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)
numpeaks = 40
coordinates = Hough_peaks(H,numpeaks)
for i in range (len(coordinates)): 
    
    x = cv2.circle(H_BGR,(coordinates[i][1], coordinates[i][0]), radius = 2, color = (0,0,255), thickness = -1)
y = Houghlines(image, coordinates)    



cv2.imshow("out", x)
cv2.imshow("o", y)

cv2.waitKey()
cv2.destroyAllWindows()
    


    