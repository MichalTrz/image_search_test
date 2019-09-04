from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# define variable for resize tratio
ratio = 1

def for_point_warp(cnt, orig):
    # we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = cnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    # summing the (x, y) coordinates together by specifying axis=1
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # Notice how our points are now stored in an imposed order: 
    # top-left, top-right, bottom-right, and bottom-left. 
    # Keeping a consistent order is important when we apply our perspective transformation
    print(rect)
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    return warp

def resize(img, width=None, height=None, interpolation = cv2.INTER_AREA):
    #global ratio
    w, h, _ = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height/h
        width = int(w*ratio)
        print(width)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width/w
        height = int(h*ratio)
        print(height)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    #global refPt, cropping
    global coords
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        print('x: {}, y: {}'.format(x,y))
        if len(coords) > 3:
            print('i have 4 coords.')
            #cv2.drawContours(flat_object_resized_copy, [coords], -1, (0,255,0), 3)
            warped = for_point_warp(np.array(coords), flat_object_resized)
            cv2.imshow("Warped ROI", warped)
            cv2.imwrite(outname,warped)
            coords = []

coords = []

#print ('Number of arguments:', len(sys.argv), 'arguments.')
#print ('Argument List:', str(sys.argv))

if len(sys.argv) != 2:
    print('Use as: python warp.py image_filename')
    exit()

fname = sys.argv[1]
name, ext = fname.split('.')
outname = name + '_crop.' + ext

flat_object = cv2.imread(fname)
#resize the image
flat_object_resized = resize(flat_object, height=700)
#make a copy
flat_object_resized_copy = flat_object_resized.copy()

np.shape(flat_object_resized_copy)
# draw a contour
our_cnt = []

cv2.imshow("orig image", flat_object_resized)
cv2.namedWindow("orig image")
cv2.setMouseCallback("orig image", click_and_crop)
print("Select the four corners of a rectangular object on the picture.")

while True:
	# wait for a keypress
    key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
 
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

cv2.waitKey()
cv2.destroyAllWindows()