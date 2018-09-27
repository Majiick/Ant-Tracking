# https://en.wikipedia.org/wiki/Kalman_filter

import numpy as np
import cv2

def flood_fill(mask, seed_x, seed_y):
    pass


cap = cv2.VideoCapture('../video3.mp4')  # Load capture video
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()  # Create a background subtractor.

while(cap.isOpened()):  # While video is opened
    ret, frame = cap.read()  # Ret is just status, frame is the actual image.
    original_frame = frame.copy()  # Save the original frame so we can use it for reference later on.

    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Blur the image using a gaussian blur with a 5x5 kernel.
    fgmask = fgbg.apply(frame)  # Apply the frame to the background subtractor. fgmask is the result of subtracting the background.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert image from RGB to HSV.
    h, s, v = cv2.split(hsv)  # Get the H, S, and V channels.
    h.fill(255)  # Set all hue values to 255
    s.fill(255)  # Set all saturation values to 255
    # Setting saturation and hue to 255 means that we only see the value.
    # v.fill(255)  
    ret,v = cv2.threshold(v,127,255,cv2.THRESH_BINARY)  # For any pixel with value of more than 127, turn that value into 255. Any pixel with value less than 255 set to 0.
    frame = cv2.merge([h, s, v])  # Merge HSV into one image.
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)  # Convert HSV to BGR

    # grey = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(fgmask, contours, -1, (0,0,255), 2)

    edges = cv2.Canny(frame,100,200)  # Use canny edge detection
    cv2.imshow('frame', frame)  # Draw the frame image
    cv2.imshow('edges', edges)  # Draw the canny edges
    cv2.imshow('original', original_frame)  # Draw the original frame image
    cv2.imshow('fgmask', fgmask)  # Draw the foreground mask

    '''
    for y in range(len(fgmask)):
        for x in range(len(fgmask[y])):
            if fgmask[y, x] == 255:
                flood_fill(fgmask, x, y)
    '''


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()