# https://en.wikipedia.org/wiki/Kalman_filter

import numpy as np
import cv2
import random

class Ant:
    def __init__(self, pixels):
        self.pixels = pixels

# Returns the pixels filled
def flood_fill(mask, seed_x, seed_y):
    assert(isinstance(mask, np.ndarray))
    assert(mask[seed_y, seed_x] == 255)
    height, width = mask.shape

    filled_pixels = set()
    queue = [(seed_x, seed_y)]

    while queue:
        cur_pixel = queue.pop()
        filled_pixels.add(cur_pixel)
        
        try:
            if (cur_pixel[0], cur_pixel[1] + 1) not in filled_pixels and mask[cur_pixel[1] +  1, cur_pixel[0]] == 255:
                queue.append((cur_pixel[0], cur_pixel[1] + 1))
        except IndexError:
            pass

        try:
            if (cur_pixel[0], cur_pixel[1] - 1) not in filled_pixels and mask[cur_pixel[1] - 1, cur_pixel[0]] == 255:
                queue.append((cur_pixel[0], cur_pixel[1] - 1))
        except IndexError:
            pass

        try:
            if (cur_pixel[0] + 1, cur_pixel[1]) not in filled_pixels and mask[cur_pixel[1], cur_pixel[0] + 1] == 255:
                queue.append((cur_pixel[0] + 1, cur_pixel[1]))
        except IndexError:
            pass

        try:
            if (cur_pixel[0] - 1, cur_pixel[1]) not in filled_pixels and mask[cur_pixel[1], cur_pixel[0] - 1] == 255:
                queue.append((cur_pixel[0] - 1, cur_pixel[1]))
        except IndexError:
            pass
    
    return filled_pixels


def find_ants(mask):
    assert(np.isscalar(mask[0, 0])) # Only accept one channel mask

    pixel_to_ant = dict()
    ants = list()

    for y in range(len(mask)):
        for x in range(len(mask[y])):
            if (x, y) not in pixel_to_ant and mask[y, x] == 255:
                ant_pixels = flood_fill(mask, x, y)
                # See if it's actually an ant here somehow. (Probably just size it up)
                new_ant = Ant(ant_pixels)
                ants.append(new_ant)
                for pixel in ant_pixels:
                    pixel_to_ant[pixel] = new_ant

    return ants


cap = cv2.VideoCapture('../video1.mp4')  # Load capture video
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()  # Create a background subtractor.

# Skip first 200 frames to build up history for the background subtractor
for _ in range(200):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    fgbg.apply(frame)
    

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
    ants = find_ants(fgmask[:,:,0])
    for ant in ants:
        random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for pixel in ant.pixels:
            fgmask[pixel[1], pixel[0]] =  random_color
    print(len(ants))
    # cv2.drawContours(fgmask, contours, -1, (0,0,255), 2)

    edges = cv2.Canny(frame,100,200)  # Use canny edge detection
    cv2.imshow('frame', frame)  # Draw the frame image
    cv2.imshow('edges', edges)  # Draw the canny edges
    cv2.imshow('original', original_frame)  # Draw the original frame image
    cv2.imshow('fgmask', fgmask)  # Draw the foreground mask


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()