import numpy as np
import cv2
import random
import math
import sys


class Ant:
    id = 0

    def __init__(self, pixels):
        self.id = Ant.id
        Ant.id+=1
        self.update(pixels)

    def update(self, pixels):
        self.pixels = pixels
        self.size = len(pixels)

        self.minX = sys.maxint
        self.maxX = 0
        self.minY = sys.maxint
        self.maxY = 0
        for pixel in pixels:
            if pixel[0] > self.maxX:
                self.maxX = pixel[0]
            elif pixel[0] < self.minX:
                self.minX = pixel[0]

            if pixel[1] > self.maxY:
                self.maxY = pixel[1]
            elif pixel[1] < self.minY:
                self.minY = pixel[1]

        self.centre = self.minX + ((self.maxX - self.minX) / 2), self.minY + ((self.maxY - self.minY) / 2)

    def search_and_update(self, mask):
        nearest = find_nearest_white(mask, self.centre)
        # Check here that the nearest pixel isn't too far away (e.g. the size of the ant).
        pixels = flood_fill(fgmask[:,:,0], nearest[0], nearest[1])
        self.update(pixels)

    def __str__(self):
        return "{}, width: {}, height: {}, size: {}".format(self.centre, self.maxX-self.minX, self.maxY-self.minY, self.size)


# https://stackoverflow.com/a/45225986
def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index][0]


def find_ants(mask):
    used_pixels = set()
    ants = list()

    for y in range(len(mask)):
        for x in range(len(mask[y])):
            if (x, y) not in used_pixels and mask[y, x] == 255:
                ant_pixels = flood_fill(mask, x, y)
                # See if it's actually an ant here somehow. (Probably just size it up)
                if len(ant_pixels) > 15:
                    new_ant = Ant(ant_pixels)
                    ants.append(new_ant)
                    used_pixels.update(ant_pixels)

    return ants


def flood_fill(mask, seed_x, seed_y):
    filled_pixels = set()
    queue = [(seed_x, seed_y)]

    while queue:
        cur_pixel = queue.pop()
        filled_pixels.add(cur_pixel)

        try:
            if (cur_pixel[0], cur_pixel[1] + 1) not in filled_pixels and mask[cur_pixel[1] + 1, cur_pixel[0]] == 255:
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


def printCoords(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print x, y


if __name__ == "__main__":
    cap = cv2.VideoCapture('../video1.mp4')  # Load capture video.
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()  # Create a background subtractor.

    cv2.namedWindow("fgmask")
    cv2.setMouseCallback("fgmask", printCoords)

    # Skip first 200 frames to build up history for the background subtractor.
    for _ in range(200):
        _, frame = cap.read()
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        bgSubtractor.apply(frame)

    firstFrame = True
    ants = list()
    blobs = dict()
    while cap.isOpened():  # While video is opened
        _, frame = cap.read()
        original_frame = frame.copy()  # Save the original frame so we can use it for reference later on.

        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Blur the image using a gaussian blur with a 5x5 kernel.
        fgmask = bgSubtractor.apply(frame)  # Apply the frame to the background subtractor. fgmask is the result of subtracting the background.
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        if firstFrame:
            firstFrame = False
            ants = find_ants(fgmask[:,:,0])
        else:
            for ant in ants:
                ant.search_and_update(fgmask[:,:,0])

        for ant in ants:
            for other in ants:
                if ant != other and not ant.pixels.isdisjoint(other.pixels):
                    print "{} and {} are in a blob".format(ant.id, other.id)

        for ant in ants:
            cv2.rectangle(img=fgmask, pt1=(ant.minX, ant.minY), pt2=(ant.maxX, ant.maxY), color=(0,0,255), thickness=1)
            cv2.putText(img=fgmask, text=str(ant.id), org=ant.centre, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,0), thickness=2)
            cv2.putText(img=fgmask, text=str(ant.id), org=ant.centre, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,255,255), thickness=1)


        # newAnts = find_ants(fgmask[:, :, 0])
        # for ant in _ants:
        #     # random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        #     ant.adjustPos(newAnts)
        #     for pixel in ant.pixels:
        #         fgmask[pixel[1], pixel[0]] = ant.color

        cv2.imshow('original', original_frame)  # Draw the original frame image
        cv2.imshow('fgmask', fgmask)  # Draw the foreground mask

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


