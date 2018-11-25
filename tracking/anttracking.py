# Ant tracking.
#
# Authors:
#   Padraig Redmond
#   Zan Smirnov
#   Evgeny Timoshin
#
# TODO: stuff we could talk about:
# Flood fill
# background extraction with running average
# arctan directional stuff
#
# ### Algorithm ###
# Pre-processing:
# We use a MOG background subtractor to isolate the ants from the background. The first 200 frames are used to build up
# the history, and a Gaussian blur is also applied to each frame before being processed by the subtractor.
#
# For each frame:
# 1. For every ant and blob, search for their new location and update.
#       If their size falls below a certain threshold they are no longer tracked.
# 2. Check if any blobs have merged, if so combine them.
# 3. Check for and new ants.
#       If these are near a blob, then we assume they came from it and update both as such.
#       Otherwise they are a new ant and are added to the list of tracked ants.
# 4. Check each ant to see if they have joined an existing blob, or formed a new one.
# 5. Now that everything has been updated for this frame, draw bounding boxes, id numbers and paths for each ant.
#
# ### UI controls ###
# The user can control the output by either clicking on ants or using the provided control buttons.
#
# The window containing the control buttons was created using PyQt4. It allows the user to toggle between playing and
# pausing the video output. They can also toggle the display of the bounding boxes and paths for all ants.
#
# Clicking on an ant will isolate that ant. It will be displayed on its own over the extracted background image.

from __future__ import print_function
import numpy as np
import cv2
import sys
import copy
import math
import bisect
import random
from PyQt4 import QtCore, QtGui
from ui import Ui_MainWindow
import easygui


MIN_ANT_SIZE = 55
MAX_FIND_NEAREST_DISTANCE = 50

app = QtGui.QApplication(sys.argv)
window = QtGui.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window)

play = True
show_boxes = True
show_lines = True
original_image = True
close = False
paths = True
boxes = True


def play_pressed():
    global play
    play = True


def pause_pressed():
    global play
    play = False


def exit_pressed():
    global close
    close = True


def toggle_paths_pressed():
    global paths
    paths = not paths


def toggle_box_pressed():
    global boxes
    boxes = not boxes


ui.playButton.clicked.connect(lambda: play_pressed())
ui.pauseButton.clicked.connect(lambda: pause_pressed())
ui.exitButton.clicked.connect(lambda: exit_pressed())
ui.togglePathsButton.clicked.connect(lambda: toggle_paths_pressed())
ui.toggleBoxesButton.clicked.connect(lambda: toggle_box_pressed())
window.show()

tracked_ant = None


# Stores relevant information about the ant and contains method definitions for updating the ants tracking.
class Ant:
    id = 0

    def __init__(self, pixels):
        self.id = Ant.id
        Ant.id+=1
        self.prev_sizes = list()
        self.prev_centres = list()
        self.update(pixels)
        self.color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))

    def update(self, pixels):
        self.pixels = pixels
        self.size = len(pixels)
        bisect.insort(self.prev_sizes, self.size)

        self.minX = sys.maxint
        self.maxX = 0
        self.minY = sys.maxint
        self.maxY = 0
        for pixel in pixels:
            if pixel[0] > self.maxX:
                self.maxX = pixel[0]
            if pixel[0] < self.minX:
                self.minX = pixel[0]
            if pixel[1] > self.maxY:
                self.maxY = pixel[1]
            if pixel[1] < self.minY:
                self.minY = pixel[1]

        self.centre = self.minX + ((self.maxX - self.minX) / 2), self.minY + ((self.maxY - self.minY) / 2)
        self.prev_centres.append(self.centre)

    def search_and_update(self, mask):
        if len(self.pixels) < MIN_ANT_SIZE:
            return True

        nearest = find_nearest_white(mask, self.centre)
        distance = abs(self.centre[0] - nearest[0]) + abs(self.centre[1] - nearest[1])
        if distance > MAX_FIND_NEAREST_DISTANCE:
            return True

        pixels = flood_fill(mask, nearest[0], nearest[1])
        if len(pixels) < MIN_ANT_SIZE:
            return True

        self.update(pixels)

    def direction(self):
        if len(self.prev_centres) < 2:

            return 0

        return direction(self.prev_centres[-2], self.prev_centres[-1])

    def median_size(self):
        return np.median(self.prev_sizes)

    def overlaps(self, other):
        return not self.pixels.isdisjoint(other.pixels)

    def draw(self, img, box_color=(0, 0, 255)):
        if boxes:
            if self == tracked_ant:
                box_color = (255, 0, 0)
            cv2.rectangle(img=img, pt1=(self.minX, self.minY), pt2=(self.maxX, self.maxY), color=box_color, thickness=1)
        draw_text(img, str(self.id), self.centre, 1)

        if paths:
            if len(self.prev_centres) > 1:
                prev_pos = self.prev_centres[0]
                for lp in self.prev_centres:
                    cv2.line(img,prev_pos,lp,self.color,1)
                    prev_pos = lp

    def __str__(self):
        return "{}: {}, width: {}, height: {}, size: {}".format(self.id, self.centre, self.maxX-self.minX, self.maxY-self.minY, self.size)

    def getSize(self):
        return self.size


# Stores information relevant for tracking ant blob, and contains method definition for tracking blob, updating and
# dealing with separation.
class AntBlob:
    id = 0

    def __init__(self, ant):
        self.id = AntBlob.id
        AntBlob.id += 1
        self.prime = copy.deepcopy(ant)
        self.prime.id = self.id
        self.ants = dict()
        self.add(ant)

    def add(self, ant):
        self.ants[ant.id] = ant

    def exit(self, leaver_ant):
        leaver_direction = direction(self.prime.prev_centres[-2], leaver_ant.centre)
        leaver_size = leaver_ant.size

        best_weight = sys.maxint
        best_key = -1
        for ant_key in self.ants:
            angle_difference = difference_in_direction(self.ants[ant_key].direction(), leaver_direction)
            size_difference = math.fabs((float(leaver_size) - self.ants[ant_key].median_size()) / leaver_size)
            current_weight = size_difference + angle_difference
            if current_weight < best_weight:
                best_weight = current_weight
                best_key = ant_key

        candidate = self.ants[best_key]
        del self.ants[best_key]
        return candidate

    def remove_last_ant(self):
        assert(len(self.ants) == 1)
        id = self.ants.iterkeys().next()
        final_ant = self.ants[id]
        del self.ants[id]
        return final_ant

    def merge(self, other):
        for ant in other.ants.itervalues():
            self.add(ant)

    def search_and_update(self, mask):
        print(len(self.ants))
        all_black = True
        for p in self.prime.pixels:
            if mask[p[1], p[0]] == 255:
                all_black = False

        if all_black:
            return True

        self.prime.search_and_update(mask)

    def overlaps(self, other):
        return self.prime.overlaps(other.prime)

    def draw(self, img):
        self.prime.draw(img, box_color=(0,255,255))

    def __str__(self):
        return str(self.prime) + ", ants: {" + ", ".join(str(ant.id) for ant in self.ants.values()) + "}"


# Returns the distance between 2 points.
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )


# Returns the direction between two points in radians.
def direction(from_, to):
    return math.atan2(to[1] - from_[1], to[0] - from_[0])


# Returns the positive value of the smaller difference between two angles in radians.
def difference_in_direction(d1, d2):
    diff = math.fabs(d1 - d2)
    if diff > math.pi:
        diff -= math.pi * 2
    return math.fabs(diff)


# Finds the nearest white pixel to a given point.
# https://stackoverflow.com/a/45225986
def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)

    return nonzero[nearest_index][0]


# Takes a list of ants and blobs, and finds any new pixels that don't belong to either.
def find_new_ants(mask, ants, antBlobs):
    used_pixels = set()
    for ant in ants:
        used_pixels.update(ant.pixels)
    for blob in antBlobs:
        used_pixels.update(blob.prime.pixels)
    new_ants = list()

    for y in range(len(mask)):
        for x in range(len(mask[y])):
            if (x, y) not in used_pixels and mask[y, x] == 255:
                ant_pixels = flood_fill(mask, x, y)
                if len(ant_pixels) > MIN_ANT_SIZE:
                    new_ant = Ant(ant_pixels)
                    new_ants.append(new_ant)
                    used_pixels.update(ant_pixels)
    return new_ants


# Flood fill algorithm which returns all white pixels connected to the seed point.
# Used for detecting the ants from frame to frame.
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


# Draws white text with a black border to improve readability.
def draw_text(img, text, org, fontScale):
    cv2.putText(img=img, text=str(text), org=org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,
                color=(0, 0, 0), thickness=2)
    cv2.putText(img=img, text=str(text), org=org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,
                color=(255, 255, 255), thickness=1)


# Prints the coordinates of the mouse on click.
def select_ant(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print ('x {} y {}'.format(x, y))

        found = False
        for ant in ants:
            if x > ant.minX and x < ant.maxX and y > ant.minY and y < ant.maxY:
                found = True
                global tracked_ant
                tracked_ant = ant
                break

        if not found:
            tracked_ant = None


f = easygui.fileopenbox()
cap = cv2.VideoCapture(f)
bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()  # Create a background subtractor.
cv2.namedWindow("Ants")
cv2.setMouseCallback("Ants", select_ant)

background_img = None

# Skip first 200 frames to build up history for the background subtractor.
for _ in range(200):
    ret, frame = cap.read()
    if background_img is not None:
        background_img = background_img.copy() + frame.copy()

    if background_img is None:
        background_img = frame.copy()
        background_img = np.array(background_img, dtype=np.uint32)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    bgSubtractor.apply(frame)

background_img = background_img * (1.0 / 200.0)
background_img = np.array(background_img, dtype=np.uint8)

# Initialize the list ants and blobs will be stored in.
ants = list()
ant_blobs = list()

# While frames are streaming from the video.
while cap.isOpened():
    if close:
        break
    if not play:
        QtCore.QCoreApplication.processEvents()
        continue
    
    _, frame = cap.read()
    frame_original = frame.copy()
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("***FRAME {}***".format(frame_num))
    original_frame = frame.copy()

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = bgSubtractor.apply(frame)
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

    # Update the positions of every ant and blob.
    # If a blob or ant shrinks below a certain size they are no longer tracked.
    ants_to_remove = []
    for ant in ants:
        if ant.search_and_update(fgmask[:,:,0]):
            ants_to_remove.append(ant)
    for ant in ants_to_remove:
        ants.remove(ant)

    for ant in ants_to_remove:
        for p in ant.pixels:
            fgmask[p[1], p[0], :] = 0

    blobs_to_remove = []
    for blob in ant_blobs:
        if blob.search_and_update(fgmask[:,:,0]):
            blobs_to_remove.append(blob)

    for blob in blobs_to_remove:
        ant_blobs.remove(blob)

    # Check for any blob overlaps and merge if so.
    for blob in ant_blobs:
        for other in ant_blobs:
            if blob != other and blob.overlaps(other):
                print("Merging blobs: {} and {}".format(blob, other))
                blob.merge(other)
                ant_blobs.remove(other)

    # Check for un-tracked ants.
    # If there is a blob within range of an un-tracked ant, assume it came from that, otherwise it is a new ant.
    new_ants = find_new_ants(fgmask[:,:,0], ants, ant_blobs)
    print("*****\nUn-tracked ants:\n")
    for new_ant in new_ants:
        from_blob = False
        print("\t{}".format(new_ant))
        for blob in ant_blobs:
            if dist(new_ant.centre, blob.prime.centre) < 50:
                from_blob = True
                print("Possible blob exit: " + str(new_ant) + " from " + str(blob))
                ant_from_blob = blob.exit(new_ant)
                if ant_from_blob is None:
                    # The blob had no ants to give.
                    ants.append(new_ant)
                else:
                    # Ant taken from blob.
                    ant_from_blob.update(new_ant.pixels)
                    ants.append(ant_from_blob)
                if len(blob.ants) == 1:
                    # actually look at an ants parameters when determining which ant should be pulled from the blob.
                    final_ant = blob.remove_last_ant()
                    final_ant.update(blob.prime.pixels)
                    ants.append(final_ant)
                    ant_blobs.remove(blob)
        if not from_blob:
            ants.append(new_ant)
    print("*****")

    # Check each ant to see if they have entered or formed a blob.
    # We do this by checking if the ants pixels overlap with an ant or a blob.
    for ant in ants:
        added_to_blob = False
        # We check against existing blobs first, otherwise we may end up having overlapping ants ending
        # up in separate blob instances.
        for blob in ant_blobs:
            if ant.overlaps(blob.prime):
                added_to_blob = True
                ants.remove(ant)
                blob.add(ant)
                print("{} has joined blob: {}".format(ant.id, blob))
                break
        # We don't need to check if it overlaps other ants if it ended up in a blob this frame, since those ants
        # will inevitably join the same blob.
        if not added_to_blob:
            for other in ants:
                if ant != other and ant.overlaps(other):
                    ants.remove(ant)
                    ants.remove(other)
                    newBlob = AntBlob(ant)
                    newBlob.add(other)
                    ant_blobs.append(newBlob)
                    print("New blob formed: " + str(newBlob))
                    break

    # Draw tracking info for ants and blobs.
    if not tracked_ant:
        for ant in ants:
            ant.draw(frame_original)
        for blob in ant_blobs:
            blob.draw(frame_original)

    # Draw the frame number in the top left corner.
    draw_text(frame_original, frame_num, (5, 30), 2)

    bg_edited = background_img.copy()
    frame_edited = frame_original.copy()
    if tracked_ant:
        frame_original = bg_edited.copy()
        frame_original[tracked_ant.minY: tracked_ant.maxY+1, tracked_ant.minX: tracked_ant.maxX+1, :] = frame_edited[tracked_ant.minY: tracked_ant.maxY+1, tracked_ant.minX: tracked_ant.maxX+1, :]
    cv2.imshow("Ants", frame_original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
