import numpy as np
import cv2
import sys
import copy
import math
import bisect

MIN_ANT_SIZE = 15   # TODO should find a way to reasonably determine this.


class Ant:
    id = 0

    def __init__(self, pixels):
        self.id = Ant.id
        Ant.id+=1
        self.prev_sizes = list()
        self.prev_centres = list()
        self.update(pixels)

    def update(self, pixels):
        self.pixels = pixels
        self.size = len(pixels)
        # Store the length in a sorted list to allow the median to quickly be determined.
        # It is important that an ant is never updated in a blob to prevent incorrect sizes ending up in this.
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
        # The coordinates of an ant are stored each frame to allow us to determine its direction of travel.
        self.prev_centres.append(self.centre)

    def search_and_update(self, mask):
        # TODO we need to deal with ants leaving the screen. At the moment, when they do leave the screen, this will
        # just find the nearest ant, which inevitably leads to a blob forming that shouldn't. Should probably do
        # something along the lines of not tracking any ants that have pixels on the border of the screen.
        nearest = find_nearest_white(mask, self.centre)
        pixels = flood_fill(mask, nearest[0], nearest[1])
        # TODO: If the size is significantly smaller here, the we've found a small subset of pixels near the ant,
        # and in that case we should continue searching somehow until we find the rest of the ant.
        self.update(pixels)

    def direction(self):
        if len(self.prev_centres) < 2:
            # Not much we can do if we have no historical data on the ant, at least returning zero will make it deterministic.
            return 0
        # TODO For now, we just get the direction based on the last two centres, we probably should get the average
        # direction over the ant's lifetime.
        return direction(self.prev_centres[-2], self.prev_centres[-1])

    def median_size(self):
        return np.median(self.prev_sizes)

    def overlaps(self, other):
        return not self.pixels.isdisjoint(other.pixels)

    def draw(self, img, box_color=(0, 0, 255)):
        cv2.rectangle(img=img, pt1=(self.minX, self.minY), pt2=(self.maxX, self.maxY), color=box_color, thickness=1)
        draw_text(img, str(self.id), self.centre, 1)

    def __str__(self):
        return "{}: {}, width: {}, height: {}, size: {}".format(self.id, self.centre, self.maxX-self.minX, self.maxY-self.minY, self.size)


class AntBlob:
    id = 0

    def __init__(self, ant):
        self.id = AntBlob.id
        AntBlob.id += 1
        # prime isn't really an ant, it's just used to keep track of where the blob is.
        self.prime = copy.deepcopy(ant)
        self.prime.id = self.id
        self.ants = dict()
        self.add(ant)

    def add(self, ant):
        self.ants[ant.id] = ant

    def exit(self, leaver_ant):
        # Check which ant is closest direction-wise.
        # We look at the prime ant's position in the last frame, since blob positions are updated before checking for
        # ants that have left.
        leaver_direction = direction(self.prime.prev_centres[-2], leaver_ant.centre)
        leaver_size = leaver_ant.size

        best_weight = sys.maxint  # Smaller weight is better.
        best_key = -1
        for ant_key in self.ants:
            angle_difference = difference_in_direction(self.ants[ant_key].direction(), leaver_direction)
            size_difference = math.fabs((float(leaver_size) - self.ants[ant_key].median_size()) / leaver_size)
            # TODO we could weight these differently to improve the accuracy.
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
        self.prime.search_and_update(mask)

    def overlaps(self, other):
        return self.prime.overlaps(other.prime)

    def draw(self, img):
        self.prime.draw(img, box_color=(0,255,255))

    def __str__(self):
        return str(self.prime) + ", ants: {" + ", ".join(str(ant.id) for ant in self.ants.values()) + "}"


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
                    new_ants.append(Ant(ant_pixels))
                    used_pixels.update(ant_pixels)
    return new_ants


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
# TODO ideally we should have a live display of the mouses current coordinates to make debugging easier.
def print_coords(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print x, y


def main():
    cap = cv2.VideoCapture('../video1.mp4')  # Load capture video.
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()  # Create a background subtractor.

    cv2.namedWindow("fgmask")
    cv2.setMouseCallback("fgmask", print_coords)

    # Skip first 200 frames to build up history for the background subtractor.
    for _ in range(200):
        ret, frame = cap.read()
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        bgSubtractor.apply(frame)

    ants = list()
    ant_blobs = list()
    while cap.isOpened():  # While video is opened
        _, frame = cap.read()
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print "***FRAME {}***".format(frame_num)
        original_frame = frame.copy()  # Save the original frame so we can use it for reference later on.

        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Blur the image using a gaussian blur with a 5x5 kernel.
        fgmask = bgSubtractor.apply(frame)  # Apply the frame to the background subtractor. fgmask is the result of subtracting the background.
        # TODO not sure if we actually need to convert this to BGR
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        # Update ant and blob positions.
        for ant in ants:
            ant.search_and_update(fgmask[:,:,0])
        for blob in ant_blobs:
            blob.search_and_update(fgmask[:,:,0])

        # Check if any blobs have merged.
        for blob in ant_blobs:
            for other in ant_blobs:
                if blob != other and blob.overlaps(other):
                    print "Merging blobs: {} and {}".format(blob, other)
                    blob.merge(other)
                    # TODO will removing from the list while iterating break stuff?
                    ant_blobs.remove(other)

        # Check for un-tracked ants.
        # If there is a blob within range of an un-tracked ant, assume it came from that, otherwise it is a new ant.
        new_ants = find_new_ants(fgmask[:,:,0], ants, ant_blobs)
        print "*****\nUn-tracked ants:\n"
        for new_ant in new_ants:
            from_blob = False
            print "\t{}".format(new_ant)
            for blob in ant_blobs:
                # TODO replace magic distance number with something reasonable
                if dist(new_ant.centre, blob.prime.centre) < 50:
                    from_blob = True
                    print "Possible blob exit: " + str(new_ant) + " from " + str(blob)
                    ant_from_blob = blob.exit(new_ant)
                    if ant_from_blob is None:
                        # The blob had no ants to give.
                        ants.append(new_ant)
                    else:
                        # Ant taken from blob.
                        ant_from_blob.update(new_ant.pixels)
                        ants.append(ant_from_blob)
                    if len(blob.ants) == 1:
                        # TODO passing the prime ant here currently does nothing, but may break stuff when we
                        # actually look at an ants parameters when determining which ant should be pulled from the blob.
                        final_ant = blob.remove_last_ant()
                        final_ant.update(blob.prime.pixels)
                        ants.append(final_ant)
                        ant_blobs.remove(blob)
            if not from_blob:
                # Ant is entirely new.
                ants.append(new_ant)
        print "*****"

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
                    print "{} has joined blob: {}".format(ant.id, blob)
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
                        print "New blob formed: " + str(newBlob)
                        break

        # Draw tracking info for ants and blobs.
        for ant in ants:
            ant.draw(fgmask)
        for blob in ant_blobs:
            blob.draw(fgmask)

        # Draw the frame number in the top left corner for debugging purposes.
        draw_text(fgmask, frame_num, (5,30), 2)

        # cv2.imshow('original', original_frame)  # Draw the original frame image
        cv2.imshow('fgmask', fgmask)  # Draw the foreground mask

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
