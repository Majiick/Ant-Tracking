import numpy as np
import cv2
import sys
import copy
import math

MIN_ANT_SIZE = 15


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
        # TODO: If the size is significantly smaller here, the we've found a small subset of pixels near the ant,
        # in that case we should continue searching somehow until we find the rest of the ant.
        self.update(pixels)

    def overlaps(self, other):
        return not self.pixels.isdisjoint(other.pixels)

    def draw(self, img, box_color=(0, 0, 255)):
        cv2.rectangle(img=img, pt1=(self.minX, self.minY), pt2=(self.maxX, self.maxY), color=box_color, thickness=1)
        cv2.putText(img=img, text=str(self.id), org=self.centre, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(img=img, text=str(self.id), org=self.centre, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=1)

    def __str__(self):
        return "{}: {}, width: {}, height: {}, size: {}".format(self.id, self.centre, self.maxX-self.minX, self.maxY-self.minY, self.size)


class AntBlob:
    id = 0

    def __init__(self, ant):
        self.id = AntBlob.id
        AntBlob.id += 1
        # TODO: deal with prime ant leaving the blob.
        self.prime = copy.deepcopy(ant)
        self.prime.id = self.id
        self.ants = dict()
        self.add(ant)

    def add(self, ant):
        self.ants[ant.id] = ant

    def exit(self, ant):
        # TODO: for now this just removes the first ant in the dict, need to update ants to store more info to determine which ant is probably leaving.
        id = self.ants.iterkeys().next()
        candidate = self.ants[id]

        del self.ants[id]
        if len(self.ants) == 1:
            # TODO there's only one ant left in the blob, return it to ant status and delete the blob.
            pass
        return candidate

    def search_and_update(self, mask):
        self.prime.search_and_update(mask)

    def overlaps(self, other):
        return self.prime.overlaps(other)

    def draw(self, img):
        self.prime.draw(img, box_color=(0,255,255))

    def __str__(self):
        return str(self.prime) + ", ants: {" + ", ".join(str(ant.id) for ant in self.ants.values()) + "}"


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

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
                if len(ant_pixels) > MIN_ANT_SIZE:
                    ants.append(Ant(ant_pixels))
                    used_pixels.update(ant_pixels)

    return ants


# TODO: This is pretty similar to find_ants, they should probably share code somehow.
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
        ret, frame = cap.read()
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        bgSubtractor.apply(frame)

    firstFrame = True
    ants = list()
    antBlobs = list()
    while cap.isOpened():  # While video is opened
        _, frame = cap.read()
        print "***FRAME {}***".format(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
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
            for blob in antBlobs:
                blob.search_and_update(fgmask[:,:,0])
            # TODO check if two blobs have overlapped and merge them
            # Search for any pixels that don't belong to an existing ant or blob.
            # If there is a blob within range, assume it came from that.
            new_ants = find_new_ants(fgmask[:,:,0], ants, antBlobs)
            print "*****\nUn-tracked ants\n"
            for new_ant in new_ants:
                from_blob = False
                print "\t{}".format(new_ant)
                for blob in antBlobs:
                    # TODO replace magic 100 with a reasonable distance
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
                            final_ant = blob.exit(blob.prime)
                            final_ant.update(blob.prime.pixels)
                            ants.append(final_ant)
                            antBlobs.remove(blob)
                if not from_blob:
                    # Ant is entirely new.
                    ants.append(new_ant)
            print "#####END"

        # Check each ant to see if they have entered or formed a blob.
        # We do this by checking if the ants pixels overlap with an ant or a blob.
        for ant in ants:
            ant_blobbed = False
            # We check against existing blobs first, otherwise we may end up having overlapping ants ending
            # up in separate blob instances.
            for blob in antBlobs:
                if blob.overlaps(ant):
                    ant_blobbed = True
                    ants.remove(ant)
                    blob.add(ant)
                    print "{} has joined blob: {}".format(ant.id, blob)
                    break

            # We don't want to check for overlapping if it ended up in a blob this frame.
            # Any overlapping ants will inevitably join the same blob.
            if not ant_blobbed:
                for other in ants:
                    if ant != other and ant.overlaps(other):
                        ants.remove(ant)
                        ants.remove(other)
                        newBlob = AntBlob(ant)
                        newBlob.add(other)
                        antBlobs.append(newBlob)
                        print "New blob formed: " + str(newBlob)
                        break

        for ant in ants:
            ant.draw(fgmask)

        for blob in antBlobs:
            blob.draw(fgmask)

        # cv2.imshow('original', original_frame)  # Draw the original frame image
        cv2.imshow('fgmask', fgmask)  # Draw the foreground mask

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


