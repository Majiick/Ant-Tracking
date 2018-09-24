import numpy as np
import cv2

cap = cv2.VideoCapture('../video1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    original_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.GaussianBlur(frame,(15,15),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h.fill(255)
    #for i in range(len(h.flat)):
        #if h.flat[i] != 255:
            #h.flat[i] = 0
    s.fill(255)
    # v.fill(255)
    # ret,v = cv2.threshold(v,127,255,cv2.THRESH_BINARY)
    frame = cv2.merge([h, s, v])
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)


    #grey = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(frame,50,60)
    cv2.imshow('edges', edges)
    cv2.imshow('original', original_frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()