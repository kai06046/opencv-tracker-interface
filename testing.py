import cv2
import numpy as np
import time

KEY_LEFT = 2424832
KEY_RIGHT = 2555904

path = '../burying_beetle/videos/[CH01] 2016-10-14 19.20.00_x264.avi'

video = cv2.VideoCapture(path)

# if not ok:
#     break

# ok, frame = video.read()
THRES = 155
while True:
    ok, frame = video.read()
    if not ok:
        break
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key in [ord('w'), KEY_RIGHT]:
        THRES += 1
        print(THRES)
    elif key in [ord('s'), KEY_LEFT]:
        THRES -= 1
        print(THRES)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    T, th = cv2.threshold(blurred, THRES, 255, cv2.THRESH_BINARY)
    T, thinv = cv2.threshold(blurred, THRES, 255, cv2.THRESH_BINARY_INV)

    res = cv2.bitwise_and(gray, gray, mask=th)
    cv2.imshow('hi', res)

cv2.destroyAllWindows()
video.release()