#!/usr/bin/python
import io
import os
import cv2
import time
from sys import stderr

cam = cv2.VideoCapture(0)
img_counter = 0
timeout = time.time() + 5
for i in range(30):
    ret, frame = cam.read()
    if time.time() > timeout:
        break

print("take photo")
img_name = "test on pc/opencv_frame_{}.png".format(img_counter)
cv2.imwrite(img_name, frame)
print("{} written!".format(img_name))
img_counter += 1
cam.release()