import urllib.request
import cv2
import numpy as np
import time
import subprocess

url='http://192.168.137.21:8080/photoaf.jpg'
timeout = time.time() + 5

while True:
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    cv2.imshow("test", img)
    k = cv2.waitKey(1)
    if time.time() > timeout:
        k = cv2.waitKey(1)
        break
    cv2.imshow('test',img)
    cv2.imwrite("img.png",img)
    # all the opencv processing is done here
    if ord('q')==cv2.waitKey(10):
        exit(0)

subprocess.call("python predict.py img.png myproject-zero ICN1534353649611479451")