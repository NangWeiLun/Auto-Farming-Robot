#!/usr/bin/python

from __future__ import print_function
import rpyc
import time
import threading
from threading import Thread
import urllib.request
import numpy as np
import time
import subprocess
import sys
import os
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
import argparse
import imutils
from imutils.video import VideoStream
import numpy as np
from numpy import pi, sin, cos
from urllib.request import urlopen
import tensorflow as tf
import cv2 as cv ,cv2
import wave
from imutils.video import VideoStream
import argparse
from keras.models import load_model
from keras.preprocessing import image

cvNet = None
myName = 'computer'
showVideoStream = False
currentClassDetecting = 'cup'

class_names = ['S1', 'S2','R','Empty']
width = 96
height = 96

host = 'http://10.73.39.97:8080/'
url = host + 'shot.jpg'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/Workshop 2/test on pc/Project-Zero-eea84c681d62.json"

project_id = "myproject-zero"  # sys.argv[2]
#model_id = "ICN1534353649611479451"  # sys.argv[3]
model_id = "ICN2049808731746633105"

conn = rpyc.classic.connect('ev3dev')
ev3 = conn.modules['ev3dev.ev3']
core = conn.modules['ev3dev.core']
subprocess = conn.modules['subprocess']
smbus = conn.modules['smbus']

motorD = ev3.MediumMotor('outD')  # register arm
ultraS = ev3.UltrasonicSensor('in1')  # register ultrasonic
ultraS.mode = 'US-DIST-CM'

model = load_model('lettuce.h5')
netModels = {
        'modelPath': 'mobilenet_ssd_v1_coco/frozen_inference_graph.pb',
        'configPath': 'mobilenet_ssd_v1_coco/ssd_mobilenet_v1_coco_2017_11_17.pbtxt',
        'classNames': { 
            0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
            18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
            32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' 
        }
}

def trackObject():
    with tf.gfile.FastGFile('TensorFlow/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    camera = cv2.VideoCapture(0)
    while True:
        (grabbed, frame) = camera.read()
        status = "No Targets"
        if not grabbed:
            break
        img = imutils.resize(frame, width=800)
        with tf.Session() as sess:
            # Restore session
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            # Read and preprocess an image.
            #img = cv.imread('IMG_20181208_020712.jpg')
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.3:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
        cv.imshow('TensorFlow MobileNet-SSD', img)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    camera.release()
    cv.destroyAllWindows()


def get_prediction(content, project_id, model_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()
    name = 'projects/{}/locations/us-central1/models/{}'.format(
        project_id, model_id)
    payload = {'image': {'image_bytes': content}}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned


def armclose():
    motorD.run_timed(time_sp=750, speed_sp=-1000)
    time.sleep(0.5)


def armopen():
    motorD.run_timed(time_sp=750, speed_sp=1000)  # open
    time.sleep(0.5)


def motor(runtime, action, speed):
    subprocess.call("python3 'test on pc/motor.py' " +str(runtime)+" "+action + " " + speed, shell=True)
    time.sleep(runtime+0.2)


def armbackforward(start, end):
    subprocess.call("python3 'test on pc/armgoset.py' " + str(start) + " " + str(end), shell=True)
    time.sleep(abs(start-end)/20)


def armupdown(value):
    motorC.run_to_rel_pos(position_sp=value, speed_sp=100, stop_action="hold")


def armin(start, end):
    subprocess.call("python3 'test on pc/armin.py' " + str(start) + " " + str(end), shell=True)
    time.sleep(abs(start-end)/20)


def armout(start, end):
    subprocess.call("python3 'test on pc/armout.py' " + str(start) + " " + str(end), shell=True)
    time.sleep(abs(start-end)/20)


def armgoout():
    armbackforward(90, 0)
    time.sleep(1)


def armgoin():
    armbackforward(0, 90)
    time.sleep(1)


def imgCaptureAndDownload():
    subprocess.call("python 'test on pc/imgtransferrpyc.py'", shell=True)
    time.sleep(2)
    rpyc.classic.download(
        conn, "test on pc/opencv_frame_0.png", "opencv_frame_0.jpg")
    time.sleep(1)

def scan():
    location = -1
    for i in range(0,3):
        print(i)
        #print(marginLeft,marginTop)
        label = takephoto()
        print(label)
        if (label == "R2" or label == "Empty"):
            location = i
            putLabel(label,location)
        elif (label == "S1"):
            print("error, scan with tensorflow keras model")
            label = takephoto2()
            if (label == "S1"):
                print("detection error")
        if (i==2):
            break
        motor(0.4,"forward","slow")
    motor((0.4*2)-0.2,"backward","slow")
    return location,label


def takephoto():
    time.sleep(0.5)
    predictions = ""
    url='http://10.73.39.97:8080/photoaf.jpg'
    timeout = time.time() + 5
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        y = 100
        x = 50
        h = 400
        w = 400
        img2 = img[y:y+h, x:x+w]
        cv2.imwrite("img.jpg", img)
        cv2.imwrite("img2.jpg", img2)
        k = cv2.waitKey(1)
        if time.time() > timeout:
            k = cv2.waitKey(1)
            break
        cv2.imshow('test', img)
        cv2.imshow('test2', img2)
        # all the opencv processing is done here
        if ord('q') == cv2.waitKey(10):
            exit(0)
    file_path = "img2.jpg"  # sys.argv[1]
    with open(file_path, 'rb') as ff:
        content = ff.read()
    label = get_prediction(content, project_id,  model_id)
    if (len(label.payload)==0):
        file_path = "img.jpg"  # sys.argv[1]
        with open(file_path, 'rb') as ff:
            content = ff.read()
        label = get_prediction(content, project_id,  model_id)
        print(label)
    if (len(label.payload)==0):
        label = ""
    else:
        label = label.payload[0].display_name
    """cv2.destroyWindow("test")
    cv2.destroyWindow("test2") """
    time.sleep(0.5)
    return label

def takephoto2():
    time.sleep(0.5)
    predictions = ""
    url='http://10.73.39.97:8080/photoaf.jpg'
    timeout = time.time() + 5
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        y = 100
        x = 50
        h = 400
        w = 400
        img2 = img[y:y+h, x:x+w]
        cv2.imwrite("img.jpg", img)
        cv2.imwrite("img2.jpg", img2)
        k = cv2.waitKey(1)
        if time.time() > timeout:
            k = cv2.waitKey(1)
            break
        cv2.imshow('test', img)
        cv2.imshow('test2', img2)
        # all the opencv processing is done here
        if ord('q') == cv2.waitKey(10):
            exit(0)
    type_1 = image.load_img('img.jpg', target_size=(width, height))
    type_1_X = np.expand_dims(type_1, axis=0)#Expand the shape of an array
    predictions = model.predict(type_1_X)
    print('The type predicted is: {}'.format(class_names[np.argmax(predictions)]))#Returns the indices of the maximum values along an axis.
    label = class_names[np.argmax(predictions)]
    type_1 = image.load_img('img2.jpg', target_size=(width, height))
    type_1_X = np.expand_dims(type_1, axis=0)#Expand the shape of an array
    predictions = model.predict(type_1_X)
    print('The type predicted is: {}'.format(class_names[np.argmax(predictions)]))#Returns the indices of the maximum values along an axis.
    label2 = class_names[np.argmax(predictions)]
    """cv2.destroyWindow("test")
    cv2.destroyWindow("test2") """
    print(label,label2) 
    if (label == label2):
        return label
    else:
         print("error")
    time.sleep(0.5)
    return ""


def grabPlant():
    armgoout()
    armclose()
    armgoin()
    time.sleep(1)


def putPlant():
    armgoout()
    armopen()
    armgoin()
    time.sleep(1)


def putLabel(label,location):
    if (label == "R2"):
        grabPlant()
        if(location == 0):
            time = 1.8
        elif(location == 1):
            time = 1.55
        elif(location == 2):
            time = 1.3
        motor(time,"forward","slow")
        while(ultraS.value() < 610 or ultraS.value() > 630):
            if(ultraS.value() < 610):
                motor(0.1,"forward","slow")
            elif(ultraS.value() > 630):
                motor(0.1,"backward","slow")
        putPlant()
        motor(0.7,"backward","slow")
        while(ultraS.value() < 426 or ultraS.value() > 442):
            if(ultraS.value() < 426):
                motor(0.1,"forward","slow")
            elif(ultraS.value() > 442):
                motor(0.1,"backward","slow")
        putNew(location)
    elif (label == "Empty"):
        if(location == 0):
            time = 1.3
        elif(location == 1):
            time = 1.1
        elif(location == 2):
            time = 0.85
        while(ultraS.value() < 422 or ultraS.value() > 436):
            if(ultraS.value() < 422 ):
                motor(0.1,"forward","slow")
            elif(ultraS.value() > 436):
                motor(0.1,"backward","slow")
        putNew(location)
    """elif (label == "S1"):
        grabPlant()
        motor(0.5+(0.4*location)-0.2,"forward","slow")
        putPlant()
        motor(0.2,"forward","slow")
        putNew(location) """
    print("put label finish")
    pass


def putNew(location):
    label = takephoto()
    if(location == 0):
        time = 1.3
        us = 40
    elif(location == 1):
        time = 1.1
        us = 139
    elif(location == 2):
        time = 0.8
        us = 217
    if (label == "S2"):
        grabPlant()
        motor(time,"backward","slow")
        while (ultraS.value() < us - 7 or ultraS.value() > us + 7):
            if (ultraS.value() < us - 7):
                motor(0.1,"forward","slow")
            elif (ultraS.value() > us + 7):
                motor(0.1,"backward","slow")
        putPlant()
    elif(label == "Empty"):
        motor(time,"backward","slow")
        while (ultraS.value() < us - 7 or ultraS.value() > us + 7):
            if (ultraS.value() < us - 7):
                motor(0.1,"forward","slow")
            elif (ultraS.value() > us + 7):
                motor(0.1,"backward","slow")
    pass


def run_video_detection():
    scoreThreshold = 0.3
    trackingThreshold = 50
    cvNet = cv.dnn.readNetFromTensorflow(netModels['modelPath'], netModels['configPath'])
    cap = create_capture()
    global showVideoStream
    while showVideoStream:
        cap = create_capture()
        img = cap
        # run detection
        cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        detections = cvNet.forward()
        track_object(img, detections[0,0,:,:], scoreThreshold, netModels['classNames'], currentClassDetecting, trackingThreshold)
        cv.imshow('Real-Time Object Detection', img)
        ch = cv.waitKey(1)
        if ch == 27:
            showVideoStream = False
            break
    print('exiting run_video_detection...')
    cv.destroyAllWindows()
    pass

def create_capture(source = 0):
    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]
    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )
    imgResp=urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    cap=cv2.imdecode(imgNp,-1)
    #cap = cv.VideoCapture(source)
    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    #if cap is None or not cap.isOpened():
    #   print('Warning: unable to open video source: ', source)
    return cap


def detect_all_objects(img, detections, score_threshold, classNames):
    for detection in detections:
        class_id = int(detection[1])
        score = float(detection[2])
        if score > score_threshold:
            label_class(img, detection, score, classNames[class_id])
    pass


def track_object(img, detections, score_threshold, classNames, className, tracking_threshold):
    for detection in detections:
        score = float(detection[2])
        class_id = int(detection[1])
        if className in classNames.values() and className == classNames[class_id] and score > score_threshold:
            rows = img.shape[0]
            cols = img.shape[1]
            global marginLeft, marginTop
            marginLeft = int(detection[3] * cols) # xLeft
            marginRight = cols - int(detection[5] * cols) # cols - xRight
            xMarginDiff = abs(marginLeft - marginRight)
            marginTop = int(detection[4] * rows) # yTop
            marginBottom = rows - int(detection[6] * rows) # rows - yBottom
            yMarginDiff = abs(marginTop - marginBottom)
            if xMarginDiff < tracking_threshold and yMarginDiff < tracking_threshold:
                boxColor = (0, 255, 0)
            else:
                boxColor = (0, 0, 255)
            label_class(img, detection, score, classNames[class_id], boxColor)
    pass 

def label_class(img, detection, score, className, boxColor=None):
    rows = img.shape[0]
    cols = img.shape[1]
    if boxColor == None:
        boxColor = (23, 230, 210)
    xLeft = int(detection[3] * cols)
    yTop = int(detection[4] * rows)
    xRight = int(detection[5] * cols)
    yBottom = int(detection[6] * rows)
    cv.rectangle(img, (xLeft, yTop), (xRight, yBottom), boxColor, thickness=4)
    label = className + ": " + str(int(round(score * 100))) + '%'
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    yTop = max(yTop, labelSize[1])
    cv.rectangle(img, (xLeft, yTop - labelSize[1]), (xLeft + labelSize[0], yTop + baseLine),
        (255, 255, 255), cv.FILLED)
    cv.putText(img, label, (xLeft, yTop), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    pass

def initev3():
    print("init start")
    while ultraS.value() > 51 or ultraS.value() < 45:
        print(ultraS.value())
        if (ultraS.value() > 80 and ultraS.value() < 700 ):
            motor(0.3,"backward","slow")
        elif (ultraS.value() > 51 and ultraS.value() < 80):
            motor(0.1,"backward","slow")
        elif (ultraS.value() < 45 ):
            motor(0.1,"forward","slow")
    armgoout()
    armgoin()
    motorD.run_timed(time_sp=750, speed_sp=-1000)
    time.sleep(0.75)
    motorD.run_timed(time_sp=750, speed_sp=-500)
    time.sleep(0.75)
    motorD.run_timed(time_sp=750, speed_sp=1000)
    time.sleep(0.75)
    print("init finish")


if __name__ == "__main__":
    showVideoStream = True
    marginLeft = 0
    marginTop = 0
    videoStreamThread = threading.Thread(target=run_video_detection)
    videoStreamThread.start()
    while True:
        initev3()
        location, label = scan()
        print(marginLeft,marginTop)
        time.sleep(1)
        print("next 7 day")
        time.sleep(5)
        motor(0.1,"swipeFleft","slow")
        motor(0.2,"forward","slow")
        
