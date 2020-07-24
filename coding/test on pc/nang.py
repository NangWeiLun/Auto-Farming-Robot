import io
import os
import cv2
import time

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
timeout = time.time() + 5
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    if time.time() > timeout:
        k = cv2.waitKey(1)
        break

img_name = "opencv_frame_{}.png".format(img_counter)
cv2.imwrite(img_name, frame)
print("{} written!".format(img_name))
img_counter += 1

cam.release()
cv2.destroyAllWindows()

#image processing part
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="D:/Workshop 2/test on pcProject-Zero-eea84c681d62.json"
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(''),
    "C:/Users/User/Documents/workshop2/opencv_frame_0.png")

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)