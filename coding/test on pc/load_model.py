import cv2
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

class_names = ['S1', 'S2','R','Empty']
width = 96
height = 96
model = load_model('lettuce.h5')

type_1 = image.load_img('./images/test/img.jpg', target_size=(width, height))
type_1_X = np.expand_dims(type_1, axis=0)#Expand the shape of an array
predictions = model.predict(type_1_X)
#Generates output predictions for the input samples.
#Computation is done in batches.
print('The type predicted is: {}'.format(class_names[np.argmax(predictions)]))#Returns the indices of the maximum values along an axis.






