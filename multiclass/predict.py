from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sn
from skimage import transform
import cv2
import os

# print(tf.__version__)

# # Read Processed Data

img_height, img_width = (224, 224)
batch_size = 16

train_data_dir = r"processed_data/train"
valid_data_dir = r"processed_data/val"
test_data_dir = r"processed_data/test"


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=.2, zoom_range=.2, horizontal_flip=True, validation_split=.4
                                   )

test_generator = train_datagen.flow_from_directory(test_data_dir, target_size=(
    img_height, img_width), batch_size=1, class_mode='categorical', subset='validation')


model = tf.keras.models.load_model('model.h5')


# Summary / Result



def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


def output_format(prediction):
    keys = test_generator.class_indices.keys()
    result = list(zip(keys,prediction[0]))
    return result

def result(prediction):
    r = output_format(prediction)
    ans_index = np.argmax(prediction)
    return r[ans_index]

image = cv2.imread('duck_5.jpg')  

image_tensor = load('duck_5.jpg')
prediction = model.predict(image_tensor)
prediction_formatted = output_format(prediction)
print(prediction_formatted)

texted_image =cv2.putText(img=np.copy(image), text=str(result(prediction)), org=(0,50),fontFace=1, fontScale=1, color=(0,0,255), thickness=1)
plt.imshow(texted_image)
plt.show()
