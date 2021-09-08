
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from io import BytesIO
import numpy as np
import os


# img = image.load_img('validation', target_size=(200, 200))


def preprocess(img):

    try:
        image_bytes = img.tobytes()  # Covert to bytes
        # Read from BytesIO (Exception!!!)
        new_image = Image.open(BytesIO(image_bytes))
        return img
    except:
        return img


print(tf.config.list_physical_devices())

train = ImageDataGenerator(rescale=1/255, preprocessing_function=preprocess)
validation = ImageDataGenerator(
    rescale=1/255, preprocessing_function=preprocess)


print("\ntraining dataset...")
train_dataset = train.flow_from_directory(
    'training', target_size=(200, 200), batch_size=10, class_mode='binary', shuffle=True)

print(train_dataset.class_indices)
# print(train_dataset.classes)


print("\n\nvalidation dataset...")

validation_dataset = train.flow_from_directory(
    'validation', target_size=(200, 200), batch_size=10, class_mode='binary', shuffle=True)

print(validation_dataset.class_indices)
# print(validation_dataset.classes)


# model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    #
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           ),
    tf.keras.layers.MaxPool2D(2, 2),
    #
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           ),
    tf.keras.layers.MaxPool2D(2, 2),

    ##
    tf.keras.layers.Flatten(),

    ##
    tf.keras.layers.Dense(512, activation='relu'),
    ##
    tf.keras.layers.Dense(1, activation='sigmoid'),

])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=.001), metrics=['accuracy'])
try:
    model_fit = model.fit(train_dataset, steps_per_epoch=5, epochs=30,
                          validation_data=validation_dataset)
except:
    pass

cwd = os.getcwd()
dir_path = "testing"

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + "/"+i, target_size=(200, 200))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    val = model.predict(images)
    if val == 0:
        print('Penguin')
    else:
        print("Puffin")
