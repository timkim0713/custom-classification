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
import os
os.add_dll_directory(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
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

filenames = test_generator.filenames
nb_samples = len(test_generator)
y_prob = []
y_act = []
test_file = []
test_generator.reset()

print(test_generator.class_indices.keys())
index = 0


for _ in range(nb_samples):

    index = next(test_generator.index_generator)
    X_test, Y_test = test_generator._get_batches_of_transformed_samples(index)
    image_name = test_generator.filenames[int(index)]

    y_prob.append(model.predict(X_test))
    y_act.append(Y_test)
    test_file.append(image_name)

predicted_class = [list(test_generator.class_indices.keys())[
    i.argmax()] for i in y_prob]
actual_class = [list(test_generator.class_indices.keys())
                [i.argmax()] for i in y_act]

print("predicted class: ", predicted_class)
print("actual class: ", actual_class)

print("test img:", test_file)

out_df = pd.DataFrame(np.vstack([predicted_class, actual_class]).T, columns=[
    'predicted_class', 'actual_class'])
confusion_matrix = pd.crosstab(out_df['actual_class'], out_df['predicted_class'], rownames=[
    'Actual'], colnames=['Predicted'])


sn.heatmap(confusion_matrix, cmap='Blues', annot=True, fmt='d')
print('test accuracy: {}'.format(
    (np.diagonal(confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))

plt.show()
