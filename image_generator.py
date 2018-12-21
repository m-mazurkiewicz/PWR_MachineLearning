import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, save_img
from matplotlib import pyplot as plt
import skimage.io as io
import numpy as np
import os
from tqdm import tqdm

base_dir_processed = '/content/gdrive/My Drive/PWr_AlexNet_data/processed/'
data_set = 'no-padding/resize/'


image_gen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=(0.5,1.5),
    shear_range=0.01,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True,
    validation_split=0.1
)

number_of_images_for_fit = -1

all_images = []
for class_name in os.listdir(base_dir_processed + data_set + 'train'):
  for image_path in tqdm(os.listdir(base_dir_processed + data_set + 'train/' + class_name)[:number_of_images_for_fit]):
    img = io.imread(base_dir_processed + data_set + 'train/' + class_name + '/' + image_path)
    all_images.append(img)
x_train = np.array(all_images)

image_gen.fit(x_train)

generator = image_gen.flow_from_directory(
    base_dir_processed + data_set + '/' + 'train',
        target_size=(227,227),
        batch_size=32,
        class_mode='categorical')

i=0
for x, y in generator:
    img = array_to_img(x[0,:,:,:])
    plt.figure()
    plt.imshow(img)
    i += 1
    if i==40:
        break