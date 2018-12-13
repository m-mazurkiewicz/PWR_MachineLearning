import os
from PIL import Image
from resizeimage import resizeimage
from tqdm import tqdm
import sys

base_dir = '/content/gdrive/My Drive/PWr_AlexNet_data/raw/'

reshape_type = sys.argv[1]
resample_method = sys.argv[2]

resample_methods_dict = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS}


def resize_image(image_path, image_name, resample_method):
    with open(image_path + image_name, 'r+b') as f:
        with Image.open(f) as image:
            img = image.resize(size=[227, 227], resample=resample_method)
            img.save(image_path.replace('raw/', 'interim/no-padding/') + image_name, img.format)


def crop_image(image_path, image_name, resample_method):
    with open(image_path + image_name, 'r+b') as f:
        with Image.open(f) as image:
            width, height = image.size
            if width <= height:
                img = resizeimage.resize_width(image, 227, resample=resample_method)
            else:
                img = resizeimage.resize_height(image, 227, resample=resample_method)
            img = resizeimage.resize_crop(img, size=[227, 227])
            img.save(image_path.replace('raw/', 'interim/no-padding/') + image_name, img.format)


def proceed_class(class_name, reshape_type, resample_method):
    os.makedirs((base_dir + class_name + '/').replace('raw/', 'interim/no-padding/'),exist_ok=True)
    if reshape_type == 'crop':
        for image in tqdm(os.listdir(base_dir + class_name)):
            resize_image(base_dir + class_name + '/', image, resample_methods_dict[resample_method])
    elif reshape_type == 'resize':
        for image in tqdm(os.listdir(base_dir + class_name)):
            resize_image(base_dir + class_name + '/', image, resample_methods_dict[resample_method])
    else:
        raise ValueError('No such reshape_type - \'{0}\' '.format(reshape_type))


for class_name in tqdm(os.listdir(base_dir)):
    proceed_class(class_name, reshape_type, resample_method)
