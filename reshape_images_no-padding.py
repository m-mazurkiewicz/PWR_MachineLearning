import os
from PIL import Image
from tqdm import tqdm
import sys

base_dir = '/content/gdrive/My Drive/PWr_AlexNet_data/raw/'

reshape_method = sys.argv[1]
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
            img.save(image_path.replace('raw/', 'interim/no-padding/resize/') + image_name, img.format)


def crop_image(image_path, image_name, resample_method):
    '''
    Crop image in the middle of the image in square shape of the biggest possible size
    :param image_path:
    :param image_name:
    :param resample_method:
    '''
    with open(image_path + image_name, 'r+b') as f:
        with Image.open(f) as image:
            width, height = image.size  # Get dimensions
            if width<=height:
                new_width = width
                new_height = width
            else:
                new_width = height
                new_height = height
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            img = image.crop((left, top, right, bottom))
            img = img.resize(size=[227, 227], resample=resample_method)
            img.save(image_path.replace('raw/', 'interim/no-padding/crop/') + image_name, image.format)


def proceed_class(class_name, reshape_method, resample_method):
    if reshape_method == 'crop':
        os.makedirs((base_dir + class_name + '/').replace('raw/', 'interim/no-padding/crop/'), exist_ok=True)
        for key,image in enumerate(os.listdir(base_dir + class_name)):
            crop_image(base_dir + class_name + '/', image, resample_methods_dict[resample_method])
            if key % 100 == 0:
                print(key)
    elif reshape_method == 'resize':
        os.makedirs((base_dir + class_name + '/').replace('raw/', 'interim/no-padding/resize/'), exist_ok=True)
        for key,image in enumerate(os.listdir(base_dir + class_name)):
            resize_image(base_dir + class_name + '/', image, resample_methods_dict[resample_method])
            if key % 100 == 0:
                print(key)
    else:
        raise ValueError('No such reshape_method - \'{0}\' '.format(reshape_method))


if __name__ == '__main__':
    for class_name in tqdm(os.listdir(base_dir)):
        proceed_class(class_name, reshape_method, resample_method)
