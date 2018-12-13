import os
from PIL import Image
from tqdm import tqdm
import sys

base_dir = '/content/gdrive/My Drive/PWr_AlexNet_data/raw/'

reshape_method = sys.argv[1]
resample_method = sys.argv[2]
padding = int(sys.argv[3])

resample_methods_dict = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS}


def resize_image(image_path, image_name, resample_method, padding):
    with open(image_path + image_name, 'r+b') as f:
        with Image.open(f) as image:
            img = image.resize(size=[227-2*padding, 227-2*padding], resample=resample_method)
            new_im = Image.new(img.mode, (227, 227))
            new_im.paste(img, ((227 - img.size[0]) // 2,
                               (227 - img.size[1]) // 2))
            new_im.save(image_path.replace('raw/', 'interim/padding_{}/resize/'.format(padding)) + image_name, img.format)


def crop_image(image_path, image_name, resample_method, padding):
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
            img = img.resize(size=[227-2*padding, 227-2*padding], resample=resample_method)
            new_im = Image.new(img.mode, (227, 227))
            new_im.paste(img, ((227 - img.size[0]) // 2,
                              (227 - img.size[1]) // 2))
            new_im.save(image_path.replace('raw/', 'interim/padding_{}/crop/'.format(padding)) + image_name, image.format)


def proceed_class(class_name, reshape_method, resample_method, padding):
    if reshape_method == 'crop':
        os.makedirs((base_dir + class_name + '/').replace('raw/', 'interim/padding_{}/crop/'.format(padding)), exist_ok=True)
        for key,image in enumerate(os.listdir(base_dir + class_name)):
            crop_image(base_dir + class_name + '/', image, resample_methods_dict[resample_method], padding)
            if key % 100 == 0:
                print(key)
    elif reshape_method == 'resize':
        os.makedirs((base_dir + class_name + '/').replace('raw/', 'interim/padding_{}/resize/'.format(padding)), exist_ok=True)
        for key,image in enumerate(os.listdir(base_dir + class_name)):
            resize_image(base_dir + class_name + '/', image, resample_methods_dict[resample_method], padding)
            if key % 100 == 0:
                print(key)
    else:
        raise ValueError('No such reshape_method - \'{0}\' '.format(reshape_method))


if __name__ == '__main__':
    for class_name in tqdm(os.listdir(base_dir)):
        proceed_class(class_name, reshape_method, resample_method, padding)
