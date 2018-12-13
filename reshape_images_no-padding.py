import os
from PIL import Image
from resizeimage import resizeimage


def reshape_image(image_path, image_name):
    with open(image_path+image_name, 'r+b') as f:
        with Image.open(f) as image:
            img = resizeimage.resize_cover(image, [227, 277])
            img.save('test-image-cover.jpeg', img.format)




for subdirectory in os.listdir('/content/gdrive/My Drive/PWr_AlexNet_data/raw/'):
    print(subdirectory)