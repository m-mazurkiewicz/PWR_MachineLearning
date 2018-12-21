import sys
import os
from tqdm import tqdm
import shutil
import numpy as np

base_dir = '/content/gdrive/My Drive/PWr_AlexNet_data/interim/'
base_dir_processed = '/content/gdrive/My Drive/PWr_AlexNet_data/processed/'

set = sys.argv[1]

def split(class_name,ratio, target_dir):
    train_dir = target_dir + 'train/' + class_name + '/'
    test_dir = target_dir + 'test/' + class_name + '/'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    list = os.listdir(base_dir+set+'/'+class_name)
    number_files = len(list)
    test_files = np.random.choice(range(number_files), size=int(np.ceil(ratio*number_files)), replace=False, )
    for key,image in enumerate(list):
        if key in test_files:
            shutil.copyfile(base_dir+set+'/'+class_name+'/'+image, test_dir+image)
        else:
            shutil.copyfile(base_dir + set + '/' + class_name + '/' + image, train_dir + image)

for class_name in tqdm(os.listdir(base_dir+set)):
    split(class_name, 0.2, base_dir_processed + set)