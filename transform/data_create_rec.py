# Author:Lin Peng
# Date:2020.08.04

import json
import os
import random
import shutil
from math import ceil


def main():
    output_dir = './train_data/handwritten_dataset/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    image_dir = './doc/data/'
    all_files = os.listdir(image_dir)
    txt_files = []
    img_files = []
    for filename in all_files:
        if filename.split('.')[-1] == 'txt':
            txt_files.append(filename)
        else:
            img_files.append(filename)
    index = list(range(len(txt_files)))
    random.seed(1)
    random.shuffle(index)  # Random sample
    train_num = ceil(len(txt_files)*4.0/5)
    train_txt_files = []
    train_img_files = []
    for i in index[:train_num]:
        train_txt_files.append(txt_files[i])  # Take 4/5 of the total dataset for training
        train_img_files.append(img_files[i])
    test_txt_files = []
    test_img_files = []
    for i in index[train_num:]:
        test_txt_files.append(txt_files[i])  # Take the rest of dataset for testing
        test_img_files.append(img_files[i])

    if not os.path.exists(os.path.join(output_dir, 'train_data')):
        os.mkdir(os.path.join(output_dir, 'train_data'))
    target_train = os.path.join(output_dir, 'train_data')
    for img_name in train_img_files:
        shutil.copy(os.path.join(image_dir, img_name), target_train)

    f_train = open(os.path.join(output_dir, 'train_label.txt'), "w")
    for filename in train_txt_files:
        with open(os.path.join(image_dir, filename), 'r', encoding='utf8') as fp:
            line = fp.readline()
            f_train.write('train_data/' + filename.split('.')[0] + '.jpg\t' + line + '\n')
    f_train.close()

    if not os.path.exists(os.path.join(output_dir, 'test_data')):
        os.mkdir(os.path.join(output_dir, 'test_data'))
    target_test = os.path.join(output_dir, 'test_data')
    for img_name in test_img_files:
        shutil.copy(os.path.join(image_dir, img_name), target_test)

    f_test = open(os.path.join(output_dir, 'test_label.txt'), "w")
    for filename in test_txt_files:
        with open(os.path.join(image_dir, filename), 'r', encoding='utf8') as fp:
            line = fp.readline()
            f_test.write('test_data/' + filename.split('.')[0] + '.jpg\t' + line + '\n')
    f_test.close()


if __name__ == '__main__':
    main()
