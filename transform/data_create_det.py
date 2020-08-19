# Author:Lin Peng
# Date:2020.08.03

import json
import os
import random
import shutil
from math import ceil


def main():
    output_dir = './train_data/invoice_det/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    label_dir = './doc/imgs/'
    all_files = os.listdir(label_dir)
    json_files = []
    img_files = []
    for filename in all_files:
        if filename.split('.')[-1] == 'json':
            json_files.append(filename)
        else:
            img_files.append(filename)

    index = list(range(len(json_files)))
    random.seed(1)
    random.shuffle(index)  # Random sample
    train_num = ceil(len(json_files)*4.0/5)
    train_json_files = []
    train_img_files = []
    for i in index[:train_num]:
        train_json_files.append(json_files[i])  # Take 4/5 of the total dataset for training
        for img in img_files:
            if json_files[i].split('.')[0] == img.split('.')[0]:
                train_img_files.append(img)
    test_json_files = []
    test_img_files = []
    for i in index[train_num:]:
        test_json_files.append(json_files[i])  # Take the rest of dataset for testing
        for img in img_files:
            if json_files[i].split('.')[0] == img.split('.')[0]:
                test_img_files.append(img)

    if not os.path.exists(os.path.join(output_dir, 'train_data')):
        os.mkdir(os.path.join(output_dir, 'train_data'))
    target_train = os.path.join(output_dir, 'train_data')
    for img_name in train_img_files:
        shutil.copy(os.path.join(label_dir, img_name), target_train)

    f_train = open(os.path.join(output_dir, 'train_label.txt'), "w", encoding='utf8')
    for filename in train_json_files:
        with open(os.path.join(label_dir, filename), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            tmp_list = []
            for label in json_data['shapes']:
                tmp_dic = {'transcription': label['label'], 'points': label['points']}
                # tmp_json = json.dumps(tmp_dic, ensure_ascii=False)
                tmp_list.append(tmp_dic)
            tmp_json = json.dumps(tmp_list, ensure_ascii=False)
            for img in train_img_files:
                if img.find(filename.split('.')[0]) == 0:
                    suffix = img.split('.')[1]
                    break
            f_train.write('train_data/' + filename.split('.')[0] + '.' + suffix + '\t' + tmp_json + '\n')
    f_train.close()

    if not os.path.exists(os.path.join(output_dir, 'test_data')):
        os.mkdir(os.path.join(output_dir, 'test_data'))
    target_test = os.path.join(output_dir, 'test_data')
    for img_name in test_img_files:
        shutil.copy(os.path.join(label_dir, img_name), target_test)

    f_test = open(os.path.join(output_dir, 'test_label.txt'), "w", encoding='utf8')
    for filename in test_json_files:
        with open(os.path.join(label_dir, filename), 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            tmp_list = []
            for label in json_data['shapes']:
                tmp_dic = {'transcription': label['label'], 'points': label['points']}
                # tmp_json = json.dumps(tmp_dic, ensure_ascii=False)
                tmp_list.append(tmp_dic)
            tmp_json = json.dumps(tmp_list, ensure_ascii=False)
            for img in train_img_files:
                if img.find(filename.split('.')[0]) == 0:
                    suffix = img.split('.')[1]
                    break
            f_test.write('test_data/' + filename.split('.')[0] + '.' + suffix + '\t' + tmp_json + '\n')
    f_test.close()


if __name__ == '__main__':
    main()
