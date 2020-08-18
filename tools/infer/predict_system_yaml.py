# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import sys
import codecs
from pathlib import Path
import yaml
import nltk

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger
logger = initial_logger()
import cv2
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import copy
import numpy as np
import math
import time
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from PIL import Image
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
            (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_templates(template_dir):
    template_file_paths = Path(template_dir).glob('*.yaml')
    templates = []

    for template_file_path in template_file_paths:
        with codecs.open(template_file_path, 'r', encoding='utf-8') as file_stream:
            template = yaml.load(file_stream, Loader=Loader)
            # content = json.load(file_stream)
            templates.append(template)

    return templates



def replace_text_format(text_format):
    pattern = re.compile(r'\((?P<name>\$.+?):(?P<type>.+?)\)')
    sub_text = pattern.sub('', text_format)
    pattern = re.compile(r'(?P<name>\(.+?(\|.+?)+\))')
    matches = pattern.finditer(sub_text)

    curr_texts = ['']
    curr_text_len = 0

    for m in matches:
        group_str = m.group('name')
        start_index, end_index = m.span()

        choice_texts = group_str[1: len(group_str) - 1].split('|')

        updated_texts = []

        for curr_text in curr_texts:
            for choice_text in choice_texts:
                update_text = curr_text + sub_text[curr_text_len: start_index] + choice_text
                updated_texts.append(update_text)

        curr_texts = updated_texts
        curr_text_len = end_index

    updated_texts = []

    for curr_text in curr_texts:
        update_text = curr_text + sub_text[curr_text_len:]
        updated_texts.append(update_text)

    curr_texts = updated_texts

    return curr_texts




def find_chinese_text(text_str):
    results = []

    for char_txt in text_str:
        if '\u4e00' <= char_txt and char_txt <= '\u9fff':
            results.append(char_txt)

    chinese_text = ''.join(results)
    return chinese_text



def text_match(templates, dt_boxes, rec_res, drop_score=0.5):
    match_results = []
    dt_num = len(dt_boxes)

    for template in templates:
        match_result_dict = {}
        partitions = template['partitions']
        image_type = template.get('image_type')
        image_type_name = image_type.get('name')
        image_head_ref = image_type.get('head_ref')

        for dno in range(dt_num):
            dt_box = dt_boxes[dno]
            text_str, score = rec_res[dno]
            if score < drop_score:
                continue
            
            # text_score_str = "%s, %.3f" % (text_str, score)
            # print(text_score_str, dt_box.shape)
            assert dt_box.shape == (4, 2), dt_box.shape
            mean_point = np.average(dt_box, axis=0)



            for partition in partitions:
                partition_type = partition.get('type')
                partition_id = partition.get('id')
                is_list = partition.get('is_list')
                text_type = partition.get('text_type')
                text_format = partition.get('text_format')
                # points = partition.get('points')
                # x1, y1 = points[0] 
                # x2, y2 = points[1]
                # x_mean = (x1 + x2) / 2
                # y_mean = (y1 + y2) / 2
                # mean_point = [x_mean, y_mean]

                shape_type = partition.get('shape_type')
                match_rules = partition.get('match_rules')

                assert partition_type in ['head', 'text'], partition_type
                texts = []
                if partition_type == 'head':
                    text = partition.get('text')
                    texts.append(text)
                else: 
                    assert partition_type == 'text', partition_type
                    text_format = partition.get('text_format')
                    texts = replace_text_format(text_format)

                match_result_dict.setdefault(partition_type, {})
                match_dict = match_result_dict[partition_type]
                match_dict.setdefault(partition_id, [])
                results = match_dict[partition_id]

                match_result = None
                
                for match_rule in match_rules:
                    # print(match_rule)
                    accept_threshold = match_rule.get('accept_threshold')
                    accept_threshold = float(accept_threshold)
                    head_ref = match_rule.get('head_ref')
                    position_to_ref = match_rule.get('position_to_ref')

                    text_compare_method = match_rule.get('text_compare_method')

                    if text_compare_method:
                        if text_compare_method == 'string.edit_distance':
                            chinese_text_str = find_chinese_text(text_str)

                            for text in texts:
                                if not (chinese_text_str and text):
                                    continue

                                # if '笔行程' in text_str: 
                                #     print('Compare: ', text_str, text)
                                edit_dist = nltk.edit_distance(chinese_text_str, text)
                                compare_score = 1.0 - edit_dist * 1.0 / max(len(chinese_text_str), len(text))

                                if compare_score >= accept_threshold:
                                    # print('Matched:', chinese_text_str, text, compare_score)
                                    curr_result = {'text': text_str, 'points': dt_box, 'mean_point': mean_point, 'score': compare_score}

                                    if not match_result or compare_score > match_result['score']:
                                        match_result = curr_result

                                    # match_result_dict.setdefault(partition_type, {})
                                    # match_dict = match_result_dict[partition_type]
                                    # match_dict.setdefault(partition_id, [])
                                    # results = match_dict[partition_id]

                                    # if is_list:
                                    #     results.append(curr_result)
                                    # else:
                                    #     if not results:
                                    #         results.append(curr_result)
                                    #     elif compare_score > results[0]['score']:
                                    #         results[0] = curr_result

                        else:
                            assert text_compare_method == 're.match', match_rule

                            if re.match(text_format, text_str):
                                curr_result = {'text': text_str, 'points': dt_box, 'mean_point': mean_point, 'score': accept_threshold}

                                if not match_result or compare_score > match_result['score']:
                                    match_result = curr_result
                                    
                if not match_result:
                    continue
                
                results.append(match_result)

        head_result_dict = match_result_dict.get('head', {})

        if image_head_ref in head_result_dict:
            match_result_dict['image_type'] = image_type_name
        
        match_results.append(match_result_dict)

    return match_results


def remove_invalid_result(results, points):
    for result_index in range(len(results)):
        if np.array_equal(results[result_index]['points'], points):
            print('Remove invalid result:', results[result_index])
            del results[result_index]

            return


def get_normed_point_diff(points):
    point1, point2, point3, point4 = points
    point_diff = point2 + point3 - point1 - point4
    norm_value = np.linalg.norm(point_diff)
    if not np.any(norm_value):
        return point_diff

    point_diff = point_diff / norm_value
    return point_diff


def get_cos(points1, points2):
    point_diff1 = get_normed_point_diff(points1)
    point_diff2 = get_normed_point_diff(points2)

    return np.abs(np.dot(point_diff1, point_diff2))




def remove_invalid_results(templates, match_results, dt_boxes, rec_res, drop_score=0.5):
    dt_num = len(dt_boxes)

    for template, match_result_dict in zip(templates, match_results):
        partitions = template['partitions']
        image_type = template.get('image_type')
        image_type_name = image_type.get('name')
        image_head_ref = image_type.get('head_ref')
        head_result_dict = match_result_dict.get('head', {})

        for dno in range(dt_num):
            dt_box = dt_boxes[dno]
            text_str, score = rec_res[dno]
            if score < drop_score:
                continue
            
            # text_score_str = "%s, %.3f" % (text_str, score)
            # print(text_score_str, dt_box.shape)
            assert dt_box.shape == (4, 2), dt_box.shape
            mean_point = np.average(dt_box, axis=0)

            for partition in partitions:
                partition_type = partition.get('type')
                partition_id = partition.get('id')
                is_list = partition.get('is_list')
                text_type = partition.get('text_type')
                text_format = partition.get('text_format')
                # points = partition.get('points')
                # x1, y1 = points[0] 
                # x2, y2 = points[1]
                # x_mean = (x1 + x2) / 2
                # y_mean = (y1 + y2 / 2)
                # mean_point = [x_mean, y_mean]

                shape_type = partition.get('shape_type')
                match_rules = partition.get('match_rules')

                assert partition_type in ['head', 'text'], partition_type
                texts = []
                if partition_type == 'head':
                    text = partition.get('text')
                    texts.append(text)
                else: 
                    assert partition_type == 'text', partition_type
                    text_format = partition.get('text_format')
                    texts = replace_text_format(text_format)

                match_dict = match_result_dict.get(partition_type)
                results = match_dict.get(partition_id)
                
                for match_rule in match_rules:
                    # print(match_rule)
                    accept_threshold = match_rule.get('accept_threshold')
                    accept_threshold = float(accept_threshold)
                    head_ref = match_rule.get('head_ref')
                    position_to_ref = match_rule.get('position_to_ref')

                    text_compare_method = match_rule.get('text_compare_method')

                    if not (head_ref and position_to_ref):
                        continue

                    if not (head_result_dict and head_ref in head_result_dict):
                        continue

                    head_ref_results = head_result_dict[head_ref]


                    if not head_ref_results:
                        continue

                    for head_ref_result in head_ref_results:
                        head_ref_points = head_ref_result['points']
                        head_ref_mean_point = head_ref_result['mean_point']

                        is_remove = False
                        
                        if position_to_ref == 'left' and mean_point[0] >= head_ref_mean_point[0]:
                            is_remove = True
                            # print('head: ', head_ref_result)

                        if position_to_ref == 'right' and mean_point[0] <= head_ref_mean_point[0]:
                            is_remove = True
                            # print('head: ', head_ref_result)

                        if position_to_ref == 'above' and mean_point[1] >= head_ref_mean_point[1]:
                            is_remove = True
                            # print('head: ', head_ref_result)

                        if position_to_ref == 'below' and mean_point[1] <= head_ref_mean_point[1]:
                            is_remove = True
                            # print('head: ', head_ref_result)

                        cos_value = get_cos(dt_box, head_ref_points)

                        if cos_value < np.cos(np.pi/6.):
                            # print('cos_value:', cos_value)
                            is_remove = True

                        if is_remove:
                            remove_invalid_result(results, dt_box) 



def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    templates = get_templates(args.template_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    tackle_img_num = 0
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        tackle_img_num += 1
        if not args.use_gpu and args.enable_mkldnn and tackle_img_num % 30 == 0:
            text_sys = TextSystem(args)
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))
        drop_score = 0.5

        match_results = text_match(templates, dt_boxes, rec_res, drop_score=drop_score)
        remove_invalid_results(templates, match_results, dt_boxes, rec_res, drop_score=drop_score)

        ocr_results = []

        for match_result in match_results:
            ocr_result = {}

            for partition_type, match_dict in match_result.items():
                ocr_dict = {}

                if partition_type  == 'head':
                    continue

                if partition_type == 'image_type':
                    ocr_dict[partition_type] = match_dict
                    continue

                for partition_id, results in match_dict.items():
                    for result in results:
                        ocr_dict.setdefault(partition_id, [])
                        ocr_patition_dict = {
                            'text': result['text'], 
                            # 'points': result['points'].tolist(), 
                            'confidence': result['score']
                        }
                        
                        ocr_dict[partition_id].append(ocr_patition_dict)

                ocr_result[partition_type] = ocr_dict


            ocr_results.append(ocr_result)
            
        print('ocr_results:', ocr_results)
        
        # dt_num = len(dt_boxes)
        # for dno in range(dt_num):
        #     dt_box = dt_boxes[dno]
        #     text, score = rec_res[dno]
        #     if score >= drop_score:
        #         text_str = "%s, %.3f" % (text, score)
        #         print(text_str, dt_box)

        #         post_process(templates, text, dt_box)

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr(
                image,
                boxes,
                txts,
                scores,
                draw_txt=True,
                drop_score=drop_score)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))


if __name__ == "__main__":
    main(utility.parse_args())
