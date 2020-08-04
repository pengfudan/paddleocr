# Author:Lin Peng
# Date:2020.08.03
import base64
import json
import os
import sys
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
from ppocr.utils.utility import get_image_file_list
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
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                                 np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                                  np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0],
                              [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    tackle_img_num = 0
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        # print(image_file)
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
        dt_num = len(dt_boxes)
        dt_boxes_final = []
        txts_final = []  # add
        for dno in range(dt_num):
            text, score = rec_res[dno]
            if score >= 0.5:
                text_str = "%s, %.3f" % (text, score)
                # print(text_str)
                dt_boxes_final.append(dt_boxes[dno])
                txts_final.append(rec_res[dno][0])  # add
        '''
        file_info_list = {}
        file_all = json.loads(json.dumps(file_info_list))
        file_list = []
        mark_out = []
        image_result = dict()
        image_result['fileUrl'] = './data/img'
        for i in range(len(txts_final)):
            label_dic = dict()
            label_dic['MarkLabel'] = txts_final[i]
            label_dic['Points'] = dt_boxes_final[i]
            mark_out.append(label_dic)
        image_result['MarkOut'] = mark_out
        file_list.append(image_result)
        file_all['fileInfoList'] = file_list
        with open("./data.json", "w", encoding='utf8') as f:
            json.dump(file_all, f, ensure_ascii=False, cls=NpEncoder, indent=4)
        '''
        image_info = {}
        image_label_info = json.loads(json.dumps(image_info))
        image_label_info['version'] = '4.5.6'
        image_label_info['flags'] = {}
        shapes = []
        # image_result['fileUrl'] = './data/img'
        for i in range(len(txts_final)):
            label_dic = dict()
            label_dic['label'] = txts_final[i]
            label_dic['points'] = dt_boxes_final[i]
            label_dic['group_id'] = 1
            label_dic['shape_type'] = 'polygon'
            label_dic['flags'] = {}
            shapes.append(label_dic)
        image_label_info['shapes'] = shapes
        img_name = os.path.basename(image_file)
        dir_name = './doc/imgs/'
        image_path = dir_name + img_name
        image_label_info['imagePath'] = img_name  # can be modified
        with open(image_path, 'rb') as f:
            base64_data = base64.b64encode(f.read())
            img_base64 = base64_data.decode()
        image_label_info['imageData'] = img_base64
        image_label_info['imageHeight'] = np.array(img).shape[0]
        image_label_info['imageWidth'] = np.array(img).shape[1]

        with open(dir_name + img_name.split('.')[0] + ".json", "w", encoding='utf8') as f:
            json.dump(image_label_info, f, ensure_ascii=False, cls=NpEncoder, indent=2)

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr(
                image, boxes, txts, scores, draw_txt=True, drop_score=0.5)
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
