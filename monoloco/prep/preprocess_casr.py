import pickle
import re
import json
import os
import glob
import datetime
import numpy as np
import torch

from .. import __version__
from ..network.process import preprocess_monoloco

gt_path = 'data/casr/annotations/casr_annotation.pickle'
res_path = '/scratch/izar/beauvill/casr/res_extended/casr*'

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_bboxes(bbox_gt, bbox_pred):
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]

    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bb_intersection_over_union(bbox_gt[i,:], bbox_pred[j,:])

    return np.argmax(iou_matrix)

def standard_bbox(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def load_gt():
    return pickle.load(open(gt_path, 'rb'), encoding='latin1')

def load_res(path=res_path):
    mono = []
    for folder in sorted(glob.glob(path), key=lambda x:float(re.findall(r"(\d+)",x)[0])):
        data_list = []
        for file in sorted(os.listdir(folder), key=lambda x:float(re.findall(r"(\d+)",x)[0])):
            if 'json' in file:
                json_path = os.path.join(folder, file)
                json_data = json.load(open(json_path))
                json_data['filename'] = json_path
                data_list.append(json_data)
        mono.append(data_list)
    return mono

def create_dic(dir_ann=res_path, std=False):
    gt=load_gt()
    res=load_res(dir_ann)
    dic_jo = {
        'train': dict(X=[], Y=[], names=[], kps=[]),
        'val': dict(X=[], Y=[], names=[], kps=[]),
        'version': __version__,
    }
    split = ['3', '4']
    if std:
        wrong = [6, 8, 9, 10, 11, 12, 14, 21, 40, 43, 55, 70, 76, 92, 109,
                 110, 112, 113, 121, 123, 124, 127, 128, 134, 136, 139, 165, 173]
        mode = 'std'
    else:
        wrong = []
        mode = ''
    for i in [x for x in range(len(res[:])) if x not in wrong]:
        for j in [x for x in range(len(res[i][:])) if 'boxes' in res[i][x]]:
            folder = gt[i][j]['video_folder']

            phase = 'val'
            if folder[7] in split:
                phase = 'train'

            gt_box = gt[i][j]['bbox_gt']

            good_idx = match_bboxes(np.array([standard_bbox(gt_box)]), np.array(res[i][j]['boxes'])[:,:4])

            keypoints = [res[i][j]['uv_kps'][good_idx]]

            gt_turn = gt[i][j]['left_or_right']
            if std and gt_turn == 3:
                gt_turn = 2
            inp = preprocess_monoloco(keypoints, torch.eye(3)).view(-1).tolist()
            dic_jo[phase]['kps'].append(keypoints)
            dic_jo[phase]['X'].append(inp)
            dic_jo[phase]['Y'].append(gt_turn)
            dic_jo[phase]['names'].append(folder+"_frame{}".format(j))

    now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")[2:]
    with open("data/casr/outputs/joints-casr-" + mode + "-right-" +
              split[0] + split[1] + "-" + now_time + ".json", 'w') as file:
        json.dump(dic_jo, file)
    return dic_jo
