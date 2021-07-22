# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     bbox_iou.py
   Description :
   Author :       jw
   date：          2021/7/22
-------------------------------------------------
   Change Activity:
                   2021/7/22:
-------------------------------------------------
"""
__author__ = 'dylan'

__OVERLAPS__ = ['IOU', 'GIOU', 'CIOU', 'DIOU']

import torch
import math


def bbox_iou(boxes1, boxes2, mode='IOU', eps=1e-7):
    assert mode in __OVERLAPS__, f'overlap compute can not support: {mode}'

    assert (boxes1.size(-1) == 4 or boxes1.size(0) == 0)
    assert (boxes2.size(-1) == 4 or boxes2.resize(0) == 0)

    assert boxes1.shape[:-2] == boxes1.shape[:-2]


    batch_size = boxes1.shape[:-2]

    # Union Area
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1]) + eps
    area2 = (boxes2[..., 2] - boxes2[..., 0] * (boxes2[..., 3] - boxes2[..., 1])) + eps

    lt = torch.max(boxes1[..., :2], boxes2[..., :2])
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    wh = (rb-lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1 + area2 - inter + eps

    iou = inter / union

    if mode == 'GIOU' or mode == 'DIOU' or mode == 'CIOU':
        enclosed_lt = torch.min(boxes1[..., :2], boxes2[..., :2])
        enclosed_rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        cw = enclosed_rb[..., 0] - enclosed_lt[..., 0]
        ch = enclosed_rb[..., 1] - enclosed_lt[..., 1]

        if mode == 'CIOU' or mode == 'DIOU':
            c2 = cw**2 + ch**2 + eps
            rho2 = ((boxes2[..., 2] + boxes2[..., 0] - boxes1[..., 2] - boxes1[..., 0])**2 + \
                  (boxes2[..., 3] + boxes2[..., 1] - boxes1[..., 3] - boxes1[..., 0])**2) / 4

            if mode == 'DIOU':
                return iou - rho2 / c2
            elif mode == 'CIOU':
                w1 = boxes1[..., 2] - boxes1[..., 0]
                h1 = boxes1[..., 3] - boxes1[..., 1]
                w2 = boxes2[..., 2] - boxes2[..., 0]
                h2 = boxes2[..., 3] - boxes2[..., 1]
                v = (4 / math.pi**2)*torch.pow(torch.atan(w2/h2) - torch.atan(w1/h1), 2)
                alpha = v / (v - iou + (1+eps))
                return iou - (rho2/c2 + v*alpha)
        else:
            c_area = ch**cw + eps
            return iou - (c_area - union) / c_area
    else:
        return iou


