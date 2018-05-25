# -*- coding: utf-8 -*-

from __future__ import print_function
from odoo import models, fields, api
import base64
from PIL import Image
import os
from collections import defaultdict
from itertools import product
from sklearn.model_selection import train_test_split
import shutil
import re
import glob
from scipy import ndimage
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
from six.moves import range
import sys
from ctypes import *
import math
import random
from box import BOX
from image import IMAGE
from metadata import METADATA
from detection import DETECTION


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

def get_setting_file():
    subpath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.abspath(os.path.join(subpath, os.pardir))
    path = path + "/setting.conf"
    return path

def get_setting():
    subpath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.abspath(os.path.join(subpath, os.pardir))
    path = path + "/setting.conf"
    with open(path) as file:
        lines = [line.rstrip('\n') for line in file]
    d = {}
    for line in lines:
        t = line.split(':')
        d[t[0]] = t[1]
    return d

setting_dict = get_setting()
libdarknet_path = setting_dict['libdarknet_path']
yolo_config_path = setting_dict['yolo_config_path']
yolo_weight_path = setting_dict['yolo_weight_path']
meta_path = setting_dict['meta_path']

lib = CDLL(libdarknet_path, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

net = load_net(yolo_config_path, yolo_weight_path, 0)
meta = load_meta(meta_path)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

class YoloDetector(models.Model):
    _name = 'yolo.detector'

    name = fields.Char("Detector Name", required=True)
    directory = fields.Char("Directory", required=True)

    @api.multi
    def detect_object(self):
        image_env = self.env['yolo.image']
        for rec in self:
            images = []
            for file in os.listdir(rec.directory):
                if file.endswith(".jpg") or file.endswith(".png"):
                    images.append(os.path.join(rec.directory, file))
            for img in images:
                r = detect(net, meta, img)
                print (r)
                vals = {
                    'path': img,
                    'result': r
                }
                image_env.create(vals)
