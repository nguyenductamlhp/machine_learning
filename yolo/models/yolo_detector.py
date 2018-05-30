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
from function import detect
from function import net
from function import meta


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
