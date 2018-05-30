# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import uuid


from odoo import _, api, fields, models, modules, tools
from odoo.exceptions import UserError
import base64
from PIL import Image
import os
import tempfile
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


class YoloImage(models.Model):
    _name = 'yolo.image'

    name = fields.Char(
        "Image Name", help="Name of image")
    result = fields.Char("Result")
    src_image = fields.Binary("Source Image", attachment=True)
    result_image = fields.Binary("Result Image", attachment=True)

    @api.multi
    def detect_object(self):
        for rec in self:
            if not rec.src_image:
                raise UserError("no image on this record")
            # decode the base64 encoded data
            data = base64.decodestring(rec.src_image)
            # create a temporary file, and save the image
            fobj = tempfile.NamedTemporaryFile(delete=False)
            fname = fobj.name
            fobj.write(data)
            fobj.close()
            # open the image with PIL
            try:
                rec.result = detect(net, meta, fname)
            finally:
                # delete the file when done
                os.unlink(fname)
