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

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

