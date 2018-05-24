# -*- coding: utf-8 -*-
from odoo import models, fields

class YoloDetector(models.Model):
    _name = 'yolo.detector'

    name = fields.Char("Detector Name", required=True)
    directory = fields.Char("Directory", required=True)