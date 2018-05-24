# -*- coding: utf-8 -*-
from odoo import models, fields

class YoloImage(models.Model):
    _name = 'yolo.image'

    path = fields.Char(
        "Image Path", required=True,
        help="Absolute path of image")
    result = fields.Char("Result")