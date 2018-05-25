# -*- coding: utf-8 -*-
import re
import uuid

from odoo import _, api, fields, models, modules, tools
from odoo.exceptions import UserError
from odoo.osv import expression
from odoo.tools import ormcache
from odoo.tools.safe_eval import safe_eval


class YoloImage(models.Model):
    _name = 'yolo.image'
    _rec_name = 'path'

    path = fields.Char(
        "Image Path", required=True,
        help="Absolute path of image")
    result = fields.Char("Result")
    src_image = fields.Binary(
        "Source Image", compute='get_src_image', attachment=True, store=True,
        help="Original image, limited to 1024x1024px.")
    result_image = fields.Binary(
        "Photo", compute='_get_src_image', attachment=True, store=True,
        help="Original image, limited to 1024x1024px.")

    @api.depends('path')
    def get_src_image(self):
        for rec in self:
            print "... rec.path", rec.path
            rec.src_image = tools.image_resize_image_big(
                open(rec.path, 'rb').read().encode('base64'))
