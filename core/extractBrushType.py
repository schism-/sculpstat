__author__ = 'christian'

import os
import bpy
import json
import pickle
import numpy
import numpy.linalg
import scipy.stats as scs
from utility import common

class BrushData(object):

    def __init__(self, model_name, max_step=0):
        self.model_name = model_name
        self.step_path = "../steps/" + self.model_name + "/steps.json"
        self.root_blend_files = "/Volumes/PART FAT/3ddata/"
        self.paths = []
        self.max_step = max_step
        f = open(self.step_path, 'r')
        self.steps = json.load(f)

    def load_brush_type(self, step_no):
        blend_file = self.root_blend_files + self.model_name + "/snap" + str(step_no).zfill(6) + ".blend"

        bpy.ops.wm.open_mainfile(filepath=blend_file, filter_blender=True,
                                 filemode=8, display_type='FILE_DEFAULTDISPLAY',
                                 load_ui=False, use_scripts=True)
        brush_type = "DRAW"
        try:
            if bpy.data.scenes["Scene"]:
                brush_type = bpy.context.tool_settings.sculpt.brush.sculpt_tool
        except KeyError:
            print('brush not found')

        return brush_type



if __name__ == "__main__":
    models = [
        ["alien",    2216],
        ["elder",    3119],
        ["elf",      4307],
        ["engineer",  987],
        ["explorer", 1858],
        ["fighter",  1608],
        ["gargoyle", 1058],
        ["gorilla",  2719],
        ["man",      1580],
        ["merman",   2619],
        ["monster", 967],
        ["ogre", 1720],
        ["sage",     2136]
    ]
    '''
    for model_name, max_step in models:
        print("saving data for " + model_name)
        bd = BrushData(model_name, max_step)
        type_json = {}
        for k in range(max_step + 1):
            type_json[str(k)] = bd.load_brush_type(k)
        common.save_json(type_json, "../steps/" + model_name + "/brush_type_new.json")
    '''
    for model_name, step in models:
        bt = common.load_json("../steps/" + model_name + "/brush_type_new.json")
        bd = BrushData(model_name, step)
        type = bd.load_brush_type(step)
        bt[str(step)] = type
        common.save_json(bt, "../steps/" + model_name + "/brush_type_new.json")
