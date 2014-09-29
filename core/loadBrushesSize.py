__author__ = 'christian'

import bpy
import json
from utility import common

def load_brush_size_from_blend(path):
    bpy.ops.wm.open_mainfile(filepath=path,
                             filter_blender=True,
                             filemode=8,
                             display_type='FILE_DEFAULTDISPLAY',
                             load_ui=False,
                             use_scripts=True)
    try:
        if bpy.data.scenes["Scene"]:
           return (bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.size,
                   bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.unprojected_radius)
    except KeyError as e:
            print('modifier not found')

def load_brushes_size(path):
    blend_files = common.get_files_from_directory(path, ['blend'])
    brush_sizes = []
    for file in blend_files:
        bpy.ops.wm.open_mainfile(filepath=file[0],
                                 filter_blender=True,
                                 filemode=8,
                                 display_type='FILE_DEFAULTDISPLAY',
                                 load_ui=False,
                                 use_scripts=True)

        try:
            if bpy.data.scenes["Scene"]:
                brush_sizes.append([bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.size,
                                    bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.unprojected_radius])
        except KeyError as e:
            print('modifier not found')
    for el in brush_sizes:
        print(el)
if __name__ == "__main__":
    load_brushes_size("../blend_files/task01/")