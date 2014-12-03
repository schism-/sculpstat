__author__ = 'christian'

import bpy
from math import *
from mathutils import *
import struct
import shutil
from utility import common

class CameraData(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.root_blend_files = "/Volumes/PART FAT/3ddata/"
        self.blend_files = common.get_files_from_directory(self.root_blend_files + self.model_name + "/", ['blend'], "snap")
        self.blend_files.sort()

    def get_camera(self):
        a_viewport = [a for a in bpy.data.window_managers[0].windows[0].screen.areas if a.type == 'VIEW_3D']
        if not a_viewport:
            # likely a snapshot while trying to save...  hack!
            try:
                bpy.ops.screen.back_to_previous()
            except Exception:
                try:
                    bpy.context.area.type = 'VIEW_3D'
                except Exception:
                    return None
            a_viewport = [a for a in bpy.data.window_managers[0].windows[0].screen.areas if a.type == 'VIEW_3D']
        a_viewport = a_viewport[0]
        r3d_viewport = a_viewport.spaces[0].region_3d

        o = r3d_viewport.view_location
        x, y, z = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
        x.rotate(r3d_viewport.view_rotation)
        y.rotate(r3d_viewport.view_rotation)
        z.rotate(r3d_viewport.view_rotation)

        return {
            'o': list(o),
            'x': list(x), 'y': list(y), 'z': list(z),
            'd': r3d_viewport.view_distance,
            'p': r3d_viewport.view_perspective,
            'q': r3d_viewport.view_rotation,
        }

    def extract_data(self):
        ret_data = {}
        i = 0
        for file in self.blend_files[:200]:
            bpy.ops.wm.open_mainfile(filepath=file[0],
                                     filter_blender=True,
                                     filemode=8,
                                     display_type='FILE_DEFAULTDISPLAY',
                                     load_ui=False,
                                     use_scripts=True)
            ret_data[i] = self.get_camera()
            i += 1
        return ret_data



if __name__ == "__main__":
    cd = CameraData("fighter")
    cameras = cd.extract_data()

    for c in cameras:
        print(cameras[c])