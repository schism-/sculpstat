__author__ = 'christian'

import bpy
import utility.common as common

class CameraData(object):

    def __init__(self, model_name, start, end):
        self.blend_files_path = "../blend_files/" + model_name
        self.obj_files_path = "../obj_files/" + model_name
        self.start = start
        self.end = end
        self.blend_files = common.get_files_from_directory(self.blend_files_path, ['blend'], "snap")

    def views(self, window):
        rtn = []
        for a in window.screen.areas:
            if a.type == 'VIEW_3D':
                rtn.append(a)
        return rtn


    def camera(self, view):
        look_at = view.spaces[0].region_3d.view_location
        matrix = view.spaces[0].region_3d.view_matrix
        camera_pos = self.camera_position(matrix)
        rotation = view.spaces[0].region_3d.view_rotation
        return look_at, camera_pos, rotation


    def camera_position(self, matrix):
        """ From 4x4 matrix, calculate camera location """
        t = (matrix[0][3], matrix[1][3], matrix[2][3])
        r = (
            (matrix[0][0], matrix[0][1], matrix[0][2]),
            (matrix[1][0], matrix[1][1], matrix[1][2]),
            (matrix[2][0], matrix[2][1], matrix[2][2])
        )
        rp = (
            (-r[0][0], -r[1][0], -r[2][0]),
            (-r[0][1], -r[1][1], -r[2][1]),
            (-r[0][2], -r[1][2], -r[2][2])
        )
        output = (
            rp[0][0] * t[0] + rp[0][1] * t[1] + rp[0][2] * t[2],
            rp[1][0] * t[0] + rp[1][1] * t[1] + rp[1][2] * t[2],
            rp[2][0] * t[0] + rp[2][1] * t[1] + rp[2][2] * t[2],
        )
        return output

    def get_camera_data(self):
        camera_data = {}
        k = 0
        for file in self.blend_files:
            if int((file[1].split('.')[0])[4:]) < self.end:
                bpy.ops.wm.open_mainfile(filepath=file[0],
                                         filter_blender=True,
                                         filemode=8,
                                         display_type='FILE_DEFAULTDISPLAY',
                                         load_ui=False,
                                         use_scripts=True)

                print(bpy.data.cameras)
                for shit in bpy.data.cameras:
                    print(shit)

                for win in bpy.data.window_managers['WinMan'].windows:
                    for v in self.views(win):
                        if k not in camera_data:
                            camera_data[k] = [self.camera(v)]
                        else:
                            camera_data[k].append(self.camera(v))
                k += 1
            else:
                break
        return camera_data

if __name__ == "__main__":
    model_name = "gargoyle2"
    start = 0
    end = 10

    cd = CameraData(model_name, start, end)
    camera_data = cd.get_camera_data()

    for el in camera_data:
        print(camera_data[el])
