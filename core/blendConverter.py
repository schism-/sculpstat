import bpy

from os import listdir
from os.path import isfile, join

from utility import common

blend_files_path = "blend_files/gargoyle2"
obj_files_path = "obj_files/gargoyle2"

onlyfiles = common.get_files_from_directory(blend_files_path, ['blend'])


def views(window):
    rtn = []
    for a in window.screen.areas:
        if a.type == 'VIEW_3D':
            rtn.append(a)
    return rtn


def camera(view):
    print(len(view.spaces))
    look_at = view.spaces[0].region_3d.view_location
    matrix = view.spaces[0].region_3d.view_matrix
    camera_pos = camera_position(matrix)
    rotation = view.spaces[0].region_3d.view_rotation
    return look_at, camera_pos, rotation


def camera_position(matrix):
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


for file in onlyfiles:
    bpy.ops.wm.open_mainfile(filepath=file[0],
                             filter_blender=True,
                             filemode=8,
                             display_type='FILE_DEFAULTDISPLAY',
                             load_ui=False,
                             use_scripts=True)

    #bpy.data.objects["Plane"].modifiers["Multires"].levels = bpy.data.objects["Plane"].modifiers["Multires"].sculpt_levels

    #for v in views(bpy.data.window_managers['WinMan'].windows[0]):
    #    print(camera(v))

    print("==========================================================")

    bpy.ops.export_scene.obj(filepath= obj_files_path + "/" + file[1].split('.')[0] + ".obj",
                             axis_forward='-Z',
                             axis_up='Y')

    #bpy.ops.export_scene.autodesk_3ds(filepath= obj_files_path + "/" + file[1].split('.')[0] + ".3ds")