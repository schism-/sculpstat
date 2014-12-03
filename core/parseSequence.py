__author__ = 'christian'

import re
import json
import time
import bpy
import numpy as np
from utility import common


debug = False


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


def parse_single_step_to_json(line_data):
    no_parse = ['initial',
                'interface',
                'bpy.ops.ed.undo',
                'bpy.ops.file.execute',
                'bpy.ops.file.highlight',
                'bpy.ops.file.select',
                'bpy.ops.file.select_bookmark',
                'bpy.ops.object.editmode_toggle',
                'bpy.ops.object.mode_set',
                'bpy.ops.paint.mask_flood_fill',
                'bpy.ops.screen.actionzone',
                'bpy.ops.screen.region_scale',
                'bpy.ops.sculpt.dynamic_topology_toggle',
                'bpy.ops.sculpt.sculptmode_toggle',
                'bpy.ops.view2d.pan',
                'bpy.ops.view3d.cursor3d',
                'bpy.ops.view3d.smoothview',
                'bpy.ops.view3d.layers',
                'bpy.ops.view3d.move',
                'bpy.ops.view3d.properties',
                'bpy.ops.view3d.rotate',
                'bpy.ops.view3d.viewnumpad',
                'bpy.ops.view3d.view_all',
                'bpy.ops.view3d.view_orbit',
                'bpy.ops.view3d.view_persportho',
                'bpy.ops.view3d.zoom',
                'bpy.ops.wm.save_as_mainfile']

    parse = ['bpy.ops.sculpt.brush_stroke',
             'bpy.ops.wm.radial_control',
             'bpy.ops.paint.brush_select',
             'bpy.ops.view3d.move',
             'bpy.ops.view3d.rotate',
             'bpy.ops.view3d.zoom',
             'bpy.ops.view3d.properties',
             'bpy.ops.view3d.view_orbit',
             'bpy.ops.ed.undo']

    argument_label_re_pattern = "([a-zA-z]+)="
    arguments_list_pattern = "\((.+)\)"

    if line_data[3].split('(')[0] in parse:
        string_form = ' '.join(line_data[3:])
        if debug:
            print(string_form)

        arg_labels = re.findall(argument_label_re_pattern, string_form)
        if debug:
            print(arg_labels)
            print()

        json_string_form = re.sub(argument_label_re_pattern, r'"\1":', string_form)
        if debug:
            print(json_string_form)
            print()

        arg_list = re.findall(arguments_list_pattern, json_string_form)
        if debug:
            print(arg_list)
            print()

        if arg_list:
            json_string_form = '{' + arg_list[0] + '}'
            json_string_form = json_string_form.replace("'", '"')
            json_string_form = json_string_form.replace("False", '"False"')
            json_string_form = json_string_form.replace("True", '"True"')
            json_string_form = json_string_form.replace("(", '[')
            json_string_form = json_string_form.replace(")", ']')
            if debug:
                print(json_string_form)
        else:
            json_string_form = '{}'

        try:
            json_form = json.loads(json_string_form)

            json_form['op_name'] = line_data[3].split('(')[0]

            if debug:
                print(json_form)

            return json_form
        except ValueError as e:
            print("++++++++++++++ PARSING ERROR +++++++++++++++++")
            print(json_string_form)
            print(e)
            print("++++++++++++++++++++++++++++++++++++++++++++++")

        if debug:
            print('=================================================')

def parse_file(files_path):
    f = open(files_path, 'r')
    data = {}
    current_step = 0
    for line in f:
        line_data = line.split(' ')
        if not line_data[0].startswith('-'):
            current_step = int(line_data[0])

        # {op, op_time, op_date}
        if not current_step in data:
            data[current_step] = []

        data[current_step].append(parse_single_step_to_json(line_data))

        #parse additional data
        #camera_data = get_camera_data(model_name, current_step)
        #data[current_step].append(camera_data)
    return data

def get_camera_data(model_name, stepno):
    bpy.ops.wm.open_mainfile(filepath=file[0],
                             filter_blender=True,
                             filemode=8,
                             display_type='FILE_DEFAULTDISPLAY',
                             load_ui=False,
                             use_scripts=True)

    for v in views(bpy.data.window_managers['WinMan'].windows[0]):
        print(camera(v))

def filter_data(data):
    filtered_data = {}
    for key, step in data.items():
        for op in data[key]:
            if op is not None:
                if key in filtered_data:
                    filtered_data[key].append(op)
                else:
                    filtered_data[key] = [op]
    return filtered_data

def get_different_ops(steps_files_path):
    f = open(steps_files_path, 'r')

    op_set = set()

    for line in f:
        line_data = line.split(' ')
        op_data = line_data[3].split('(')
        op = op_data[0]
        op_set.add(op)

    print("Found " + str(len(op_set)) + " different ops")
    for el in op_set:
        print(el)

def cluster_data(filtered_data, steps):
    clustered_data = {}
    for d in filtered_data:
        if d // steps not in clustered_data:
            clustered_data[d // steps] = filtered_data[d]
        else:
            clustered_data[d // steps] += filtered_data[d]

    return clustered_data




if __name__ == "__main__":

    models = [
        "alien",
        "elder",
        "elf",
        "engineer",
        "explorer",
        "fighter",
        "gargoyle",
        "gorilla",
        "man",
        "merman",
        "monster",
        "ogre",
        "sage"]

    for model_name in models:
        steps_files_path = "../steps/" + model_name + "/steps.txt"
        steps = 1

        start = time.time()
        final_data = parse_file(steps_files_path)
        end = time.time()
        print("Took %f seconds" % (end - start))

        filtered_data = filter_data(final_data)

        # clustered_data = cluster_data(filtered_data, steps)
        #for step in clustered_data:
        #    print(step)
        #    for op in clustered_data[step]:
        #        print(str(op)[:100])
        # out = open("../steps/" + model_name + "/steps_clust"+ str(steps) +".json", "w")

        out = open("../steps/" + model_name + "/steps.json", "w")
        json.dump(filtered_data, out, sort_keys=True, indent=2)
        out.close()