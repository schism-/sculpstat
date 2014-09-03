import numpy as np
import re
import json
import time

from utility import common

debug = False

def parse_single_step_to_json(line_data):
    no_parse = ['bpy.ops.sculpt.dynamic_topology_toggle',
                'bpy.ops.view3d.view_persportho',
                'bpy.ops.object.editmode_toggle',
                'bpy.ops.view3d.smoothview',
                'bpy.ops.screen.actionzone',
                'bpy.ops.view3d.view_all',
                'interface',
                'bpy.ops.view3d.move',
                'bpy.ops.file.execute',
                'bpy.ops.sculpt.sculptmode_toggle',
                'bpy.ops.file.select',
                'bpy.ops.view3d.properties',
                'bpy.ops.view3d.layers',
                'bpy.ops.screen.region_scale',
                'bpy.ops.view3d.viewnumpad',
                'bpy.ops.view3d.view_orbit',
                'bpy.ops.wm.save_as_mainfile',
                'bpy.ops.file.highlight',
                'bpy.ops.ed.undo',
                'bpy.ops.file.select_bookmark',
                'bpy.ops.view3d.rotate',
                'bpy.ops.view3d.cursor3d',
                'bpy.ops.view2d.pan',
                'initial',
                'bpy.ops.object.mode_set',
                'bpy.ops.paint.mask_flood_fill',
                'bpy.ops.view3d.zoom']

    parse = ['bpy.ops.sculpt.brush_stroke',
             'bpy.ops.wm.radial_control',
             'bpy.ops.paint.brush_select']

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

        json_string_form = '{' + arg_list[0] + '}'
        json_string_form = json_string_form.replace("'", '"')
        json_string_form = json_string_form.replace("False", '"False"')
        json_string_form = json_string_form.replace("True", '"True"')
        json_string_form = json_string_form.replace("(", '[')
        json_string_form = json_string_form.replace(")", ']')
        if debug:
            print(json_string_form)

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

    return data

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



if __name__ == "__main__":
    steps_files_path = "../steps/gargoyle2/steps.txt"

    start = time.time()

    final_data = parse_file(steps_files_path)

    end = time.time()

    print("Took %f seconds" % (end - start))

    filtered_data = {}
    for key, step in final_data.items():
        for op in final_data[key]:
            if op is not None:
                if key in filtered_data:
                    filtered_data[key].append(op)
                else:
                    filtered_data[key] = [op]

    for step in filtered_data:
        print(step)
        for op in filtered_data[step]:
            print(str(op)[:100])

    out = open("../steps/gargoyle2/steps.json", "w")
    json.dump(filtered_data, out, indent=4, separators=(',', ': '))
    out.close()

    #get_different_ops(steps_files_path)
