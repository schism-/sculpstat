__author__ = 'christian'

'''
    == attributes to scatter ==

    average length
	avg unproj size
	avg pressure
	avg haus distance
	diag obbox
	avg of angles
	avg of angles_2d

'''

import pandas as pd
import matplotlib.pyplot as plt
import json
import os.path
import numpy as np
from utility import common

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle

def get_angles(brush_data):
    path_points = np.array(brush_data["paths"][0])
    angles = []
    for k in range(len(path_points)-2):
        v0 = np.array(path_points[k+1]) - np.array(path_points[k])
        v1 = np.array(path_points[k+2]) - np.array(path_points[k+1])
        angles.append(angle_between(v0, v1))
    return angles

def get_2d_angles(brush_data, filtered=False):
    angles = []
    if filtered:
        f = []
        for el in brush_data["brush_2d_pos"]:
            el = tuple(el)
            if el not in f:
                f.append(el)
        path_points = np.array(f)
    else:
        path_points = np.array(brush_data["brush_2d_pos"])

    if len(path_points) >= 3:
        for k in range(len(path_points)-2):
            p0 = path_points[k]
            p1 = path_points[k+1]
            p2 = path_points[k+2]

            v0 = np.array(p1) - np.array(p0)
            v1 = np.array(p2) - np.array(p1)

            angles.append(angle_between(v0, v1))
    return angles

def get_max_diagonal(brush_data):
    obb_center = np.array(brush_data["obboxes"][0][0])
    obb_extent = np.array(brush_data["obboxes"][0][1])
    p1= obb_center - obb_extent
    p2= obb_center + obb_extent
    return np.linalg.norm(p2-p1)

# ========================================================================
#
# ========================================================================

model_names = [
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
    "sage"
]

all_b_type = {}
brush_features_intro = {}

on_surface = ['BLOB',
              'CLAY',
              'CLAY_STRIPS',
              'CREASE',
              'DRAW',
              'FLATTEN',
              'INFLATE',
              'LAYER',
              'PINCH',
              'SCRAPE',
              'SMOOTH']

global_brush = ['GRAB',
                'SNAKE_HOOK']

mask_brush = ['MASK']

lengths = {}
sizes = {}
pressures = {}
distances = {}
obboxes = {}
angles = {}
angles_2d = {}
angles_2d_f = {}
colors = {}

for model in model_names:
    print("analyzing model %s" % model)
    fd = json.load(open("/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/complete/" + model + "/final_data_3.json", "r"))

    lengths[model] = []
    sizes[model] = []
    pressures[model] = []
    distances[model] = []
    obboxes[model] = []
    angles[model] = []
    angles_2d[model] = []
    angles_2d_f[model] = []
    colors[model] = []

    k = 0
    max_fd = len(fd)
    for step in fd:
        print("step " + str(k) + "/" + str(max_fd)),
        k += 1

        if fd[step]["valid"]:
            bd = fd[step]["brush_data"]
            dd = fd[step]["distance_data"]

            if bd["brush_type"] in on_surface:
                colors[model].append(1)
            elif bd["brush_type"] in global_brush:
                colors[model].append(2)
            elif bd["brush_type"] in mask_brush:
                colors[model].append(3)

            lengths[model].append(bd["lenghts"][0])

            sizes[model].append(bd["size"][0][1])

            if bd["pressure"][0]:
                pressures[model].append(np.mean(bd["pressure"][0]))
            else:
                pressures[model].append(0.0)

            if dd["distance_mean"]:
                distances[model].append(dd["distance_mean"])
            else:
                distances[model].append(0.0)

            obboxes[model].append(get_max_diagonal(bd))

            ang = get_angles(bd)
            if ang:
                angles[model].append(np.mean(ang))
            else:
                angles[model].append(0.0)

            ang2 = get_2d_angles(bd)
            if ang2:
                angles_2d[model].append(np.mean(ang2))
            else:
                angles_2d[model].append(0.0)

            ang2f = get_2d_angles(bd, True)
            if ang2f:
                angles_2d_f[model].append(np.mean(ang2f))
            else:
                angles_2d_f[model].append(0.0)


common.save_json(lengths, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_lengths.json", compressed=False)
common.save_json(sizes, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_sizes.json", compressed=False)
common.save_json(pressures, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_pressures.json", compressed=False)
common.save_json(distances, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_distances.json", compressed=False)
common.save_json(angles, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_angles.json", compressed=False)
common.save_json(angles_2d, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_angles_2d.json", compressed=False)
common.save_json(angles_2d_f, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_angles_2d_f.json", compressed=False)
common.save_json(obboxes, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_obboxes.json", compressed=False)
common.save_json(colors, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/scatter_colors.json", compressed=False)
