__author__ = 'christian'

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

def get_2d_angles(brush_data):
    path_points = np.array(brush_data["brush_2d_pos"])
    angles = []
    for k in range(len(path_points)-2):
        v0 = np.array(path_points[k+1]) - np.array(path_points[k])
        v1 = np.array(path_points[k+2]) - np.array(path_points[k+1])
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

# 'SMOOTH', 'CLAY_STRIPS', 'PINCH', 'FLATTEN', 'MASK', 'GRAB', 'INFLATE', 'DRAW', 'LAYER', 'CREASE', 'SCRAPE', 'BLOB', 'CLAY', 'SNAKE_HOOK'

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

'''
scaling_factors = {
    "alien":1.0,
    "elder":1.0,
    "elf":1.0,
    "engineer":1.0,
    "explorer":1.0,
    "fighter":1.0,
    "gargoyle":1.0,
    "gorilla":1.0,
    "man":1.0,
    "merman":1.0,
    "monster":1.0,
    "ogre":1.0,
    "sage":1.0
}

scaling_factors = {
    "alien":           4.88411,
    "elder":           18.1198,
    "elf":             5.62413,
    "engineer":        2.03795,
    "explorer":        17.2768,
    "fighter":         2.0678,
    "gorilla":         12.3074,
    "gargoyle":        10.5912,
    "man":             4.38392,
    "merman":          4.90317,
    "monster":         6.87775,
    "sage":            4.91853,
    "ogre":            20.0907
}
'''

#scaling_factors = common.load_json("/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/bounding_spheres.json")

# blender bboxes
'''
scaling_factors = {
    "alien": [2.070, 3.958, 1.975],
    "elder": [11.029, 11.677, 8.387],
    "elf": [ 3.153, 4.497, 1.194 ],
    "engineer": [0.890, 1.804, 0.328 ],
    "explorer": [6.090, 15.633, 4.126 ],
    "fighter": [0.890, 1.842, 0.301 ],
    "gargoyle": [8.490, 2.693, 5.731 ],
    "gorilla": [5.898, 8.698, 6.405 ],
    "man": [2.160, 3.360, 1.806 ],
    "merman": [1.971, 3.272, 3.074 ],
    "monster": [4.184, 3.945, 3.632 ],
    "ogre": [12.015, 11.846, 10.906],
    "sage": [2.715, 3.547, 2.060 ]
}
'''

# MAX EXTENT
'''
scaling_factors = {
    "alien": 3.958,
    "elder": 11.677,
    "elf": 4.497,
    "engineer": 1.804,
    "explorer": 15.633,
    "fighter": 1.842,
    "gargoyle": 8.490,
    "gorilla": 8.698,
    "man": 3.360,
    "merman": 3.272,
    "monster": 4.184,
    "ogre": 12.015,
    "sage": 3.547
}
'''

# EYE BOUNDING SPHERE
scaling_factors = {
    "alien": 0.172,
    "elder": 1.433,
    "elf": 0.155,
    "engineer": 0.049,
    "explorer": 0.252,
    "fighter": 0.039,
    "gargoyle": 0.324,
    "gorilla": 0.376,
    "man": 0.181,
    "merman": 0.441,
    "monster": 0.742,
    "ogre": 1.751,
    "sage": 0.171
}

for model in model_names:
    print("analyzing model %s" % model)
    fd = common.load_json("/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/complete/" + model + "/final_data_3.json")

    brush_features_intro[model] = {}
    for pre in ["on", "gl", "mask"]:
        brush_features_intro[model][pre + "_brush_count"] =  0
        # path lenght info
        brush_features_intro[model][pre + "_lenghts"] =      []
        brush_features_intro[model][pre + "_avg_lenght"] =   0.0
        brush_features_intro[model][pre + "_std_lenght"] =   0.0
        # 2D path lenght info
        brush_features_intro[model][pre + "_2d_lenghts"] =      []
        brush_features_intro[model][pre + "_avg_2d_lenght"] =   0.0
        brush_features_intro[model][pre + "_std_2d_lenght"] =   0.0
        # brush unprojected size info
        brush_features_intro[model][pre + "_unproj"] =       []
        brush_features_intro[model][pre + "_avg_unproj"] =   0.0
        brush_features_intro[model][pre + "_std_unproj"] =   0.0
        # brush pressure info
        brush_features_intro[model][pre + "_pressures"] =    []
        brush_features_intro[model][pre + "_avg_pressure"] = 0.0
        brush_features_intro[model][pre + "_std_pressure"] = 0.0
        # haussdorf distance info
        brush_features_intro[model][pre + "_hauss"] =        []
        brush_features_intro[model][pre + "_avg_hauss"] =    0.0
        brush_features_intro[model][pre + "_std_hauss"] =    0.0
        # brush mode info
        brush_features_intro[model][pre + "_normal"] =       0
        brush_features_intro[model][pre + "_invert"] =       0
        brush_features_intro[model][pre + "_smooth"] =       0
        # oriented bounding box info
        brush_features_intro[model][pre + "_obboxes"] =      []
        brush_features_intro[model][pre + "_avg_obbox"] =    0.0
        brush_features_intro[model][pre + "_std_obbox"] =    0.0
        # angle between polyline segments info
        brush_features_intro[model][pre + "_angles"] =       []
        brush_features_intro[model][pre + "_avg_angle"] =    0.0
        brush_features_intro[model][pre + "_std_angle"] =    0.0
        # angle between 2D polyline segments info
        brush_features_intro[model][pre + "_2d_angles"] =       []
        brush_features_intro[model][pre + "_avg_2d_angle"] =    0.0
        brush_features_intro[model][pre + "_std_2d_angle"] =    0.0

    k = 0
    max_fd = len(fd)
    for step in fd:
        print("step " + str(k) + "/" + str(max_fd)),
        k += 1

        if fd[step]["valid"]:
            bd = fd[step]["brush_data"]
            dd = fd[step]["distance_data"]

            if bd["brush_type"] in on_surface:
                prefix = "on"
            elif bd["brush_type"] in global_brush:
                prefix = "gl"
            elif bd["brush_type"] in mask_brush:
                prefix = "mask"

            brush_features_intro[model][prefix + "_brush_count"] += 1
            brush_features_intro[model][prefix + "_lenghts"].append(float(bd["lenghts"][0]) / scaling_factors[model])
            brush_features_intro[model][prefix + "_2d_lenghts"].append(float(bd["lenght_2d"]))
            brush_features_intro[model][prefix + "_unproj"].append(float(bd["size"][0][1]) / scaling_factors[model])
            brush_features_intro[model][prefix + "_pressures"].append(bd["pressure"][0])


            if dd["distance_mean"]:
                brush_features_intro[model][prefix + "_hauss"].append(float(dd["distance_mean"]) / scaling_factors[model])
            else:
                brush_features_intro[model][prefix + "_hauss"].append(0.0)

            if bd["mode"][0] == "NORMAL":
                brush_features_intro[model][prefix + "_normal"] += 1
            elif bd["mode"][0] == "INVERT":
                brush_features_intro[model][prefix + "_invert"] += 1
            elif bd["mode"][0] == "SMOOTH":
                brush_features_intro[model][prefix + "_smooth"] += 1

            brush_features_intro[model][prefix + "_obboxes"].append(float(get_max_diagonal(bd)) / scaling_factors[model])
            brush_features_intro[model][prefix + "_angles"].append(get_angles(bd))
            brush_features_intro[model][prefix + "_2d_angles"].append(get_2d_angles(bd))
print()
for model in model_names:
    print("second pass for " + model)
    for prefix in ["on", "gl", "mask"]:
        print("prefix " + prefix)
        if prefix + "_lenghts" in brush_features_intro[model]:
            brush_features_intro[model][prefix + "_avg_lenght"] = np.mean(brush_features_intro[model][prefix + "_lenghts"])
            brush_features_intro[model][prefix + "_std_lenght"] = np.std(brush_features_intro[model][prefix + "_lenghts"])
            del brush_features_intro[model][prefix + "_lenghts"]

            brush_features_intro[model][prefix + "_avg_2d_lenght"] = np.mean(brush_features_intro[model][prefix + "_2d_lenghts"])
            brush_features_intro[model][prefix + "_std_2d_lenght"] = np.std(brush_features_intro[model][prefix + "_2d_lenghts"])
            del brush_features_intro[model][prefix + "_2d_lenghts"]

            brush_features_intro[model][prefix + "_avg_unproj"] = np.mean(brush_features_intro[model][prefix + "_unproj"])
            brush_features_intro[model][prefix + "_std_unproj"] = np.std(brush_features_intro[model][prefix + "_unproj"])
            del brush_features_intro[model][prefix + "_unproj"]

            all_pressures = []
            for el in brush_features_intro[model][prefix + "_pressures"]:
                all_pressures += el
            brush_features_intro[model][prefix + "_avg_pressure"] = np.mean(all_pressures)
            brush_features_intro[model][prefix + "_std_pressure"] = np.std(all_pressures)
            del brush_features_intro[model][prefix + "_pressures"]

            brush_features_intro[model][prefix + "_avg_hauss"] = np.mean(brush_features_intro[model][prefix + "_hauss"])
            brush_features_intro[model][prefix + "_std_hauss"] = np.std(brush_features_intro[model][prefix + "_hauss"])
            del brush_features_intro[model][prefix + "_hauss"]

            brush_features_intro[model][prefix + "_avg_obbox"] = np.mean(brush_features_intro[model][prefix + "_obboxes"])
            brush_features_intro[model][prefix + "_std_obbox"] = np.std(brush_features_intro[model][prefix + "_obboxes"])
            del brush_features_intro[model][prefix + "_obboxes"]

            all_angles = []
            for el in brush_features_intro[model][prefix + "_angles"]:
                all_angles += el
            brush_features_intro[model][prefix + "_avg_angle"] = np.mean(all_angles)
            brush_features_intro[model][prefix + "_std_angle"] = np.std(all_angles)
            del brush_features_intro[model][prefix + "_angles"]

            all_2d_angles = []
            for el in brush_features_intro[model][prefix + "_2d_angles"]:
                all_2d_angles += el
            brush_features_intro[model][prefix + "_avg_2d_angle"] = np.mean(all_2d_angles)
            brush_features_intro[model][prefix + "_std_2d_angle"] = np.std(all_2d_angles)
            del brush_features_intro[model][prefix + "_2d_angles"]

print("saving")
fh = open("/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/brushes_introduction_eye.json", "w")
json.dump(brush_features_intro, fh, sort_keys=True, indent=2)
print("saved")

