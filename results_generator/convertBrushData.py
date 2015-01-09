__author__ = 'christian'

import json
from utility import common

#brush_dir = "/Users/christian/Desktop/Ph.D./sculptAnalysis/steps/"
brush_dir = "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/complete/"

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
              'SMOOTH',
              'MASK']

global_brush = ['GRAB',
                'SNAKE_HOOK']

for model_name in model_names:
    print("Analyzing model " + model_name)
    fd_json = common.load_json(brush_dir + model_name + "/final_data_3.json")

    final_list = []
    final_list_flipped = []
    for k in range(len(fd_json)):
        print(str(k) + "/" + str(len(fd_json)), )
        bd_json = fd_json[str(k)]["brush_data"]
        if bd_json["valid"]:
            points = bd_json["paths"][0]
            flattened_points = []
            flattened_points_flipped = []
            flattened_indices = []
            for idx, p in enumerate(points):
                flattened_points += p
                flattened_points_flipped += [-p[0], p[1], p[2]]
                if idx < len(points) - 1:
                    flattened_indices.append(idx)
                    flattened_indices.append(idx+1)

            final_list.append(
                {
                    "pos": flattened_points,
                    "line": flattened_indices,
                    "category": "ON_SURFACE" if bd_json["brush_type"] in on_surface else "GLOBAL",
                    "type": bd_json["brush_type"]
                }
            )

            final_list_flipped.append(
                {
                    "pos": flattened_points_flipped,
                    "line": flattened_indices,
                    "category": "ON_SURFACE" if bd_json["brush_type"] in on_surface else "GLOBAL",
                    "type": bd_json["brush_type"]
                }
            )

        else:
            final_list.append(
                {
                    "pos": [9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0],
                    "line": [0, 1],
                    "category": "ON_SURFACE",
                    "type": "CLAY"
                }
            )

            final_list_flipped.append(
                {
                    "pos": [9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0],
                    "line": [0, 1],
                    "category": "ON_SURFACE",
                    "type": "CLAY"
                }
            )


    common.save_json(final_list, brush_dir + model_name + "/" + model_name + "_flattened_strokes.json", compressed=False)
    common.save_json(final_list_flipped, brush_dir + model_name + "/" + model_name + "_flattened_strokes_flipped.json", compressed=False)