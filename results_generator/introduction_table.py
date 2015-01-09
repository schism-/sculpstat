__author__ = 'christian'

'''

| Model name 	| Modeling technique 	| Artist    |
|------------	|--------------------	|--------   |
| alien      	| subd               	|   rr      |
| elder      	| subd               	|   rr      |
| elf        	| retopology         	|   rr      |
| engineer   	| subd               	|   jw      |
| explorer   	| retopology         	|   rr      |
| fighter    	| subd               	|   rr      |
| gargoyle   	| retopology         	|   jw      |
| gorilla    	| retopology         	|   rr      |
| man        	| subd               	|   rr      |
| merman     	| subd               	|   jw      |
| monster    	| retopology         	|   jw      |
| ogre       	| subd               	|   jw      |
| sage       	| subd               	|   jw      |

- tabella dati raw (per ogni modello):
	- nome modello
	- numero finale facce modelli;
	- subd or re-topo;
	- "nome" artista (A or B);
	- number of strokes;
	- numero cambi camera, numero zoom

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import common

data_dir = "../steps/"
final_data_dir = "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/complete/"

header = "Model Name, Number of faces, Technique, Artist, Number of strokes, Number of camera changes\n"

data = {
    "alien" :   {"technique": "subd", "artist": "rr"},
    "elder" :   {"technique": "subd", "artist": "rr"},
    "elf" :     {"technique": "reto", "artist": "rr"},
    "engineer" :{"technique": "subd", "artist": "jw"},
    "explorer" :{"technique": "reto", "artist": "rr"},
    "fighter" : {"technique": "subd", "artist": "rr"},
    "gargoyle" :{"technique": "reto", "artist": "jw"},
    "gorilla" : {"technique": "reto", "artist": "rr"},
    "man" :     {"technique": "subd", "artist": "rr"},
    "merman" :  {"technique": "subd", "artist": "jw"},
    "monster" : {"technique": "reto", "artist": "jw"},
    "ogre" :    {"technique": "subd", "artist": "jw"},
    "sage" :    {"technique": "subd", "artist": "jw"}
}

brush_types = {}

for model_name in data:
    mesh_data = common.load_json(data_dir + model_name + "/mesh_data.json")
    max_step_num = str(len(mesh_data.keys()) - 1)
    data[model_name]["faces_no"] = mesh_data[max_step_num]["faces_no"]

    final_data = common.load_json(final_data_dir + model_name + "/final_data_3.json")
    brush_no = 0
    for step in final_data:
        if final_data[step]["brush_data"]["valid"]:
            brush_no += final_data[step]["brush_data"]["brush_number"]
            if final_data[step]["brush_data"]["brush_type"] in brush_types:
                brush_types[final_data[step]["brush_data"]["brush_type"]] += 1
            else:
                brush_types[final_data[step]["brush_data"]["brush_type"]] = 1
    data[model_name]["brush_no"] = brush_no

    camera_data = common.load_json(data_dir + model_name + "/camera_movements.json")
    camera_mov = 0
    for step in camera_data:
        camera_mov += len(camera_data[step])
    data[model_name]["camera_movements"] = camera_mov

    #data[model_name]["model_name"] = model_name


for line in data:
    print(line, " ")
    print(data[line])

common.save_json(data, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/introduction_table.json", compressed=False)
common.save_json(brush_types, "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/ipython/data/all_brushes_type.json", compressed=False)

a = pd.DataFrame(data)

print(a.T)

print(brush_types)