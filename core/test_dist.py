__author__ = 'christian'

# IMPORTS
import json
import numpy
import scipy as sp
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac

from utility import plotting_methods

# MODELS TO LOAD
model_names = ["alien", "elder", "elf", "engineer", "explorer", "fighter", "gargoyle", "gorilla", "merman", "monster", "ogre", "sage", "man"]

models_limits = {
        "alien":    2216,
        "elder":    3119,
        "elf":      4307,
        "engineer":  987,
        "explorer": 1858,
        "fighter":  1608,
        "gargoyle": 1058,
        "gorilla":  2719,
        "man":      1580,
        "merman":   2619,
        "monster": 967,
        "ogre": 1720,
        "sage":     2136
    }

model_names = ["monster", "gargoyle"]

model_tech = {"alien": " (subd)",
              "elder": " (subd)",
              "elf": " (re-topo)",
              "engineer": " (subd)",
              "explorer": " (re-topo)",
              "fighter": " (subd)",
              "gargoyle": " (re-topo)",
              "gorilla": " (re-topo)",
              "merman": " (subd)",
              "monster": " (re-topo)",
              "ogre": " (subd)",
              "sage": " (subd)",
              "man": " (subd)"}

# VARIABLES INITIALISATION
root_directory = "/Users/christian/Desktop/Ph.D./sculptAnalysis/"


brush_data_file = {}
brush_data = {}

path_points = {}
path_lenghts = {}
path_centroids = {}

obb_points = {}
obb_volumes = {}
obb_areas = {}

aabb_points = {}
aabb_volumes = {}

int_labels = {}
int_b_labels = {}

sizes = {}
unp_sizes = {}
modes = {}


final_datas = plotting_methods.load_data(model_names, root_directory)


# # MODELS LIST
#
# | ID | Model name 	| Modeling technique 	|
# |----|------------	|--------------------	|
# | 1  | alien      	| subd               	|
# | 2  | elder      	| subd               	|
# | 3  | elf        	| retopology         	|
# | 4  | engineer   	| subd               	|
# | 5  | explorer   	| retopology         	|
# | 6  | fighter    	| subd               	|
# | 7  | gargoyle   	| retopology         	|
# | 8  | gorilla    	| retopology         	|
# | 9  | man        	| subd               	|
# | 10 | merman     	| subd               	|
# | 11 | monster    	| retopology         	|
# | 12 | ogre       	| subd               	|
# | 13 | sage       	| subd               	|


def normalize(array):
    norm = numpy.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm

pos_idx = 1
gpr = 1
fig = plt.figure(figsize=(24, 6 * len(model_names)))
for model_name in model_names:
    lenght_data, lenght_labels = plotting_methods.filter_brush_attribute(final_datas[model_name], "lenghts", False)
    proj_size_data, proj_size_labels = plotting_methods.filter_brush_attribute(final_datas[model_name], "size", False)
    pressure_data, pressure_labels = plotting_methods.filter_brush_attribute(final_datas[model_name], "pressure", False)
    p_size = []
    for proj, unpr in proj_size_data:
        p_size.append(unpr)
    pressure = []
    for press_list in pressure_data:
        pressure.append(numpy.mean(numpy.array(press_list)))


    ret1 = plotting_methods.draw_plot(fig, normalize(numpy.array(lenght_data)), lenght_labels, model_names, model_tech, model_name, pos_idx, gpr, bar_color="r")
    ret1.set_yscale("log")

    ret2 = plotting_methods.draw_plot(fig, normalize(numpy.array(pressure)), pressure_labels, model_names, model_tech, model_name, pos_idx, gpr, bar_color="g")
    ret2.set_yscale("log")

    ret3 = plotting_methods.draw_plot(fig, normalize(numpy.array(p_size)), proj_size_labels, model_names, model_tech, model_name, pos_idx, gpr, bar_color="b")
    ret3.set_yscale("log")

    prod = [0.0, ] * len(lenght_data)
    for k in range(len(lenght_data)):
        prod[k] = p_size[k] * lenght_data[k] * pressure[k]

    #_ = plotting_methods.draw_plot(fig, normalize(numpy.array(prod)), proj_size_labels, model_names, model_tech, model_name, pos_idx, gpr, bar_color="y")

    dist_directory = "/Users/christian/Desktop/Ph.D./sculptAnalysis/steps/"+ model_name +"/distance_data.json"
    fh = open(dist_directory, 'r')
    dist_json = json.load(fh)

    distances = []
    dist_labels = []
    for k in range(1, len(dist_json)):
        dist_labels.append(k)
        distances.append(dist_json[str(k)])

    ret4 = plotting_methods.draw_plot(fig, normalize(numpy.array(distances)), dist_labels, model_names, model_tech, model_name, pos_idx, gpr, bar_color="k")
    ret4.set_yscale("linear")


    step = float(models_limits[model_name]) / 3.0
    curr_step = 1
    ret4.set_xlim([curr_step * step, (curr_step + 1) * step])

    pos_idx += 1
plt.show()



