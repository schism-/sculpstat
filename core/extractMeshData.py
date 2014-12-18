__author__ = 'christian'

from utility import common
import os
import json

class MeshData(object):

    def __init__(self, model_name, obj_path = "../obj2_files/", steps_path = "../steps/"):
        self.model_name = model_name
        self.obj_path = obj_path + self.model_name + "/"
        self.mesh_no = len(common.get_files_from_directory(self.obj_path, ["obj"], start_with="snap"))
        self.steps_path = steps_path


    def extract_data_from_mesh(self, mesh_path):
        f1 = open(mesh_path, 'r')
        f1_v_lines = []
        f1_n_lines = []
        f1_f_lines = []

        for line in f1:
            if line.startswith('v '):
                f1_v_lines.append(line)
            elif line.startswith('vn '):
                f1_n_lines.append(line)
            elif line.startswith('f '):
                f1_f_lines.append(line)
        f1.close()

        data = {}
        data["vertices_no"] = len(f1_v_lines)
        data["normals_no"] = len(f1_n_lines)
        data["faces_no"] = len(f1_f_lines)

        return data


    def extract_data(self):
        seq_data = {}
        for mesh_idx in range(self.mesh_no):
            seq_data[mesh_idx] = self.extract_data_from_mesh(self.obj_path + "/snap" + str(mesh_idx).zfill(6) + ".obj")
            print("Model %s step %d -- %s" % (self.model_name, mesh_idx, seq_data[mesh_idx]))

        if not os.path.exists(self.steps_path + self.model_name + "/"):
            os.makedirs(self.steps_path + self.model_name + "/")
        out = open(self.steps_path + self.model_name + "/mesh_data.json", "w")
        json.dump(seq_data, out)
        out.close()


if __name__ == "__main__":
    '''
        Input data is in the form
            ["alien", 1024, 2216, 1]
            [model name, start step, end step, stride]
            - end step might at maximum be (number of snaps - 1)

        ["alien",       0,      2216,   1],
        ["elder",       0,      3119,   1],
        ["elf",         0,      4307,   1],
        ["engineer",    0,       987,   1],
        ["explorer",    0,      1858,   1],
        ["fighter",     0,      1608,   1],
        ["gargoyle",    0,      1058,   1],
        ["gorilla",     0,      2719,   1],
        ["man",         0,      1580,   1],
        ["merman",      0,      2619,   1],
        ["monster",     0,       967,   1],
        ["ogre",        0,      1720,   1],
        ["sage",        0,      2136,   1],
    '''

    model_names = [
        "sage"
    ]

    for mn in model_names:
        #md = MeshData(mn, obj_path="/Volumes/Part Mac/obj_smooth_normals_files/")
        md = MeshData(mn, obj_path="/Users/christian/Desktop/obj_smooth_normals_files/")
        mesh_data = md.extract_data()