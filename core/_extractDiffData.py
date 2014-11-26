__author__ = 'christian'

import utility.common as common
import pickle
import json

class DiffData(object):

    def __init__(self, model_name, start, end):
        self.model_name = model_name
        self.diff_files_path = "../diff_new/" + self.model_name + "/step_1"
        self.start = start
        self.end = end
        self.diff_data = {}
        self.diff_files = common.get_files_from_directory(self.diff_files_path, None, "diff")

    def get_diff_data(self):
        for file in self.diff_files:
            print(file)
            diff_no = int(file[1].split('_')[1])
            if start < diff_no < end:
                fh = open(file[0], 'rb')
                data = pickle.load(fh)
                if data["valid"]:
                    self.diff_data[diff_no] = {}
                    self.diff_data[diff_no]['mod_vertices'] = len(data['mod_vertices']) if 'mod_vertices' in data else 0
                    self.diff_data[diff_no]['new_vertices'] = len(data['new_vertices']) if 'new_vertices' in data else 0
                    self.diff_data[diff_no]['mod_normals'] = len(data['mod_normals']) if 'mod_normals' in data else 0
                    self.diff_data[diff_no]['new_normals'] = len(data['new_normals']) if 'new_normals' in data else 0
                    self.diff_data[diff_no]['mod_faces'] = len(data['mod_faces']) if 'mod_faces' in data else 0
                    self.diff_data[diff_no]['new_faces'] = len(data['new_faces']) if 'new_faces' in data else 0

if __name__ == "__main__":
    models = ["monster"]
    start = 0
    end = 100000

    for model_name in models:
        cd = DiffData(model_name, start, end)
        cd.get_diff_data()

        out = open("../steps/" + cd.model_name + "/diff_data.json", "w")
        json.dump(cd.diff_data, out)
        out.close()