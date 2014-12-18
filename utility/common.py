__author__ = 'christian'


from os import listdir, makedirs
from os.path import isfile, join, exists
import json
import pickle


def get_files_from_directory(dir_path, filters=[], start_with=None):
    if not filters:
        if not start_with:
            onlyfiles = [[join(dir_path, f), f]
                            for f in listdir(dir_path)
                            if isfile(join(dir_path, f))]
        else:
            onlyfiles = [[join(dir_path, f), f]
                            for f in listdir(dir_path)
                            if (isfile(join(dir_path, f)) and f.startswith(start_with))]
    else:
        if not start_with:
            onlyfiles = [[join(dir_path, f), f]
                    for f in listdir(dir_path)
                    if (isfile(join(dir_path, f)) and f.split('.')[1] in filters)]
        else:
            onlyfiles = [[join(dir_path, f), f]
                    for f in listdir(dir_path)
                    if (isfile(join(dir_path, f)) and f.split('.')[1] in filters and f.startswith(start_with))]

    return onlyfiles

def make_dirs(path):
    if not exists(path):
        makedirs(path)

def load_json(file_path):
    f = open(file_path, 'r')
    data = json.load(f)
    f.close()
    return data


def save_json(json_data, filepath, compressed=False):
    out = open(filepath, "w")
    if compressed:
        json.dump(json_data, out)
    else:
        json.dump(json_data, out, sort_keys=True, indent=2)
    out.close()


def load_pickle(file_path):
    fh = open(file_path, 'rb')
    data = pickle.loads(fh.read())
    return  data

def save_pickle(file_path, data):
    fh = open(file_path, "wb+")
    pickle.dump(data, fh)
    fh.close()


def load_obj(file_path):
    fh = open(file_path, 'r')
    f_v_lines = []
    f_n_lines = []
    f_f_lines = []
    for line in fh:
        if line.startswith('v '):
            f_v_lines.append(line)
        elif line.startswith('vn '):
            f_n_lines.append(line)
        elif line.startswith('f '):
            f_f_lines.append(line)
    fh.close()
    return f_v_lines, f_n_lines, f_f_lines