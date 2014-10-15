__author__ = 'christian'

import time
import pickle
import os.path
import numpy as np

class DiffEntry(object):
    def __init__(self, key, data, set_name):
        self.key = key
        self.data = data
        self.other_key = None
        self.set_name = set_name

    def __eq__(self, other):
        if self.data == other.data:
            self.other_key = other.key
            other.other_key = self.key
            return True
        else:
            return False

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "%s Key: %s --- Data: %s --- Key from: %s" % (self.set_name, self.key, self.data, self.other_key)

    def __repr__(self):
        return 'DiffEntry(key=%s, data=%s, key_from=%s, set_name="%s")' % (self.key, self.data, self.other_key, self.set_name)

    def reset_other_key(self):
        self.other_key = None

def compute_diff(file_path, file1, file2, withUV= False):
    start_dumbdiff = time.time()
    f1 = open(file_path + "/snap" + str(file1).zfill(6) + ".obj", 'r')
    f2 = open(file_path + "/snap" + str(file2).zfill(6) + ".obj", 'r')

    f1_v_lines = [line.strip() for line in f1 if line.startswith('v ')]
    f2_v_lines = [line.strip() for line in f2 if line.startswith('v ')]
    f1.close()
    f2.close()

    f1 = open(file_path + "/snap" + str(file1).zfill(6) + ".obj", 'r')
    f2 = open(file_path + "/snap" + str(file2).zfill(6) + ".obj", 'r')

    f1_f_lines = [line.strip() for line in f1 if line.startswith('f ')]
    f2_f_lines = [line.strip() for line in f2 if line.startswith('f ')]

    f1.close()
    f2.close()

    diff_lines = []
    updated_verts = []

    if not withUV:
        for k in range(min(len(f1_v_lines), len(f2_v_lines))):
            if not(f1_v_lines[k] == f2_v_lines[k]):
                diff_lines.append(["vm", k, f1_v_lines[k].split(' ')[1:], f2_v_lines[k].split(' ')[1:]])
                updated_verts.append(k)
        if len(f1_v_lines) < len(f2_v_lines):
            for idx, el in enumerate(f2_v_lines[len(f1_v_lines):]):
                diff_lines.append(["va", len(f1_v_lines) + 1 + idx, '', el.split(' ')[1:]])
                updated_verts.append(len(f1_v_lines) + idx)
        elif len(f2_v_lines) < len(f1_v_lines):
            for idx, el in enumerate(f1_v_lines[len(f2_v_lines):]):
                diff_lines.append(["vd", len(f2_v_lines) + 1 + idx, el.split(' ')[1:], ''])

        '''
        for idx, el in enumerate(f2_f_lines):
            verts = el.split(' ')[1:]
            for v in verts:
                v = int(v)
                if (v + 1) in updated_verts:
                    diff_lines.append(["fu", idx, None, None])
        '''

        for k in range(min(len(f1_f_lines), len(f2_f_lines))):
            if not(f1_f_lines[k] == f2_f_lines[k]):
                diff_lines.append(["fm", k, f1_f_lines[k].split(' ')[1:], f2_f_lines[k].split(' ')[1:]])
        if len(f1_f_lines) < len(f2_f_lines):
            for idx, el in enumerate(f2_f_lines[len(f1_f_lines):]):
                diff_lines.append(["fa", len(f1_f_lines) + 1 + idx, '', el.split(' ')[1:]])
        elif len(f2_f_lines) < len(f1_f_lines):
            for idx, el in enumerate(f1_f_lines[len(f2_f_lines):]):
                diff_lines.append(["fd", len(f2_f_lines) + 1 + idx, el.split(' ')[1:], ''])
    else:
        for k in range(min(len(f1_v_lines), len(f2_v_lines))):
            if not(f1_v_lines[k] == f2_v_lines[k]):
                diff_lines.append(["vm", k, [x.split('/')[0] for x in f1_v_lines[k].split(' ')[1:]],
                                            [x.split('/')[0] for x in f2_v_lines[k].split(' ')[1:]] ])
                updated_verts.append(k)
        if len(f1_v_lines) < len(f2_v_lines):
            for idx, el in enumerate(f2_v_lines[len(f1_v_lines):]):
                diff_lines.append(["va", len(f1_v_lines) + 1 + idx, '',
                                  [x.split('/')[0] for x in el.split(' ')[1:]]])
                updated_verts.append(len(f1_v_lines) + idx)
        elif len(f2_v_lines) < len(f1_v_lines):
            for idx, el in enumerate(f1_v_lines[len(f2_v_lines):]):
                diff_lines.append(["vd", len(f2_v_lines) + 1 + idx,
                                   [x.split('/')[0] for x in el.split(' ')[1:]], ''])

        for k in range(min(len(f1_f_lines), len(f2_f_lines))):
            if not(f1_f_lines[k] == f2_f_lines[k]):
                diff_lines.append(["fm", k,
                                   [x.split('/')[0] for x in f1_f_lines[k].split(' ')[1:]],
                                   [x.split('/')[0] for x in f2_f_lines[k].split(' ')[1:]] ])
        if len(f1_f_lines) < len(f2_f_lines):
            for idx, el in enumerate(f2_f_lines[len(f1_f_lines):]):
                diff_lines.append(["fa", len(f1_f_lines) + 1 + idx, '',
                                   [x.split('/')[0] for x in el.split(' ')[1:]]] )
        elif len(f2_f_lines) < len(f1_f_lines):
            for idx, el in enumerate(f1_f_lines[len(f2_f_lines):]):
                diff_lines.append(["fd", len(f2_f_lines) + 1 + idx,
                                   [x.split('/')[0] for x in el.split(' ')[1:]], ''])

    dumb_diff_time = time.time() - start_dumbdiff
    return [diff_lines, dumb_diff_time, 0.0]

def compute_diff_set(file_path, file1, file2):
    start_set_diff = time.time()

    f1 = open(file_path + "/snap" + str(file1).zfill(6) + ".obj", 'r')
    f2 = open(file_path + "/snap" + str(file2).zfill(6) + ".obj", 'r')

    f1_v_lines = []
    f1_n_lines = []
    f1_f_lines = []
    f2_v_lines = []
    f2_n_lines = []
    f2_f_lines = []

    for line in f1:
        if line.startswith('v '):
            f1_v_lines.append(line)
        elif line.startswith('vn '):
            f1_n_lines.append(line)
        elif line.startswith('f '):
            f1_f_lines.append(line)
    for line in f2:
        if line.startswith('v '):
            f2_v_lines.append(line)
        elif line.startswith('vn '):
            f2_n_lines.append(line)
        elif line.startswith('f '):
            f2_f_lines.append(line)

    f1.close()
    f2.close()
    print("=====================================================")
    print("Verts 1: %s" % len(f1_v_lines))
    print("Verts 2: %s" % len(f2_v_lines))
    print("Normals 1: %s" % len(f1_n_lines))
    print("Normals 2: %s" % len(f2_n_lines))
    print("Faces 1: %s" % len(f1_f_lines))
    print("Faces 2: %s" % len(f2_f_lines))

    f1_v_lines_exp = set()
    for idx, f1_v in enumerate(f1_v_lines):
        f1_v_lines_exp.add(DiffEntry(idx, [x.strip() for x in f1_v.split(' ')[1:]], 'from'))
    f2_v_lines_exp = set()
    for idx, f2_v in enumerate(f2_v_lines):
        f2_v_lines_exp.add(DiffEntry(idx, [x.strip() for x in f2_v.split(' ')[1:]], 'to'))

    f1_n_lines_exp = set()
    for idx, f1_n in enumerate(f1_n_lines):
        f1_n_lines_exp.add(DiffEntry(idx, [x.strip() for x in f1_n.split(' ')[1:]], 'from'))
    f2_n_lines_exp = set()
    for idx, f2_n in enumerate(f2_n_lines):
        f2_n_lines_exp.add(DiffEntry(idx, [x.strip() for x in f2_n.split(' ')[1:]], 'to'))

    f1_f_lines_exp = set()
    for idx, f1_f in enumerate(f1_f_lines):
        f1_f_lines_exp.add(DiffEntry(idx, [x.strip() for x in f1_f.split(' ')[1:]], 'from'))
    f2_f_lines_exp = set()
    for idx, f2_f in enumerate(f2_f_lines):
        f2_f_lines_exp.add(DiffEntry(idx, [x.strip() for x in f2_f.split(' ')[1:]], 'to'))

    verts_no = len(f1_v_lines)
    diff_mod_verts = []
    diff_new_verts = []

    normals_no = len(f1_n_lines)
    diff_mod_normals = []
    diff_new_normals = []


    faces_no = len(f1_f_lines)
    diff_mod_faces = []
    diff_new_faces = []

    # ================
    # Diffing vertices
    # ================

    new_verts = (f1_v_lines_exp ^ f2_v_lines_exp) & f2_v_lines_exp
    #removed_verts = (f1_v_lines_exp ^ f2_v_lines_exp) & f1_v_lines_exp

    for el in f1_v_lines_exp:
        el.reset_other_key()
    for el in f2_v_lines_exp:
        el.reset_other_key()

    same_verts_to = f1_v_lines_exp & f2_v_lines_exp
    mod_verts = []
    for el in same_verts_to:
        if el.key != el.other_key:
            mod_verts.append(el)

    for v_m in mod_verts:
        diff_mod_verts.append([v_m.other_key, v_m.key, v_m.set_name[0], v_m.data])
        #diff_new_verts.append([v_m.key, v_m.data])
        if v_m.set_name == "from":
            verts_no = max(verts_no, v_m.other_key + 1)
        else:
            verts_no = max(verts_no, v_m.key + 1)

    for v_a in new_verts:
        diff_new_verts.append([v_a.key, v_a.data])
        verts_no = max(verts_no, v_a.key + 1)

    if verts_no == -1:
        verts_no = len(f2_v_lines_exp)

    # ================
    # Diffing normals
    # ================

    new_normals = (f1_n_lines_exp ^ f2_n_lines_exp) & f2_n_lines_exp
    #removed_normals = (f1_n_lines_exp ^ f2_n_lines_exp) & f1_n_lines_exp

    for el in f1_n_lines_exp:
        el.reset_other_key()
    for el in f2_n_lines_exp:
        el.reset_other_key()

    same_normals_to = f1_n_lines_exp & f2_n_lines_exp
    mod_normals = []
    for el in same_normals_to:
        if el.key != el.other_key:
            mod_normals.append(el)

    for n_m in mod_normals:
        diff_mod_normals.append([n_m.other_key, n_m.key, n_m.set_name[0], n_m.data])
        #diff_new_normals.append([n_m.key, n_m.data])
        if n_m.set_name == "from":
            normals_no = max(normals_no, n_m.other_key + 1)
        else:
            normals_no = max(normals_no, n_m.key + 1)

    for n_a in new_normals:
        diff_new_normals.append([n_a.key, n_a.data])
        normals_no = max(normals_no, n_a.key + 1)

    if normals_no == -1:
        normals_no = len(f2_n_lines_exp)

    # ================
    # Diffing faces
    # ================

    new_faces = (f1_f_lines_exp ^ f2_f_lines_exp) & f2_f_lines_exp
    #removed_faces = (f1_f_lines_exp ^ f2_f_lines_exp) & f1_f_lines_exp

    for el in f1_f_lines_exp:
        el.reset_other_key()
    for el in f2_f_lines_exp:
        el.reset_other_key()

    same_faces_to = f1_f_lines_exp & f2_f_lines_exp
    mod_faces = []
    for el in same_faces_to:
        if el.key != el.other_key:
            mod_faces.append(el)

    for f_m in mod_faces:
        diff_mod_faces.append([f_m.other_key, f_m.key, f_m.set_name[0], f_m.data])
        #diff_new_faces.append([f_m.key, f_m.data])
        if f_m.set_name == "from":
            faces_no = max(faces_no, f_m.other_key + 1)
        else:
            faces_no = max(faces_no, f_m.key + 1)

    for f_a in new_faces:
        diff_new_faces.append([f_a.key, f_a.data])
        faces_no = max(faces_no, f_a.key + 1)

    if faces_no == -1:
        faces_no = len(f2_f_lines_exp)

    set_diff_time = time.time() - start_set_diff

    print()
    print("Mod verts %s" % len(diff_mod_verts))
    print("New verts %s" % len(diff_new_verts))

    print("Mod norms %s" % len(diff_mod_normals))
    print("New norms %s" % len(diff_new_normals))

    print("Mod faces %s" % len(diff_mod_faces))
    print("New faces %s" % len(diff_new_faces))


    return [diff_mod_verts, diff_new_verts, verts_no,
            diff_mod_normals, diff_new_normals, normals_no,
            diff_mod_faces, diff_new_faces, faces_no, set_diff_time]

def reallocate_array(old_array):
    temp = np.zeros(old_array.shape, old_array.dtype)
    return np.concatenate((old_array, temp), axis=0)

def generate_diff(models, obj_path, diff_path):
    """
    Generates diff between OBJ files
    :param models: in the form of [model_name, start, end, step]
    :param obj_path: path where the OBJ files are found
    :param diff_path: path where the diff files will be saved
    """
    c = 0
    for name, start, end, step in models:     # ["gargoyle2", 1058]   ["monster", 967]]   ["task02", 2619]    ["task06", 987]
        total_time = 0

        diff = {}
        for j in range(start, end, step):
            obj_files_path = obj_path + name
            diff_files_path = diff_path + name + "/step_" + str(step)

            if not os.path.exists(diff_files_path):
                os.makedirs(diff_files_path)

            if j+step > end:
                diff_mod_verts, diff_new_verts, verts_no,\
                diff_mod_normals, diff_new_normals, normals_no,\
                diff_mod_faces, diff_new_faces, faces_no, set_diff_time = compute_diff_set(obj_files_path, j, end)
            else:
                diff_mod_verts, diff_new_verts, verts_no,\
                diff_mod_normals, diff_new_normals, normals_no,\
                diff_mod_faces, diff_new_faces, faces_no, set_diff_time = compute_diff_set(obj_files_path, j, j + step)

            diff["valid"] = True
            diff["mod_verts"] = diff_mod_verts
            diff["new_verts"] = diff_new_verts
            diff["verts_no"] = verts_no
            diff["mod_normals"] = diff_mod_normals
            diff["new_normals"] = diff_new_normals
            diff["normals_no"] = normals_no
            diff["mod_faces"] = diff_mod_faces
            diff["new_faces"] = diff_new_faces
            diff["faces_no"] = faces_no

            total_time += set_diff_time

            diff_lines = len(diff_mod_verts) + len(diff_new_verts) + \
                         len(diff_mod_normals) + len(diff_new_normals) + \
                         len(diff_mod_faces) + len(diff_new_faces)

            fh = open(diff_files_path + "/diff_" + str(c), "wb+")
            if diff_lines > 0:
                pickle.dump(diff, fh)
            else:
                pickle.dump({"valid": False}, fh)
            fh.close()
            c += 1
            print("SAVED DIFF " + str(j) + " for " + name)
            print("=====================================================")


if __name__ == "__main__":
    models = [["task01", 0, 500, 10]]  # ["gargoyle2", 1058]   ["monster", 967]]   ["task02", 2619]    ["task06", 987]
    obj_files_path = "../obj2_files/"
    diff_files_path = "../diff_new/"
    start = 0
    generate_diff(models, obj_files_path, diff_files_path)

'''
if __name__ == "__main__":
    for name, end in [["task06", 987]]:     # ["gargoyle2", 1058]   ["monster", 967]]   ["task02", 2619]    ["task06", 987]
        obj_files_path = "../obj2_files/" + name
        diff_files_path = "../diff_new/" + name
        start = 0

        total_time = 0

        diff = {}
        for j in range(start, end):
            diff_mod_verts, diff_new_verts, verts_no,\
            diff_mod_normals, diff_new_normals, normals_no,\
            diff_mod_faces, diff_new_faces, faces_no, set_diff_time = compute_diff_set(obj_files_path, j, j+1)

            diff["valid"] = True
            diff["mod_verts"] = diff_mod_verts
            diff["new_verts"] = diff_new_verts
            diff["verts_no"] = verts_no
            diff["mod_normals"] = diff_mod_normals
            diff["new_normals"] = diff_new_normals
            diff["normals_no"] = normals_no
            diff["mod_faces"] = diff_mod_faces
            diff["new_faces"] = diff_new_faces
            diff["faces_no"] = faces_no

            total_time += set_diff_time

            diff_lines = len(diff_mod_verts) + len(diff_new_verts) + \
                         len(diff_mod_normals) + len(diff_mod_normals) + \
                         len(diff_mod_faces) + len(diff_mod_faces)

            fh = open(diff_files_path + "/diff_" + str(j), "wb+")
            if diff_lines > 0:
                pickle.dump(diff, fh)
            else:
                pickle.dump({"valid": False}, fh)
            fh.close()
            print("SAVED DIFF " + str(j) + " for " + name)
            print("=====================================================")
'''