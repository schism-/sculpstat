__author__ = 'christian'

import time
import pickle
import os.path
import numpy as np
from utility import common

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
        return "%s Key: %s --- Data: %s --- Key other: %s" % (self.set_name, self.key, self.data, self.other_key)


    def __repr__(self):
        return 'DiffEntry(key=%s, data=%s, key_from=%s, set_name="%s")' % (self.key, self.data, self.other_key, self.set_name)


    def reset_other_key(self):
        self.other_key = None


    @staticmethod
    def create_set_from_obj_line(data_list, label):
        diff_entry_set = set()
        for idx, el in enumerate(data_list):
            diff_entry_set.add(DiffEntry(idx, [x.strip() for x in el.split(' ')[1:]], label))
        return diff_entry_set


def compute_diff_set(file_path, file1, file2):
    start_set_diff = time.time()

    f1 = open(file_path + "/snap" + str(file1).zfill(6) + ".obj", 'r')
    f2 = open(file_path + "/snap" + str(file2).zfill(6) + ".obj", 'r')

    f1_v_lines = []; f1_n_lines = []; f1_f_lines = []
    f2_v_lines = []; f2_n_lines = []; f2_f_lines = []

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

    print("==================== %d to %d =========================" % (file1, file2))
    print("#v1 - #v2: %d-%d" % (len(f1_v_lines), len(f2_v_lines)))
    print("#n1 - #n2: %d-%d" % (len(f1_n_lines), len(f2_n_lines)))
    print("#f1 - #f2: %d-%d" % (len(f1_f_lines), len(f2_f_lines)))

    f1_v_lines_exp = DiffEntry.create_set_from_obj_line(f1_v_lines, "from")
    f2_v_lines_exp = DiffEntry.create_set_from_obj_line(f2_v_lines, "to")

    f1_n_lines_exp = DiffEntry.create_set_from_obj_line(f1_n_lines, "from")
    f2_n_lines_exp = DiffEntry.create_set_from_obj_line(f2_n_lines, "to")

    f1_f_lines_exp = DiffEntry.create_set_from_obj_line(f1_f_lines, "from")
    f2_f_lines_exp = DiffEntry.create_set_from_obj_line(f2_f_lines, "to")

    verts_no = min(len(f1_v_lines), len(f2_v_lines))
    diff_mod_verts = []
    diff_new_verts = []
    diff_del_verts = []

    normals_no = min(len(f1_n_lines), len(f2_n_lines))
    diff_mod_normals = []
    diff_new_normals = []


    faces_no = min(len(f1_f_lines), len(f2_f_lines))
    diff_mod_faces = []
    diff_new_faces = []

    # ================
    # Diffing vertices
    # ================

    new_verts = (f1_v_lines_exp ^ f2_v_lines_exp) & f2_v_lines_exp
    removed_verts = (f1_v_lines_exp ^ f2_v_lines_exp) & f1_v_lines_exp

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

    for v_d in removed_verts:
        diff_del_verts.append([v_d.key, v_d.data])

    if verts_no == -1:
        verts_no = len(f2_v_lines_exp)

    print()

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

    print("verts no %s" % verts_no)
    print("norms no %s" % normals_no)
    print("faces no %s" % faces_no)


    return [diff_mod_verts, diff_new_verts, verts_no,
            diff_mod_normals, diff_new_normals, normals_no,
            diff_mod_faces, diff_new_faces, faces_no, set_diff_time, diff_del_verts]


def reallocate_array(old_array):
    temp = np.zeros(old_array.shape, old_array.dtype)
    return np.concatenate((old_array, temp), axis=0)


def generate_diff(models, obj_path, diff_path, serialize=False):
    """
    Generates diff between OBJ files
    :param models: in the form of [model_name, start, end, step]
    :param obj_path: path where the OBJ files are found
    :param diff_path: path where the diff files will be saved
    """

    for name, start, end, step in models:
        total_time = 0
        c = 0
        diff = {}

        if serialize:
            if not os.path.exists(diff_path + name + "/step_" + str(step)):
                os.makedirs(diff_path + name + "/step_" + str(step))
            fs = open(diff_path + name + "/step_" + str(step) + "/serialized.txt", "w")
            fs.write("true")
            fs.close()

        for j in range(start, end, step):
            obj_files_path = obj_path + name
            diff_files_path = diff_path + name + "/step_" + str(step)

            if not os.path.exists(diff_files_path):
                os.makedirs(diff_files_path)

            if j+step > end:
                diff_mod_verts, diff_new_verts, verts_no,\
                diff_mod_normals, diff_new_normals, normals_no,\
                diff_mod_faces, diff_new_faces, faces_no, set_diff_time, diff_del_verts = compute_diff_set(obj_files_path, j, end)
            else:
                diff_mod_verts, diff_new_verts, verts_no,\
                diff_mod_normals, diff_new_normals, normals_no,\
                diff_mod_faces, diff_new_faces, faces_no, set_diff_time, diff_del_verts = compute_diff_set(obj_files_path, j, j + step)

            diff["valid"] = True
            diff["mod_verts"] = diff_mod_verts
            diff["new_verts"] = diff_new_verts
            diff["del_verts"] = diff_new_verts
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

            #temp_f = open("../steps/" + name + "/diffset_del_v.txt", "a")
            #temp_f.write(str(len(diff_del_verts)) + "\n")
            #temp_f.write("\t" + str([el[0] for el in diff_del_verts]) + "\n\n")
            #temp_f.close()

            if not serialize:
                fh = open(diff_files_path + "/diff_" + str(j), "wb+")
                if diff_lines > 0:
                    pickle.dump(diff, fh)
                else:
                    pickle.dump({"valid": False}, fh)
                fh.close()
            else:
                if diff_lines > 0:
                    if not os.path.exists(diff_files_path + "/diff_" + str(j) + "/"):
                        os.makedirs(diff_files_path + "/diff_" + str(j) + "/")

                    fh = open(diff_files_path + "/diff_" + str(j) + "/diff_head", "wb+")
                    diff_head = {}
                    diff_head["valid"] = True
                    diff_head["mod_verts"] = len(diff_mod_verts)
                    diff_head["new_verts"] = len(diff_new_verts)
                    diff_head["verts_no"] = verts_no
                    diff_head["mod_normals"] = len(diff_mod_normals)
                    diff_head["new_normals"] = len(diff_new_normals)
                    diff_head["normals_no"] = normals_no
                    diff_head["mod_faces"] = len(diff_mod_faces)
                    diff_head["new_faces"] = len(diff_new_faces)
                    diff_head["faces_no"] = faces_no
                    pickle.dump(diff_head, fh)

                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/mod_verts", "wb+")
                    pickle.dump(diff_mod_verts, fh_d)
                    fh_d.close()

                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/new_verts", "wb+")
                    pickle.dump(diff_new_verts, fh_d)
                    fh_d.close()

                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/mod_normals", "wb+")
                    pickle.dump(diff_mod_normals, fh_d)
                    fh_d.close()

                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/new_normals", "wb+")
                    pickle.dump(diff_new_normals, fh_d)
                    fh_d.close()

                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/mod_faces", "wb+")
                    pickle.dump(diff_mod_faces, fh_d)
                    fh_d.close()

                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/new_faces", "wb+")
                    pickle.dump(diff_new_faces, fh_d)
                    fh_d.close()
                else:
                    if not os.path.exists(diff_files_path + "/diff_" + str(j) + "/"):
                        os.makedirs(diff_files_path + "/diff_" + str(j) + "/")
                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/diff_head", "wb+")
                    pickle.dump({"valid":False}, fh_d)
                    fh_d.close()

            c += 1
            print("SAVED DIFF " + str(j) + " for " + name)
            print("=====================================================")


def generate_deleted_elements(models, obj_path, diff_path):
    for name, start, end, step, is_serialized in models:
        obj_files_path = obj_path + name
        diff_files_path = diff_path + name + "/step_" + str(step)
        for j in range(start, end, step):
            if not is_serialized:
                fh = open(diff_files_path + "/diff_" + str(j), "rb")
                diff_data = pickle.loads(fh.read())
                if 'mod_verts' in diff_data:
                    mv_data = diff_data['mod_verts']
                else:
                    mv_data = []
            else:
                fh = open(diff_files_path + "/diff_" + str(j) + "/mod_verts", "rb")
                mv_data = pickle.loads(fh.read())

            obj_fh = open(obj_files_path + "/snap" + str(j).zfill(6) + ".obj", 'r')
            verts = []
            for line in obj_fh:
                if line.startswith('v '):
                    verts.append([float(x) for x in line.strip().split(' ')[1:]])
                elif line.startswith('vn ') or line.startswith('f '):
                    break

            next_obj_fh = open(obj_files_path + "/snap" + str(j+1).zfill(6) + ".obj", 'r')
            next_verts = []
            for line in next_obj_fh:
                if line.startswith('v '):
                    next_verts.append([float(x) for x in line.strip().split(' ')[1:]])
                elif line.startswith('vn ') or line.startswith('f '):
                    break

            not_del_verts_idx = []
            for mv in mv_data:
                # [v_m.other_key, v_m.key, v_m.set_name[0], v_m.data]
                if mv[2] == "f":
                    not_del_verts_idx.append(mv[1])
                else:
                    not_del_verts_idx.append(mv[0])

            deleted_verts = []
            for idx_v, v in enumerate(verts):
                try:
                    if (idx_v in not_del_verts_idx) or (v == next_verts[idx_v]):
                        continue
                    deleted_verts.append([idx_v, v])
                except IndexError:
                    deleted_verts.append([idx_v, v])

            if not is_serialized:
                pass
                # fh = open(diff_files_path + "/diff_" + str(j), "rb")
                # diff_data = pickle.loads(fh.read())
            else:
                pass
                # fh = open(diff_files_path + "/diff_" + str(j) + "/mod_verts", "rb")
                # diff_data = pickle.loads(fh.read())




if __name__ == "__main__":
    # ["gargoyle2", 1058]   ["monster", 967]]   ["task02", 2619]    ["task06", 987]
    # name, start, end, step  ["fighter", 0, 1609, 1], ["explorer", 1730, 1858, 1],
    # models = [["sage", 1677, 2136, 1], ["gorilla", 0, 2719, 1], ["elf", 0, 4307, 1], ["elder", 2430, 3119, 1]]

    models = [["alien", 1024, 2216, 1],
              ["man", 0, 1580, 1],
              ["merman", 1000, 2619, 1]]

    models = [["ogre", 0, 50, 1], ["monster", 0, 50, 1]]
    # alien to 1752
    obj_files_path = "/Volumes/Part Mac/obj2_files/"
    diff_files_path = "/Volumes/PART FAT/diff_new/"
    # generate_diff(models, obj_files_path, diff_files_path, serialize=True)

    models = [["monster", 0, 50, 1, False]]
    generate_deleted_elements(models, obj_files_path, diff_files_path)
