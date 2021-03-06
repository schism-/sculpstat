__author__ = 'christian'

import sys
sys.path.append("/Users/christian/Desktop/Ph.D./sculptAnalysis/core/")

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

def get_lines_from_obj(fh):
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



def compute_diff_set(file_path, file1, file2, is_diff_vertices, is_diff_normals, is_diff_faces):
    start_set_diff = time.time()

    f1 = open(file_path + "/snap" + str(file1).zfill(6) + ".obj", 'r')
    f2 = open(file_path + "/snap" + str(file2).zfill(6) + ".obj", 'r')

    f1_v_lines, f1_n_lines, f1_f_lines = get_lines_from_obj(f1)
    f2_v_lines, f2_n_lines, f2_f_lines = get_lines_from_obj(f2)

    print("==================== %d to %d =========================" % (file1, file2))
    print("#v1 - #v2: %d-%d" % (len(f1_v_lines), len(f2_v_lines)))
    print("#n1 - #n2: %d-%d" % (len(f1_n_lines), len(f2_n_lines)))
    print("#f1 - #f2: %d-%d" % (len(f1_f_lines), len(f2_f_lines)))

    if is_diff_vertices:
        f1_v_lines_exp = DiffEntry.create_set_from_obj_line(f1_v_lines, "from")
        f2_v_lines_exp = DiffEntry.create_set_from_obj_line(f2_v_lines, "to")
    else:
        f1_v_lines_exp = []
        f2_v_lines_exp = []

    if is_diff_normals:
        f1_n_lines_exp = DiffEntry.create_set_from_obj_line(f1_n_lines, "from")
        f2_n_lines_exp = DiffEntry.create_set_from_obj_line(f2_n_lines, "to")
    else:
        f1_n_lines_exp = []
        f2_n_lines_exp = []

    if is_diff_faces:
        f1_f_lines_exp = DiffEntry.create_set_from_obj_line(f1_f_lines, "from")
        f2_f_lines_exp = DiffEntry.create_set_from_obj_line(f2_f_lines, "to")
    else:
        f1_f_lines_exp = []
        f2_f_lines_exp = []

    verts_no = min(len(f1_v_lines), len(f2_v_lines))
    diff_mod_verts = []
    diff_new_verts = []
    diff_del_verts = []

    normals_no = min(len(f1_n_lines), len(f2_n_lines))
    diff_mod_normals = []
    diff_new_normals = []
    diff_del_normals = []

    faces_no = min(len(f1_f_lines), len(f2_f_lines))
    diff_mod_faces = []
    diff_new_faces = []
    diff_del_faces = []

    # ================================================================
    #                        Diffing vertices
    # ================================================================

    if is_diff_vertices:
        verts_diff_simm = f1_v_lines_exp ^ f2_v_lines_exp
        new_verts = verts_diff_simm & f2_v_lines_exp
        removed_verts = verts_diff_simm & f1_v_lines_exp

        for el in f1_v_lines_exp:
            el.reset_other_key()
        for el in f2_v_lines_exp:
            el.reset_other_key()

        mod_verts = []
        same_verts_to = f1_v_lines_exp & f2_v_lines_exp
        for el in same_verts_to:
            if el.key != el.other_key:
                mod_verts.append(el)

        for v_m in mod_verts:
            diff_mod_verts.append([v_m.other_key, v_m.key, v_m.set_name[0], v_m.data])
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


    # ================================================================
    #                        Diffing normals
    # ================================================================

    if is_diff_normals:
        normals_diff_simm = f1_n_lines_exp ^ f2_n_lines_exp
        new_normals = normals_diff_simm & f2_n_lines_exp
        removed_normals = normals_diff_simm & f1_n_lines_exp

        for el in f1_n_lines_exp:
            el.reset_other_key()
        for el in f2_n_lines_exp:
            el.reset_other_key()

        mod_normals = []
        same_normals_to = f1_n_lines_exp & f2_n_lines_exp
        for el in same_normals_to:
            if el.key != el.other_key:
                mod_normals.append(el)

        for n_m in mod_normals:
            diff_mod_normals.append([n_m.other_key, n_m.key, n_m.set_name[0], n_m.data])
            if n_m.set_name == "from":
                normals_no = max(normals_no, n_m.other_key + 1)
            else:
                normals_no = max(normals_no, n_m.key + 1)

        for n_a in new_normals:
            diff_new_normals.append([n_a.key, n_a.data])
            normals_no = max(normals_no, n_a.key + 1)

        for n_d in removed_normals:
            diff_del_normals.append([n_d.key, n_d.data])

        if normals_no == -1:
            normals_no = len(f2_n_lines_exp)

    # ================================================================
    #                        Diffing faces
    # ================================================================

    if is_diff_faces:
        faces_diff_simm = f1_f_lines_exp ^ f2_f_lines_exp
        new_faces = faces_diff_simm & f2_f_lines_exp
        removed_faces = faces_diff_simm & f1_f_lines_exp

        for el in f1_f_lines_exp:
            el.reset_other_key()
        for el in f2_f_lines_exp:
            el.reset_other_key()

        mod_faces = []
        same_faces_to = f1_f_lines_exp & f2_f_lines_exp
        for el in same_faces_to:
            if el.key != el.other_key:
                mod_faces.append(el)

        for f_m in mod_faces:
            diff_mod_faces.append([f_m.other_key, f_m.key, f_m.set_name[0], f_m.data])
            if f_m.set_name == "from":
                faces_no = max(faces_no, f_m.other_key + 1)
            else:
                faces_no = max(faces_no, f_m.key + 1)

        for f_a in new_faces:
            diff_new_faces.append([f_a.key, f_a.data])
            faces_no = max(faces_no, f_a.key + 1)

        for f_d in removed_faces:
            diff_del_faces.append([f_d.key, f_d.data])

        if faces_no == -1:
            faces_no = len(f2_f_lines_exp)

    set_diff_time = time.time() - start_set_diff

    print()
    print("Verts (Add, Mod, Del, tot) = %d, %d, %d, %d" % (len(diff_new_verts), len(diff_mod_verts), len(diff_del_verts), verts_no))
    print("Verts (Add, Mod, Del, tot) = %d, %d, %d, %d" % (len(diff_new_normals), len(diff_mod_normals), len(diff_del_normals), normals_no))
    print("Verts (Add, Mod, Del, tot) = %d, %d, %d, %d" % (len(diff_new_faces), len(diff_mod_faces), len(diff_del_faces), faces_no))

    return [diff_mod_verts, diff_new_verts, verts_no,
            diff_mod_normals, diff_new_normals, normals_no,
            diff_mod_faces, diff_new_faces, faces_no, set_diff_time,
            diff_del_verts, diff_del_normals, diff_del_faces]


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
    for name, start, end, step, diff_type in models:
        diff = {}

        is_diff_vertices = False
        is_diff_normals = False
        is_diff_faces = False
        for flag in diff_type:
            if flag == "a":
                is_diff_vertices = True
                is_diff_normals = True
                is_diff_faces = True
                break
            elif flag == "v":
                is_diff_vertices = True
            elif flag == "n":
                is_diff_normals = True
            elif flag == "f":
                is_diff_faces = True

        if serialize:
            if not os.path.exists(diff_path + name + "/step_" + str(step)):
                os.makedirs(diff_path + name + "/step_" + str(step))
            fs = open(diff_path + name + "/step_" + str(step) + "/serialized.txt", "w")
            fs.write("true")
            fs.close()

        obj_files_path = obj_path + name
        diff_files_path = diff_path + name + "/step_" + str(step)
        if not os.path.exists(diff_files_path):
            os.makedirs(diff_files_path)

        for j in range(start, end, step):
            if j+step > end:
                diff_mod_verts, diff_new_verts, verts_no,\
                diff_mod_normals, diff_new_normals, normals_no,\
                diff_mod_faces, diff_new_faces, faces_no, set_diff_time, \
                diff_del_verts, diff_del_normals, diff_del_faces  = compute_diff_set(obj_files_path,
                                                                                     j,
                                                                                     end,
                                                                                     is_diff_vertices,
                                                                                     is_diff_normals,
                                                                                     is_diff_faces)
            else:
                diff_mod_verts, diff_new_verts, verts_no,\
                diff_mod_normals, diff_new_normals, normals_no,\
                diff_mod_faces, diff_new_faces, faces_no, set_diff_time, \
                diff_del_verts, diff_del_normals, diff_del_faces = compute_diff_set(obj_files_path,
                                                                                    j,
                                                                                    j + step,
                                                                                    is_diff_vertices,
                                                                                    is_diff_normals,
                                                                                    is_diff_faces)

            diff["valid"] = True

            diff["mod_verts"] = diff_mod_verts
            diff["new_verts"] = diff_new_verts
            diff["del_verts"] = diff_del_verts
            diff["verts_no"] = verts_no

            diff["mod_normals"] = diff_mod_normals
            diff["new_normals"] = diff_new_normals
            diff["del_normals"] = diff_del_normals
            diff["normals_no"] = normals_no

            diff["mod_faces"] = diff_mod_faces
            diff["new_faces"] = diff_new_faces
            diff["del_faces"] = diff_del_faces
            diff["faces_no"] = faces_no

            diff_lines = len(diff_mod_verts) + len(diff_new_verts) + \
                         len(diff_mod_normals) + len(diff_new_normals) + \
                         len(diff_mod_faces) + len(diff_new_faces)

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
                    diff_head["del_verts"] = len(diff_new_verts)
                    diff_head["verts_no"] = verts_no

                    diff_head["mod_normals"] = len(diff_mod_normals)
                    diff_head["new_normals"] = len(diff_new_normals)
                    diff_head["del_normals"] = len(diff_new_normals)
                    diff_head["normals_no"] = normals_no

                    diff_head["mod_faces"] = len(diff_mod_faces)
                    diff_head["new_faces"] = len(diff_new_faces)
                    diff_head["del_faces"] = len(diff_del_faces)
                    diff_head["faces_no"] = faces_no

                    pickle.dump(diff_head, fh)

                    if is_diff_vertices:
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/mod_verts", diff_mod_verts)
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/new_verts", diff_new_verts)
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/del_verts", diff_del_verts)
                    if is_diff_normals:
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/mod_normals", diff_mod_normals)
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/new_normals", diff_new_normals)
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/del_normals", diff_del_normals)

                    if is_diff_faces:
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/mod_faces", diff_mod_faces)
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/new_faces", diff_new_faces)
                        common.save_pickle(diff_files_path + "/diff_" + str(j) + "/del_faces", diff_del_faces)
                else:
                    if not os.path.exists(diff_files_path + "/diff_" + str(j) + "/"):
                        os.makedirs(diff_files_path + "/diff_" + str(j) + "/")
                    fh_d = open(diff_files_path + "/diff_" + str(j) + "/diff_head", "wb+")
                    pickle.dump({"valid":False}, fh_d)
                    fh_d.close()

            print("SAVED DIFF " + str(j) + " for " + name)
            print("=====================================================")

def update_diff_head(models, diff_path, obj_path):
    for model in models:
        print("updating " + model[0])
        for step in range(model[2]):
            print("step" + str(step))
            diff_files_path = diff_path + model[0] + "/step_1"

            head = common.load_pickle(diff_files_path + "/diff_" + str(step) + "/diff_head")
            if head["valid"]:
                new_v = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/new_verts")
                mod_v = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/mod_verts")
                del_v = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/del_verts")

                new_n = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/new_normals")
                mod_n = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/mod_normals")
                del_n = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/del_normals")


                new_f = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/new_faces")
                mod_f = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/mod_faces")
                del_f = common.load_pickle(diff_files_path + "/diff_" + str(step)  + "/del_faces")

                v_lines, n_lines,  f_lines = common.load_obj(obj_path + model[0] + "/snap" + str(step + 1).zfill(6) + ".obj")

                head["new_verts"] = len(new_v)
                head["mod_verts"] = len(mod_v)
                head["del_verts"] = len(del_v)
                head["verts_no"] = len(v_lines)

                head["new_normals"] = len(new_n)
                head["mod_normals"] = len(mod_n)
                head["del_normals"] = len(del_n)
                head["normals_no"] = len(n_lines)

                head["new_faces"] = len(new_f)
                head["mod_faces"] = len(mod_f)
                head["del_faces"] = len(del_f)
                head["faces_no"] = len(f_lines)
            else:
                head["new_verts"] = 0
                head["mod_verts"] = 0
                head["del_verts"] = 0
                head["verts_no"] = 0

                head["new_normals"] = 0
                head["mod_normals"] = 0
                head["del_normals"] = 0
                head["normals_no"] = 0

                head["new_faces"] = 0
                head["mod_faces"] = 0
                head["del_faces"] = 0
                head["faces_no"] = 0

            common.save_pickle(diff_files_path + "/diff_" + str(step) + "/diff_head", head)


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
    '''
        Input data is in the form
            ["alien", 1024, 2216, 1]
            [model name, start step, end step, stride]
            - end step might at maximum be (number of snaps - 1)

        ["alien",       0,      1000,   1,    "a"],
        ["elder",       0,      3119,   1,    "a"],
        ["elf",         0,      4307,   1,    "a"],
        ["engineer",    0,       987,   1,    "a"],
        ["explorer",    0,      1858,   1,    "a"],
        ["fighter",     0,      1608,   1,    "a"],
        ["gargoyle",    0,      1058,   1,    "a"],
        ["gorilla",     0,      2719,   1,    "a"],
        ["man",         0,      1580,   1,    "a"],
        ["merman",      0,      2619,   1,    "a"],
        ["monster",     0,       967,   1,    "a"],
        ["ogre",        0,      1720,   1,    "a"],
        ["sage",        0,      2136,   1,    "a"],
    '''

    models = [ ["sage",        1451,      2136,   1,    "a"]] # 1451


    #obj_files_path = "/Volumes/Part Mac/obj_smooth_normals_files/"
    obj_files_path = "/Users/christian/Desktop/obj_smooth_normals_files/"

    #diff_files_path = "/Volumes/PART FAT/diff_completi/"
    diff_files_path = "/Users/christian/Desktop/diff_completi/"


    start = time.time()
    # update_diff_head(models, diff_files_path, obj_files_path)

    generate_diff(models, obj_files_path, diff_files_path, serialize=True)

    print("Diff generation took %f " % (time.time() - start))