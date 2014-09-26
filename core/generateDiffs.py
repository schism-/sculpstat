__author__ = 'christian'

import numpy as np
import time
import pickle

class DiffEntry(object):
    def __init__(self, key, data):
        self.key = key
        self.data = data

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "Key: %s --- Data: %s" % (self.key, self.data)

    def __repr__(self):
        return 'DiffEntry(key=%s, data=%s)' % (self.key, self.data)


def compute_diff(file_path, file1, file2):
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

    for k in range(min(len(f1_v_lines), len(f2_v_lines))):
        if not(f1_v_lines[k] == f2_v_lines[k]):
            diff_lines.append(["vm", k, f1_v_lines[k].split(' ')[1:], f2_v_lines[k].split(' ')[1:]])
    if len(f1_v_lines) < len(f2_v_lines):
        for idx, el in enumerate(f2_v_lines[len(f1_v_lines):]):
            diff_lines.append(["va", len(f1_v_lines) + 1 + idx, '', el.split(' ')[1:]])
    elif len(f2_v_lines) < len(f1_v_lines):
        for idx, el in enumerate(f1_v_lines[len(f2_v_lines):]):
            diff_lines.append(["vd", len(f2_v_lines) + 1 + idx, el.split(' ')[1:], ''])

    for k in range(min(len(f1_f_lines), len(f2_f_lines))):
        if not(f1_f_lines[k] == f2_f_lines[k]):
            diff_lines.append(["fm", k, f1_f_lines[k].split(' ')[1:], f2_f_lines[k].split(' ')[1:]])
    if len(f1_f_lines) < len(f2_f_lines):
        for idx, el in enumerate(f2_f_lines[len(f1_f_lines):]):
            diff_lines.append(["fa", len(f1_f_lines) + 1 + idx, '', el.split(' ')[1:]])
    elif len(f2_f_lines) < len(f1_f_lines):
        for idx, el in enumerate(f1_f_lines[len(f2_f_lines):]):
            diff_lines.append(["fd", len(fd_f_lines) + 1 + idx, el.split(' ')[1:], ''])

    dumb_diff_time = time.time() - start_dumbdiff
    return [diff_lines, dumb_diff_time, 0.0]

def compute_diff_set(file_path, file1, file2):
    #print("set-Diff between %d and %d" % (file1, file2))
    start_set_diff = time.time()
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

    f1_v_lines_exp = set()
    for idx, f1_v in enumerate(f1_v_lines):
        f1_v_lines_exp.add(DiffEntry(idx, f1_v.split(' ')[1:]))
    f2_v_lines_exp = set()
    for idx, f2_v in enumerate(f2_v_lines):
        f2_v_lines_exp.add(DiffEntry(idx, f2_v.split(' ')[1:]))

    f1_f_lines_exp = set()
    for idx, f1_f in enumerate(f1_f_lines):
        f1_f_lines_exp.add(DiffEntry(idx, f1_f.split(' ')[1:]))
    f2_f_lines_exp = set()
    for idx, f2_f in enumerate(f2_f_lines):
        f2_f_lines_exp.add(DiffEntry(idx, f2_f.split(' ')[1:]))

    #print("\t Lines in file 1 \t%d \t(v: \t%d, f: \t%d)" % (len(f1_v_lines) + len(f1_f_lines),
    #                                                           len(f1_v_lines),
    #                                                           len(f1_f_lines)))

    #print("\t Lines in file 2 \t%d \t(v: \t%d, f: \t%d)" % (len(f2_v_lines) + len(f2_f_lines),
    #                                                           len(f2_v_lines),
    #                                                           len(f2_f_lines)))

    diffed_verts = (f1_v_lines_exp ^ f2_v_lines_exp) & f2_v_lines_exp
    same_verts = f1_v_lines_exp & f2_v_lines_exp

    diffed_faces = (f1_f_lines_exp ^ f2_f_lines_exp) & f2_f_lines_exp
    same_faces = f1_f_lines_exp & f2_f_lines_exp

    end_set_time = time.time() - start_set_diff

    return [diffed_verts, same_verts, diffed_faces, same_faces, end_set_time]

def reallocate_array(old_array):
    temp = np.zeros(old_array.shape, old_array.dtype)
    return np.concatenate((old_array, temp), axis=0)


if __name__ == "__main__":
    obj_files_path = "../obj_files/task01"
    diff_files_path = "../diff/task01"

    ddtotal = 0.0
    sdtotal = 0.0

    start = 0
    end = 10

    # noinspection PyTypeChecker
    for j in range(start, end):
        diff_lines, dumb_diff_time, _ = compute_diff(obj_files_path, j, j+1)
        diff_set_verts, same_verts, diff_set_faces, same_faces, set_diff_time = compute_diff_set(obj_files_path, j, j+1)

        print("==============================================")
        print("Diff lines(dumb): " + str(len(diff_lines)))
        print("----------------------------------------------")
        print("Diff verts(set): " + str(len(diff_set_verts)))
        print("Same verts(set): " + str(len(same_verts)))
        print()
        print("Diff faces(set): " + str(len(diff_set_faces)))
        print("Same faces(set): " + str(len(same_faces)))
        print()
        print("Diff lines(set): " + str(len(diff_set_verts) + len(diff_set_faces)))
        print("==============================================")

        ddtotal += dumb_diff_time
        sdtotal += set_diff_time

        '''
        fh = open(diff_files_path + "/diff_" + str(j), "wb+")
        if len(diffed_lines) > 0:
            pickle.dump(diffed_lines, fh)
        else:
            pickle.dump(["nodiff"], fh)
        fh.close()
        '''

    print("- TOTAL TIMES - \t dumbdiff: %f \tsetdiff: %f" % (ddtotal, sdtotal))
    print("- MEAN TIMES - \t dumbdiff: %f \tsetdiff: %f" % (ddtotal/(end - 1), sdtotal/(end - 1)))

