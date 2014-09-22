__author__ = 'christian'

import numpy as np
import time
import math

def compute_diff(file_path, file1, file2):
    f1 = open(file_path + "/snap" + str(file1).zfill(6) + ".obj", 'r')
    f2 = open(file_path + "/snap" + str(file2).zfill(6) + ".obj", 'r')

    f1_lines = [line.strip() for line in f1 if line.startswith('v') or line.startswith('f')]
    f2_lines = [line.strip() for line in f2 if line.startswith('v') or line.startswith('f')]

    start_dumbdiff = time.time()
    diff_lines = 0
    for k in range(min(len(f1_lines), len(f2_lines))):
        if not(f1_lines[k] == f2_lines[k]):
            print("Line %d: %s --> %s" % (k, f1_lines[k], f2_lines[k]))
            diff_lines += 1

    if len(f1_lines) < len(f2_lines):
        print("Remaining on f2 lines..")
        diff_lines += (len(f2_lines) - len(f1_lines))
        for l in f2_lines[len(f2_lines) - 1:]:
            print("Exc2: " + l)
    elif len(f2_lines) < len(f1_lines):
        print("Remaining on f1 lines..")
        diff_lines += (len(f1_lines) - len(f2_lines))
        for l in f1_lines[len(f1_lines) - 1:]:
            print("Exc1: " + l)
    print()
    print("Diffed lines # %d" % (diff_lines))
    print("Dumbdiff took %f" % (time.time() - start_dumbdiff))

    start_setdiff = time.time()
    f1_set = set(f1_lines)
    f2_set = set(f2_lines)
    print("Sets are equals? %s, %s" % (len(f1_lines) == len(f1_set), len(f2_lines) == len(f2_set)))
    res = f1_set ^ f2_set
    print()
    print("Diffed lines # %d" % (len(res)))
    print("Setdiff took %f" % (time.time() - start_setdiff))


def reallocate_array(old_array):
    temp = np.zeros(old_array.shape, old_array.dtype)
    return np.concatenate((old_array, temp), axis=0)


if __name__ == "__main__":
    obj_files_path = "../obj_files/task01"
    compute_diff(obj_files_path, 1, 700)
