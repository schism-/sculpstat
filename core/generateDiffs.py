__author__ = 'christian'

import numpy as np
import time
import pickle

def compute_diff(file_path, file1, file2):
    print("Diff between %d and %d" % (file1, file2))

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

    print("\t Lines in file 1 \t%d \t(v: \t%d, f: \t%d)" % (len(f1_v_lines) + len(f1_f_lines),
                                                               len(f1_v_lines),
                                                               len(f1_f_lines)))

    print("\t Lines in file 2 \t%d \t(v: \t%d, f: \t%d)" % (len(f2_v_lines) + len(f2_f_lines),
                                                               len(f2_v_lines),
                                                               len(f2_f_lines)))

    start_dumbdiff = time.time()
    diff_lines_no = 0
    diff_lines = []

    # noinspection PyTypeChecker
    for k in range(min(len(f1_v_lines), len(f2_v_lines))):
        if not(f1_v_lines[k] == f2_v_lines[k]):
            diff_lines.append(["vm", k, f1_v_lines[k].split(' ')[1:], f2_v_lines[k].split(' ')[1:]])
            diff_lines_no += 1
    if len(f1_v_lines) < len(f2_v_lines):
        print("\t -f2 has more verts-\t\t Excess: \t%d" % (len(f2_v_lines) - len(f1_v_lines)))
        for idx, el in enumerate(f2_v_lines[len(f1_v_lines):]):
            diff_lines.append(["va", len(f1_v_lines) + 1 + idx, '', el.split(' ')[1:]])
        diff_lines_no += (len(f2_v_lines) - len(f1_v_lines))
    elif len(f2_v_lines) < len(f1_v_lines):
        print("\t -f1 has more verts-\t\t Excess: \t%d" % (len(f1_v_lines) - len(f2_v_lines)))
        for idx, el in enumerate(f1_v_lines[len(f2_v_lines):]):
            diff_lines.append(["vd", len(f2_v_lines) + 1 + idx, el.split(' ')[1:], ''])
        diff_lines_no += (len(f1_v_lines) - len(f2_v_lines))

    # noinspection PyTypeChecker
    for k in range(min(len(f1_f_lines), len(f2_f_lines))):
        if not(f1_f_lines[k] == f2_f_lines[k]):
            diff_lines.append(["fm", k, f1_f_lines[k].split(' ')[1:], f2_f_lines[k].split(' ')[1:]])
            diff_lines_no += 1
    if len(f1_f_lines) < len(f2_f_lines):
        print("\t -f2 has more faces-\t\t Excess: \t%d" % (len(f2_f_lines) - len(f1_f_lines)))
        for idx, el in enumerate(f2_f_lines[len(f1_f_lines):]):
            diff_lines.append(["fa", len(f1_f_lines) + 1 + idx, '', el.split(' ')[1:]])
        diff_lines_no += (len(f2_f_lines) - len(f1_f_lines))
    elif len(f2_f_lines) < len(f1_f_lines):
        print("\t -f1 has more faces-\t\t Excess: \t%d" % (len(f1_f_lines) - len(f2_f_lines)))
        for idx, el in enumerate(f1_f_lines[len(f2_f_lines):]):
            diff_lines.append(["fd", len(fd_f_lines) + 1 + idx, el.split(' ')[1:], ''])
        diff_lines_no += (len(f1_v_lines) - len(f2_v_lines))

    dumb_diff_time = time.time() - start_dumbdiff
    print("\t -Dumbdiff-\t\t lines diffed: \t%d - took \t%f" % (diff_lines_no, dumb_diff_time))

    '''
    start_setdiff = time.time()
    f1_set = set(f1_lines)
    f2_set = set(f2_lines)
    print("\t Sets are equals? %s, %s" % (len(f1_lines) == len(f1_set), len(f2_lines) == len(f2_set)))
    res = f1_set ^ f2_set
    setdiff_time = time.time() - start_setdiff
    print("\t -Setdiff-\t\t lines diffed: \t%d - took \t%f - sets equal %s, %s" % (len(res),
                                                                                   setdiff_time,
                                                                                   len(f1_lines) == len(f1_set),
                                                                                   len(f2_lines) == len(f2_set)))
    print()
    '''
    return [diff_lines, dumb_diff_time, 0.0]

def reallocate_array(old_array):
    temp = np.zeros(old_array.shape, old_array.dtype)
    return np.concatenate((old_array, temp), axis=0)


if __name__ == "__main__":
    obj_files_path = "../obj_files/task01"
    diff_files_path = "../diff/task01"
    ddtotal = 0.0
    sdtotal = 0.0
    start = 0
    end = 1720
    # noinspection PyTypeChecker
    for j in range(start, end):
        diffed_lines, ddtime, sdtime = compute_diff(obj_files_path, j, j+1)
        ddtotal += ddtime
        sdtotal += sdtime
        fh = open(diff_files_path + "/diff_" + str(j), "wb+")
        #print(diffed_lines)
        if len(diffed_lines) > 0:
            pickle.dump(diffed_lines, fh)
        else:
            pickle.dump(["nodiff"], fh)
        fh.close()

    print("- TOTAL TIMES - \t dumbdiff: %f \tsetdiff: %f" % (ddtotal, sdtotal))
    print("- MEAN TIMES - \t dumbdiff: %f \tsetdiff: %f" % (ddtotal/(end - 1), sdtotal/(end - 1)))

