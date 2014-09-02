import numpy as np
from utility import common


def parse_single_obj(file_path):
    fh = open(file_path, 'r')

    vertex_count = 0
    vertices_list = np.zeros((10000, 3), dtype=np.float32)
    vertices_list_idx = 0

    face_count = 0
    tri_count = 0
    quad_count = 0
    poly_count = 0

    for line in fh:
        if line.startswith('#'):
            pass
        elif line.startswith('mtllib'):
            #maybe save also material info? for clustering with mesh color
            pass
        elif line.startswith('o'):
            pass
        elif line.startswith('v'):
            vertex_count += 1
            #v_data = line.strip().split(' ')

            #vertices_list[vertices_list_idx][0] = v_data[1]
            #vertices_list[vertices_list_idx][1] = v_data[2]
            #vertices_list[vertices_list_idx][2] = v_data[3]
            #vertices_list_idx += 1
            #if vertices_list_idx >= vertices_list.shape[0] * 0.8:
            #    print("reallocating array...")
            #    vertices_list = reallocate_array(vertices_list)
        elif line.startswith('f'):
            face_count += 1
            f_data = line.split(' ')
            if len(f_data) == 4:
                tri_count += 1
            elif len(f_data) == 5:
                quad_count += 1
            else:
                poly_count += 1

    # static mesh analysis
    # compute bounding box?
    # segmentation ?

    return [vertex_count, face_count, tri_count, quad_count, poly_count]

def reallocate_array(old_array):
    temp = np.zeros(old_array.shape, old_array.dtype)
    return np.concatenate((old_array, temp), axis=0)

def parse_dir(files_path):
    onlyfiles = common.get_files_from_directory(files_path, ['obj'])

    for file in onlyfiles:
        print(file)
        data = parse_single_obj(file[0])
        print(data)
        print()

if __name__ == "__main__":
    obj_files_path = "../obj_files/task01"

    parse_dir(obj_files_path)
