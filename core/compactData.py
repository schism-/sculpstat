__author__ = 'christian'

import json
import pickle
import utility.common as common
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy
import scipy.stats as scs
import os.path


'''
=================================================================================================
|                                       UTILITY METHODS                                         |
=================================================================================================
'''

def null_diff_data():
    diff_data = {}
    diff_data["added_vertices"] = None
    diff_data["deleted_vertices"] = None
    diff_data["added_faces"] = None
    diff_data["deleted_faces"] = None
    diff_data["diff_centroids"] = None
    diff_data["diff_bbox"] = None
    return diff_data

def null_distance_data():
    distance_data = {}
    distance_data["distance_mean"] = None
    distance_data["distance_variance"] = None
    distance_data["distance_skewness"] = None
    distance_data["distance_curtosis"] = None
    return distance_data

def null_brush_data():
    brush_data = {}
    brush_data["valid"] = False
    brush_data["size"] = None
    brush_data["mode"] = None
    brush_data["brush_number"] = None
    brush_data["paths"] = None
    brush_data["obboxes"] = None
    brush_data["aabboxes"] = None
    brush_data["centroids"] = None
    brush_data["lenghts"] = None
    return brush_data

def sanitize_brush_data(brush_data):
    if not brush_data["valid"]:
        return null_brush_data()
    else:
        labels = ["size", "mode", "brush_number", "paths",
                  "obboxes", "aabboxes", "centroids", "lenghts",
                  "pressure", "pressure_mean", "pressure_variance", "pressure_skewness", "pressure_curtosis",
                  "path_mean", "path_variance", "path_skewness", "path_curtosis"]
        for l in labels:
            if not brush_data[l]:
                brush_data[l] = None

        if brush_data["size"] is not None:
            brush_data["size"] = [[brush_data["size"][0], brush_data["size"][1]]]

        if brush_data["obboxes"] is not None:
            temp = []
            for bb in brush_data["obboxes"]:
                temp.append([bb["bbox_center"], bb["bbox_ext"]])
            brush_data["obboxes"] = temp[:]

        del brush_data["aabboxes"]

        return brush_data

def get_diff_data_step(diff_data, step_no):
    step_diff_data = {}
    step_diff_data["added_vertices"] = diff_data["added_vertices"][step_no]
    step_diff_data["deleted_vertices"] = diff_data["deleted_vertices"][step_no]
    step_diff_data["added_normals"] = diff_data["added_normals"][step_no]
    step_diff_data["deleted_normals"] = diff_data["deleted_normals"][step_no]
    step_diff_data["added_faces"] = diff_data["added_faces"][step_no]
    step_diff_data["deleted_faces"] = diff_data["deleted_faces"][step_no]

    step_diff_data["diff_added_centroids"] = diff_data["diff_added_centroids"][step_no]
    step_diff_data["diff_added_bbox"] = diff_data["diff_added_bbox"][step_no]
    step_diff_data["diff_deleted_centroids"] = diff_data["diff_deleted_centroids"][step_no]
    step_diff_data["diff_deleted_bbox"] = diff_data["diff_deleted_bbox"][step_no]

    step_diff_data["added_mean"] = diff_data["added_mean"][step_no]
    step_diff_data["added_variance"] = diff_data["added_variance"][step_no]
    step_diff_data["added_skewness"] = diff_data["added_skewness"][step_no]
    step_diff_data["added_curtosis"] = diff_data["added_curtosis"][step_no]

    step_diff_data["deleted_mean"] = diff_data["deleted_mean"][step_no]
    step_diff_data["deleted_variance"] = diff_data["deleted_variance"][step_no]
    step_diff_data["deleted_skewness"] = diff_data["deleted_skewness"][step_no]
    step_diff_data["deleted_curtosis"] = diff_data["deleted_curtosis"][step_no]
    return step_diff_data

def get_distance_data_step(distance_data, step_no):
    step_distance_data = {}
    try:
        step_distance_data["distance_mean"] = distance_data[step_no]["distance_mean"]
        step_distance_data["distance_variance"] = distance_data[step_no]["distance_variance"]
        step_distance_data["distance_skewness"] = distance_data[step_no]["distance_skewness"]
        step_distance_data["distance_curtosis"] = distance_data[step_no]["distance_curtosis"]
    except KeyError:
        step_distance_data["distance_mean"] = None
        step_distance_data["distance_variance"] = None
        step_distance_data["distance_skewness"] = None
        step_distance_data["distance_curtosis"] = None
    return step_distance_data

def get_centroid(points):
    acc = [0.0, 0.0, 0.0]
    for key, data in points:
        acc[0] += float(data[0])
        acc[1] += float(data[1])
        acc[2] += float(data[2])
    acc[0] = float(acc[0] / float(len(points)))
    acc[1] = float(acc[1] / float(len(points)))
    acc[2] = float(acc[2] / float(len(points)))
    return acc


def get_bbox(diff_entry):
    mu = numpy.asarray([0.0, 0.0, 0.0], 'f')
    cov_m = numpy.zeros((3, 3), 'f')

    points = []
    for key, data in diff_entry:
        points.append(numpy.array(data, 'f'))

    for p in points:
        mu = mu + p
    mu = mu * numpy.asarray([1.0 / float(len(points)), 1.0 / float(len(points)), 1.0 / float(len(points))])

    cxx=0.0; cxy=0.0; cxz=0.0; cyy=0.0; cyz=0.0; czz=0.0

    for p in points:
        cxx += p[0] * p[0] - mu[0] * mu[0]
        cxy += p[0] * p[1] - mu[0] * mu[1]
        cxz += p[0] * p[2] - mu[0] * mu[2]
        cyy += p[1] * p[1] - mu[1] * mu[1]
        cyz += p[1] * p[2] - mu[1] * mu[2]
        czz += p[2] * p[2] - mu[2] * mu[2]

    cov_m[0,0] = cxx;   cov_m[0,1] = cxy;   cov_m[0,2] = cxz
    cov_m[1,0] = cxy;   cov_m[1,1] = cyy;   cov_m[1,2] = cyz
    cov_m[2,0] = cxz;   cov_m[2,1] = cyz;   cov_m[2,2] = czz

    center, m_pos, m_ext, m_rot = build_from_covariance_matrix(cov_m, points)

    vol = obb_volume(m_ext)
    obb_points, bbox_pos, m_ext, r, u, f = get_bounding_box(m_rot, m_ext, m_pos)

    return [obb_points, bbox_pos, m_ext, r, u, f]


def build_from_covariance_matrix(cov_m, points):
    eigval, eigvec = numpy.linalg.eig(cov_m)
    m_rot = numpy.zeros((3, 3), 'f')

    r = numpy.asarray([eigvec[0,0], eigvec[1,0], eigvec[2,0]])
    u = numpy.asarray([eigvec[0,1], eigvec[1,1], eigvec[2,1]])
    f = numpy.asarray([eigvec[0,2], eigvec[1,2], eigvec[2,2]])
    r = normalize(r)
    u = normalize(u)
    f = normalize(f)

    m_rot[0, 0] = r[0];     m_rot[0, 1]=u[0];   m_rot[0, 2] = f[0]
    m_rot[1, 0] = r[1];     m_rot[1, 1]=u[1];   m_rot[1, 2] = f[1]
    m_rot[2, 0] = r[2];     m_rot[2, 1]=u[2];   m_rot[2, 2] = f[2]

    minim = numpy.asarray([1e10, 1e10, 1e10])
    maxim = numpy.asarray([-1e10, -1e10, -1e10])

    for p in points:
        p_prime = numpy.asarray([numpy.dot(r, p), numpy.dot(u, p), numpy.dot(f, p)])
        minim = numpy.minimum(minim, p_prime)
        maxim = numpy.maximum(maxim, p_prime)

    center = (maxim + minim) * 0.5
    m_pos = numpy.asarray([numpy.dot(m_rot[0], center), numpy.dot(m_rot[1], center), numpy.dot(m_rot[2], center)])
    m_ext = (maxim - minim) * 0.5

    return center, m_pos, m_ext, m_rot


def obb_volume(m_ext):
    return 8 * m_ext[0] * m_ext[1] * m_ext[2]


def get_bounding_box(m_rot, m_ext, m_pos):
    r =  numpy.asarray([m_rot[0][0], m_rot[1][0], m_rot[2][0]])
    u =  numpy.asarray([m_rot[0][1], m_rot[1][1], m_rot[2][1]])
    f =  numpy.asarray([m_rot[0][2], m_rot[1][2], m_rot[2][2]])
    p = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    p[0] = m_pos - r * m_ext[0] - u * m_ext[1] - f * m_ext[2]
    p[1] = m_pos + r * m_ext[0] - u * m_ext[1] - f * m_ext[2]
    p[2] = m_pos + r * m_ext[0] - u * m_ext[1] + f * m_ext[2]
    p[3] = m_pos - r * m_ext[0] - u * m_ext[1] + f * m_ext[2]
    p[4] = m_pos - r * m_ext[0] + u * m_ext[1] - f * m_ext[2]
    p[5] = m_pos + r * m_ext[0] + u * m_ext[1] - f * m_ext[2]
    p[6] = m_pos + r * m_ext[0] + u * m_ext[1] + f * m_ext[2]
    p[7] = m_pos - r * m_ext[0] + u * m_ext[1] + f * m_ext[2]
    return [p,
            [float(m_pos[0]), float(m_pos[1]), float(m_pos[2])],
            [float(m_ext[0]), float(m_ext[1]), float(m_ext[2])],
            [float(r[0]), float(r[1]), float(r[2])],
            [float(u[0]), float(u[1]), float(u[2])],
            [float(f[0]), float(f[1]), float(f[2])]]


def normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
       return v
    return v/norm


'''
=================================================================================================
|                                        MAIN  METHODS                                          |
=================================================================================================
'''

def generate_final_data(model_names):
    """
        Produces the final JSON, with all the data extracted from brush stroke and diff

        For each step, it saves:

        final_data = {
            diff_data = {
                "added_vertices"
                "deleted_vertices"
                "added_faces"
                "deleted_faces"
                "diff_added_centroids"
                "diff_added_bbox"
                "diff_deleted_centroids"
                "diff_deleted_bbox"
                "added_mean"
                "added_variance"
                "added_skewness"
                "added_curtosis"
                "deleted_mean"
                "deleted_variance"
                "deleted_skewness"
                "deleted_curtosis"
            }

            brush_data = {
                "valid"
                "size"
                "mode"
                "brush_number"
                "paths"
                "centroid"
                "obboxes"
                "aabboxes"
                "lenghts"
                "pressure"
            }

            distance_data = {
                "distance_mean"
                "distance_variance"
                "distance_skewness"
                "distance_curtosis"
            }
        }
    """

    for model_name in model_names:
        print("Creating fina data for " + model_name[0])

        final_data = {}
        brush_data = common.load_json("../steps/" + model_name[0] + "/brush_data.json")
        diff_data  = common.load_json("../steps/" + model_name[0] + "/diff_plot_data.json")
        distance_data  = common.load_json("../steps/" + model_name[0] + "/distance_data.json")

        final_data[0] = {
            "step_number" : 0,
            "valid" : brush_data['0']["valid"],
            "brush_data" : sanitize_brush_data(brush_data['0']),
            "diff_data" : null_diff_data(),
            "distance_data" : null_distance_data()
        }

        for step_idx in range(1, len(brush_data)):
            print(str(step_idx) + " ",)
            final_data[step_idx] = {}
            final_data[step_idx]["step_number"] = step_idx
            final_data[step_idx]["valid"] = brush_data[str(step_idx)]["valid"]
            final_data[step_idx]["brush_data"] = sanitize_brush_data(brush_data[str(step_idx)])
            final_data[step_idx]["diff_data"] = get_diff_data_step(diff_data, step_idx - 1)
            final_data[step_idx]["distance_data"] = get_distance_data_step(distance_data, str(step_idx))

        common.save_json(final_data, "../final_data/" + model_name[0] + "/final_data.json", compressed=False)

def distance_compressing(model_name, single_file=False):
    '''

    Produces the JSON for the data on mesh distances between steps

    For every step, it saves:
        - distance mean;
        - distance variance;
        - distance skewness;
        - distance curtosis;

    '''
    if single_file:
        file_name = "../steps/" + model_name[0] + "/distance_data.txt"

        fh = open(file_name, 'r')

        i = 0
        distances = {}
        idx = 0
        for line in fh:
            if i % 2 == 0:
                idx = int(line)
            else:
                data = line.split(' ')
                data = [float(el) for idx, el in enumerate(data) if idx % 2 == 1]
                if data:
                    np_data = numpy.array(data)
                    distances[idx] = {}
                    distances[idx]["distance_mean"] = numpy.mean(np_data, axis=0)
                    distances[idx]["distance_variance"] = numpy.var(np_data, axis=0)
                    distances[idx]["distance_skewness"] = scs.skew(np_data, axis=0)
                    distances[idx]["distance_curtosis"] = scs.kurtosis(np_data, axis=0)
                else:
                    distances[idx] = {}
                    distances[idx]["distance_mean"] = None
                    distances[idx]["distance_variance"] = None
                    distances[idx]["distance_skewness"] = None
                    distances[idx]["distance_curtosis"] = None
            i += 1
        fh.close()
        common.save_json(distances, "../steps/" + model_name[0] + "/distance_data.json", compressed=False)
    else:
        dir_name = "../steps/" + model_name[0] + "/dist_data/"

        files = common.get_files_from_directory(dir_name)

        distances = {}
        for path, filename in files:
            step = int(filename[4:])
            fh = open(path, 'r')
            dists = []
            for line in fh:
                data = line.split(' ')
                dists.append(float(data[1].strip()))
            if dists:
                distances[step] = {}
                distances[step]["distance_mean"] = numpy.mean(dists, axis=0)
                distances[step]["distance_variance"] = numpy.var(dists, axis=0)
                distances[step]["distance_skewness"] = scs.skew(dists, axis=0)
                distances[step]["distance_curtosis"] = scs.kurtosis(dists, axis=0)
            else:
                distances[step] = {}
                distances[step]["distance_mean"] = None
                distances[step]["distance_variance"] = None
                distances[step]["distance_skewness"] = None
                distances[step]["distance_curtosis"] = None

        common.save_json(distances, "../steps/" + model_name[0] + "/distance_data.json", compressed=False)


def brush_flattening(model_names):
    '''
        Saves a flattened version of the brush data (for Weka analysis and such)
    '''

    feature_vectors = {}

    for model_name in model_names:
        brush_data = common.load_json("../steps/" + model_name[0] + "/brush_data.json")
        feature_vectors[model_name[0]] = []
        for step_idx in brush_data:
            print("Model %s | Step %s" % (model_name[0], step_idx))
            data = brush_data[str(step_idx)]
            if data["valid"]:
                sizes = float(data["size"][0])
                unp_sizes = float(data["size"][1])
                modes = data["mode"]
                b_number = data["brush_number"]
                path_lenghts = 0
                for i in range(b_number + 1):
                    path_lenghts = float(data["lenghts"][i])
                    path_centroids = data["centroids"][i]

                    obb_center = [None, None, None]
                    obb_center[0] = data["obboxes"][i]["bbox_center"][0]
                    obb_center[1] = data["obboxes"][i]["bbox_center"][1]
                    obb_center[2] = data["obboxes"][i]["bbox_center"][2]

                    obb_dimensions = [None, None, None]
                    obb_dimensions[0] = data["obboxes"][i]["bbox_ext"][0]
                    obb_dimensions[1] = data["obboxes"][i]["bbox_ext"][1]
                    obb_dimensions[2] = data["obboxes"][i]["bbox_ext"][2]
                    break

                pressure_mean = data["pressure_mean"]
                pressure_variance = data["pressure_variance"]
                pressure_skewness = data["pressure_skewness"]
                pressure_curtosis = data["pressure_curtosis"]

                path_mean = data["path_mean"]
                path_variance = data["path_variance"]
                path_skewness = data["path_skewness"]
                path_curtosis = data["path_curtosis"]

                feature_vectors[model_name[0]].append(
                    [sizes, unp_sizes, modes[0], path_lenghts,
                     path_centroids[0],path_centroids[1],path_centroids[2],
                     obb_center[0], obb_center[1], obb_center[2],
                     obb_dimensions[0], obb_dimensions[1], obb_dimensions[2],
                     pressure_mean, pressure_variance, pressure_skewness, pressure_curtosis,
                     path_mean, path_variance, path_skewness, path_curtosis,
                     int(step_idx)]
                )

        common.save_json(feature_vectors[model_name[0]], "../steps/" + model_name[0] + "/feature_vector.json", compressed=False)

        out = open("../steps/" + model_name[0] + "/feature_vector.csv", "w")
        out.write('size,unp_size,mode,lenght,' + \
                  'centroid_x,centroid_y,centroid_z,' + \
                  'obb_cen_x,obb_cen_y,obb_cen_z,'+ \
                  'obb_dim_x,obb_dim_y,obb_dim_z,'+ \
                  'pressure_mean,pressure_variance,pressure_skewness,pressure_curtosis,'+ \
                  'path_mean,path_variance,path_skewness,path_curtosis,'+ \
                  'step\n')
        for line in feature_vectors[model_name[0]]:
            l = ','.join([str(el) for el in line])
            out.write(l + '\n')
        out.close()

def final_data_flattening(model_name):
    '''
        Saves a flattened version of the brush data, duplicating the data for each point of the polyline.
    '''
    final_data = common.load_json("/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/complete/" + model_name[0] + "/final_data.json")

    flattened_data = []
    print("Flattening model %s" % model_name[0])
    for step in final_data:
        print("%s / %d" % (step, len(final_data) - 1), )
        step_data = final_data[step]["brush_data"]
        if step_data["valid"]:
            for idx, point in enumerate(step_data["paths"][0]):
                single_data = {
                    "step": int(step),
                    "projected_size": step_data["size"][0][0],
                    "unprojected_size": step_data["size"][0][1],
                    "position": point,
                    "pressure": step_data["pressure"][0][idx],
                    "mode": step_data["mode"][0],
                }
                flattened_data.append(single_data)

    save_path = "/Users/christian/Desktop/Ph.D./sculptAnalysis_final_data/flattened/" + model_name[0] + "/"
    common.make_dirs(save_path)
    common.save_json(flattened_data,
                     save_path + "flattened_data.json",
                     compressed=False)


if __name__ == "__main__":
    '''
    models = [
        ["alien",    2216],
        ["elder",    3119],
        ["elf",      4307],
        ["engineer",  987],
        ["explorer", 1858],
        ["fighter",  1608],
        ["gargoyle", 1058],
        ["gorilla",  2719],
        ["man",      1580],
        ["merman",   2619],
        ["monster", 967],
        ["ogre", 1720],
        ["sage",     2136]
    ]
    '''

    models = [
        ["elder",    3119],
        ["engineer",  987],
        ["explorer", 1858],
        ["fighter",  1608],
        ["gargoyle", 1058],
        ["gorilla",  2719],
        ["man",      1580],
        ["merman",   2619],
        ["monster", 967],
        ["ogre", 1720],
        ["sage",     2136]
    ]

    # brush_flattening(models)

    #generate_final_data(models)

    #for model_name in models:
    #    distance_compressing(model_name, False)

    for model_name in models:
        final_data_flattening(model_name)