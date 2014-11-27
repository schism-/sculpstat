__author__ = 'christian'

import json
import pickle
import utility.common as common
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy
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
        if not brush_data["size"]:
            brush_data["size"] = None
        if not brush_data["paths"]:
            brush_data["paths"] = None
        if not brush_data["obboxes"]:
            brush_data["obboxes"] = None
        else:
            temp = []
            for bb in brush_data["obboxes"]:
                temp.append([bb["bbox_center"], bb["bbox_ext"]])
            brush_data["obboxes"] = temp[:]
        if not brush_data["aabboxes"]:
            brush_data["aabboxes"] = None
        return brush_data

def get_diff_data_step(diff_data, step_no):
    step_diff_data = {}

    step_diff_data["added_vertices"] = diff_data["added_vertices"][step_no]
    step_diff_data["deleted_vertices"] = diff_data["deleted_vertices"][step_no]
    step_diff_data["added_faces"] = diff_data["added_faces"][step_no]
    step_diff_data["deleted_faces"] = diff_data["deleted_faces"][step_no]
    # list of [c_x, c_y, c_z]
    step_diff_data["diff_centroids"] = diff_data["diff_centroids"][step_no]
    # [pos, bbox exts]
    step_diff_data["diff_bbox"] = diff_data["diff_bbox"][step_no]

    return step_diff_data


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
        For each step

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
            }

            brush_data = {
                "valid"
                "size"
                "mode"
                "brush_number"
                "paths"
                "obboxes"
                "aabboxes"
                "lenghts"
            }
        }
    """

    for model_name in model_names:
        print("Creating fina data for " + model_name)

        final_data = {}
        brush_data = common.load_json("../steps/" + model_name[0] + "/brush_data.json")
        diff_data  = common.load_json("../steps/" + model_name[0] + "/diff_plot_data.json")

        final_data[0] = {
            "step_number" : 0,
            "valid" : brush_data[0]["valid"],
            "brush_data" : sanitize_brush_data(brush_data['0']),
            "diff_data" : null_diff_data()
        }

        for step_idx in range(1, len(brush_data)):
            print(str(step_idx) + " ",)
            final_data[step_idx] = {}
            final_data[step_idx]["step_number"] = step_idx
            final_data[step_idx]["valid"] = brush_data[str(step_idx)]["valid"]
            final_data[step_idx]["brush_data"] = sanitize_brush_data(brush_data[str(step_idx)])
            final_data[step_idx]["diff_data"] = get_diff_data_step(diff_data, step_idx - 1)

        common.save_json(final_data, "../steps/" + model_name[0] + "/final_data.json", compressed=False)


def diff_compressing(diff_root_path, diff_model_names):
    for model_name in diff_model_names:
        print("Compressing diff data for " + model_name,)
        added_vertices = []
        deleted_vertices = []
        added_faces = []
        deleted_faces = []
        diff_centroids = []
        diff_bbox = []
        diff_added_centroids = []
        diff_deleted_centroids = []
        diff_added_bbox = []
        diff_deleted_bbox = []

        step_path = "../steps/" + model_name[0] + "/mesh_data.json"
        mesh_data = common.load_json(step_path)

        serialized = False
        if os.path.isfile(diff_root_path + model_name[0] + "/step_1/serialized.txt"):
            serialized = True
            print("SERIALIZED")
        else:
            print("NOT SERIALIZED")

        for diff_no in range(0, model_name[1]):
            print(str(diff_no) + "/" + str(model_name[1]) + " ",)

            if not serialized:
                data_temp = common.load_pickle(diff_root_path + model_name[0] + "/step_1/diff_" + str(diff_no))

                if not data_temp["valid"]:
                    data_c = {"valid":False}
                else:
                    data_c["valid"] = True
                    data_c['new_verts'] = len(data_temp['new_verts'])
                    data_c['verts_no'] = int(data_temp['verts_no'])
                    data_c['new_faces'] = len(data_temp['new_faces'])
                    data_c['faces_no'] = int(data_temp['faces_no'])
                    data_av_c = data_temp["new_verts"]
            else:
                data_c = common.load_pickle(diff_root_path + model_name[0] + "/step_1/diff_" + str(diff_no) + "/diff_head")
                if data_c["valid"]:
                    data_av_c = common.load_pickle(diff_root_path + model_name[0] + "/step_1/diff_" + str(diff_no) + "/new_verts")

            if not data_c['valid']:
                added_vertices.append(0)
                deleted_vertices.append(0)
                added_faces.append(0)
                deleted_faces.append(0)
                diff_bbox.append([])
                diff_centroids.append([])
            else:
                added_vertices.append(data_c["new_verts"])
                if data_c["new_verts"] - (int(data_c["verts_no"]) - int(mesh_data[str(diff_no-1)]["vertices_no"])) < 0:
                    deleted_vertices.append(int(mesh_data[str(diff_no-1)]["vertices_no"]))
                else:
                    deleted_vertices.append(data_c["new_verts"] - (int(data_c["verts_no"]) - int(mesh_data[str(diff_no-1)]["vertices_no"])))

                added_faces.append(data_c["new_faces"])
                if data_c["new_faces"] - (int(data_c["faces_no"]) - int(mesh_data[str(diff_no-1)]["faces_no"])) < 0:
                    deleted_faces.append(int(mesh_data[str(diff_no-1)]["faces_no"]))
                else:
                    deleted_faces.append(data_c["new_faces"] - (int(data_c["faces_no"]) - int(mesh_data[str(diff_no-1)]["faces_no"])))

                if len(data_av_c) > 0:
                    # list of [c_x, c_y, c_z]
                    diff_centroids.append(get_centroid(data_av_c))
                    # list of [pos, bbox exts]
                    obb_points, bbox_pos, m_ext, r, u, f = get_bbox(data_av_c)
                    diff_bbox.append([bbox_pos, m_ext])
                else:
                    diff_bbox.append([])
                    diff_centroids.append([])

        final_data = {}
        final_data["added_vertices"] = added_vertices
        final_data["deleted_vertices"] = deleted_vertices
        final_data["added_faces"] = added_faces
        final_data["deleted_faces"] = deleted_faces
        final_data["diff_centroids"] = diff_centroids
        final_data["diff_bbox"] = diff_bbox
        common.save_json(final_data, "../steps/" + model_name[0] + "/diff_plot_data.json", compressed=False)

def brush_compressing(model_names):
    feature_vectors = {}

    for model_name in model_names:
        brush_data = common.load_json("../steps/" + model_name + "/brush_data.json")
        feature_vectors[model_name] = []
        for step_idx in brush_data:
            print("Model %s | Step %s" % (model_name, step_idx))
            data = brush_data[str(step_idx)]
            if data["valid"]:
                sizes = float(data["size"][0])
                unp_sizes = float(data["size"][1])
                modes = int(data["mode"])
                b_number = data["brush_number"]
                for i in range(b_number + 1):
                    path_points = data["paths"][i]
                    path_lenghts = float(data["lenghts"][i])
                    path_centroids = data["centroids"][i]

                    obb_pts = data["obboxes"][i][0]
                    obb_verts = [None, None, None]
                    obb_verts[0] = numpy.asarray([obb_pts[0][0], obb_pts[0][1], obb_pts[0][2]])
                    obb_verts[1] = numpy.asarray([obb_pts[1][0], obb_pts[1][1], obb_pts[1][2]])
                    obb_verts[2] = numpy.asarray([obb_pts[2][0], obb_pts[2][1], obb_pts[2][2]])
                    dist0 = numpy.linalg.norm(obb_verts[1] - obb_verts[0])
                    dist1 = numpy.linalg.norm(obb_verts[2] - obb_verts[0])
                    dist2 = numpy.linalg.norm(obb_verts[2] - obb_verts[1])

                    obb_dimensions = [dist0, dist1, dist2]
                    obb_points = obb_pts
                    obb_volumes = float(data["obboxes"][i][1])

                    aabb_pts = data["aabboxes"][i][0]
                    aabb_verts = [None, None, None]
                    aabb_verts[0] = numpy.asarray([aabb_pts[0][0], aabb_pts[0][1], aabb_pts[0][2]])
                    aabb_verts[1] = numpy.asarray([aabb_pts[1][0], aabb_pts[1][1], aabb_pts[1][2]])
                    aabb_verts[2] = numpy.asarray([aabb_pts[2][0], aabb_pts[2][1], aabb_pts[2][2]])
                    dist0 = numpy.linalg.norm(aabb_verts[1] - aabb_verts[0])
                    dist1 = numpy.linalg.norm(aabb_verts[2] - aabb_verts[0])
                    dist2 = numpy.linalg.norm(aabb_verts[2] - aabb_verts[1])

                    aabb_dimensions = [dist0, dist1, dist2]
                    aabb_points = data["aabboxes"][i][0]
                    aabb_volumes = float(data["aabboxes"][i][1])
                    break

                feature_vectors[model_name].append(
                    [sizes, unp_sizes, modes, path_lenghts,
                     path_centroids[0],path_centroids[1],path_centroids[2],
                     obb_dimensions[0], obb_dimensions[1], obb_dimensions[2],
                     aabb_dimensions[0], aabb_dimensions[1], aabb_dimensions[2], int(step_idx)]
                )

        common.save_json(feature_vectors[model_name], "../steps/" + model_name + "/feature_vector.json", compressed=True)

        out = open("../steps/" + model_name + "/feature_vector.csv", "w")
        out.write('size,unp_size,mode,lenght,centroid_x,centroid_y,centroid_z,obb_dim_1,obb_dim_2,obb_dim_3,aabb_dim_1,aabb_dim_2,aabb_dim_3,step\n')
        for line in feature_vectors[model_name]:
            l = ','.join([str(el) for el in line])
            out.write(l + '\n')
        out.close()


if __name__ == "__main__":
    models = [["elder", 3119], ["elf", 4307], ["engineer", 987],
              ["explorer", 1858], ["fighter", 1608], ["gargoyle", 1058],
              ["gorilla", 2719], ["monster", 967],
              ["ogre", 1720], ["sage", 2136]]

    # ["merman", 1003],

    # diff_compressing("/Volumes/PART FAT/diff_new/", models)
    # brush_compressing(models)

    models = [["monster", 967]]
    generate_final_data(models)