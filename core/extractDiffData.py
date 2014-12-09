__author__ = 'christian'

import numpy
import os.path
from utility import common
import scipy.stats as scs

class DiffData(object):

    def __init__(self, model, diff_root_path):
        self.model_name = model[0]
        self.end_step = model[1]
        self.diff_root_path = diff_root_path

    def diff_compressing(self):
        print("Compressing diff data for " + self.model_name + ": ",)

        # Numeric summary
        added_vertices = []
        deleted_vertices = []
        added_normals = []
        deleted_normals = []
        added_faces = []
        deleted_faces = []

        # Geometric properties
        diff_added_bbox = []
        diff_added_centroids = []
        diff_deleted_bbox = []
        diff_deleted_centroids = []

        # Basic statistics of added/deleted points (average, variance, skewness, curtosis)
        added_mean = []
        added_variance = []
        added_skewness = []
        added_curtosis = []

        deleted_mean = []
        deleted_variance = []
        deleted_skewness = []
        deleted_curtosis = []

        serialized = False
        if os.path.isfile(self.diff_root_path + self.model_name + "/step_1/serialized.txt"):
            serialized = True
            print("SERIALIZED")
        else:
            print("NOT SERIALIZED")

        for diff_no in range(self.end_step):
            print("step %d/%d" % (diff_no, self.end_step))
            data_av_c = []
            data_dv_c = []
            if not serialized:
                data_temp = common.load_pickle(self.diff_root_path + self.model_name + "/step_1/diff_" + str(diff_no))
                if not data_temp["valid"]:
                    data_c = {"valid":False}
                else:
                    data_c["valid"] = True
                    data_c['new_verts'] = len(data_temp['new_verts'])
                    data_c['new_normals'] = len(data_temp['new_normals'])
                    data_c['new_faces'] = len(data_temp['new_faces'])
                    data_c['verts_no'] = int(data_temp['verts_no'])
                    data_c['normals_no'] = int(data_temp['normals_no'])
                    data_c['faces_no'] = int(data_temp['faces_no'])
                    data_av_c = data_temp["new_verts"]
                    data_dv_c = data_temp["del_verts"]
            else:
                data_c = common.load_pickle(self.diff_root_path + self.model_name + "/step_1/diff_" + str(diff_no) + "/diff_head")
                if data_c["valid"]:
                    data_av_c = common.load_pickle(self.diff_root_path + self.model_name + "/step_1/diff_" + str(diff_no) + "/new_verts")
                    data_dv_c = common.load_pickle(self.diff_root_path + self.model_name + "/step_1/diff_" + str(diff_no) + "/del_verts")

            if not data_c['valid']:
                added_vertices.append(0)
                deleted_vertices.append(0)

                added_normals.append(0)
                deleted_normals.append(0)

                added_faces.append(0)
                deleted_faces.append(0)

                diff_added_bbox.append([])
                diff_added_centroids.append([])

                diff_deleted_bbox.append([])
                diff_deleted_centroids.append([])

                added_mean.append([None, None, None])
                added_variance.append([None, None, None])
                added_skewness.append([None, None, None])
                added_curtosis.append([None, None, None])

                deleted_mean.append([None, None, None])
                deleted_variance.append([None, None, None])
                deleted_skewness.append([None, None, None])
                deleted_curtosis.append([None, None, None])
            else:
                # adding numeric features
                added_vertices.append(data_c["new_verts"])
                deleted_vertices.append(data_c["del_verts"])
                added_normals.append(data_c["new_normals"])
                deleted_normals.append(data_c["del_normals"])
                added_faces.append(data_c["new_faces"])
                deleted_faces.append(data_c["del_faces"])

                added_vertices_pos = []
                for key, data in data_av_c:
                    added_vertices_pos.append([float(el) for el in data])

                deleted_vertices_pos = []
                for key, data in data_dv_c:
                    deleted_vertices_pos.append([float(el) for el in data])

                # adding diff geometric features (bbox, centroids, curvature)
                if len(added_vertices_pos) > 0:
                    diff_added_centroids.append(get_centroid(added_vertices_pos))
                    obb_points, bbox_pos, m_ext, r, u, f = get_bbox(added_vertices_pos)
                    diff_added_bbox.append([bbox_pos, m_ext])
                else:
                    diff_added_bbox.append([])
                    diff_added_centroids.append([])

                if len(deleted_vertices_pos) > 0:
                    diff_deleted_centroids.append(get_centroid(deleted_vertices_pos))
                    obb_points, bbox_pos, m_ext, r, u, f = get_bbox(deleted_vertices_pos)
                    diff_deleted_bbox.append([bbox_pos, m_ext])
                else:
                    diff_deleted_bbox.append([])
                    diff_deleted_centroids.append([])

                # TODO: compute curvature

                # adding statistics
                if len(added_vertices_pos) > 0:
                    numpy_arr_av = numpy.array(added_vertices_pos)
                    added_mean.append(numpy.mean(numpy_arr_av, axis=0))
                    added_variance.append(numpy.var(numpy_arr_av, axis=0))
                    added_skewness.append(scs.skew(numpy_arr_av, axis=0))
                    added_curtosis.append(scs.kurtosis(numpy_arr_av, axis=0))
                else:
                    added_mean.append([None, None, None])
                    added_variance.append([None, None, None])
                    added_skewness.append([None, None, None])
                    added_curtosis.append([None, None, None])

                if len(deleted_vertices_pos) > 0:
                    numpy_arr_dv = numpy.array(deleted_vertices_pos)
                    deleted_mean.append(numpy.mean(numpy_arr_dv, axis=0))
                    deleted_variance.append(numpy.var(numpy_arr_dv, axis=0))
                    deleted_skewness.append(scs.skew(numpy_arr_dv, axis=0))
                    deleted_curtosis.append(scs.kurtosis(numpy_arr_dv, axis=0))
                else:
                    deleted_mean.append([None, None, None])
                    deleted_variance.append([None, None, None])
                    deleted_skewness.append([None, None, None])
                    deleted_curtosis.append([None, None, None])


        final_data = {}
        final_data["added_vertices"] = added_vertices
        final_data["deleted_vertices"] = deleted_vertices
        final_data["added_normals"] = added_normals
        final_data["deleted_normals"] = deleted_normals
        final_data["added_faces"] = added_faces
        final_data["deleted_faces"] = deleted_faces

        final_data["diff_added_centroids"] = diff_added_centroids
        final_data["diff_added_bbox"] = diff_added_bbox
        final_data["diff_deleted_centroids"] = diff_deleted_centroids
        final_data["diff_deleted_bbox"] = diff_deleted_bbox

        final_data["added_mean"] = numpy2list(added_mean)
        final_data["added_variance"] = numpy2list(added_variance)
        final_data["added_skewness"] = numpy2list(added_skewness)
        final_data["added_curtosis"] = numpy2list(added_curtosis)

        final_data["deleted_mean"] = numpy2list(deleted_mean)
        final_data["deleted_variance"] = numpy2list(deleted_variance)
        final_data["deleted_skewness"] = numpy2list(deleted_skewness)
        final_data["deleted_curtosis"] = numpy2list(deleted_curtosis)

        common.save_json(final_data, "../steps/" + self.model_name + "/diff_plot_data.json", compressed=False)

def numpy2list(array):
    ret_list = []
    for el in array:
        triple = []
        for x in el:
            triple.append(float(x) if x else None)
        ret_list.append(triple)
    return ret_list


def get_centroid(points):
    acc = [0.0, 0.0, 0.0]
    for data in points:
        acc[0] += float(data[0])
        acc[1] += float(data[1])
        acc[2] += float(data[2])
    acc[0] = float(acc[0] / float(len(points)))
    acc[1] = float(acc[1] / float(len(points)))
    acc[2] = float(acc[2] / float(len(points)))
    return acc


def get_bbox(vertices):
    mu = numpy.asarray([0.0, 0.0, 0.0], 'f')
    cov_m = numpy.zeros((3, 3), 'f')

    points = []
    for data in vertices:
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

if __name__ == "__main__":

    models = [
        ["alien",       2216],
        ["elder",       3119],
        ["elf",         4307],
        ["engineer",     987],
        ["explorer",    1858],
        ["fighter",     1608],
        ["gargoyle",    1058],
        ["gorilla",     2719],
        ["man",         1580],
        ["merman",      2619],
        ["monster",      967],
        ["ogre",        1720],
        ["sage",        2136]
    ]

    models = [
        ["explorer",    1858],
        ["fighter",     1608],
        ["gargoyle",    1058]
    ]
    diff_root = "/Volumes/PART FAT/diff_completi/"

    for model in models:
        dd = DiffData(model, diff_root)
        dd.diff_compressing()