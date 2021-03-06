__author__ = 'christian'

import os
import bpy
import json
import pickle
import numpy
import numpy.linalg
import scipy.stats as scs
from utility import common

class BrushData(object):

    def __init__(self, model_name, max_step=0):
        self.model_name = model_name
        self.step_path = "../steps/" + self.model_name + "/steps.json"
        self.root_blend_files = "/Volumes/PART FAT/3ddata/"
        self.paths = []
        self.max_step = max_step
        f = open(self.step_path, 'r')
        self.steps = json.load(f)


    @staticmethod
    def load_brush_size_from_blend(path):
        bpy.ops.wm.open_mainfile(filepath=path, filter_blender=True,
                                 filemode=8, display_type='FILE_DEFAULTDISPLAY',
                                 load_ui=False, use_scripts=True)
        try:
            if bpy.data.scenes["Scene"]:
               return (bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.size,
                       bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.unprojected_radius)
        except KeyError:
                print('modifier not found')


    def load_brushes_size(self):
        blend_files = common.get_files_from_directory(self.root_blend_files + self.model_name + "/", ['blend'], "snap")
        brush_sizes = []
        for file in blend_files:
            bpy.ops.wm.open_mainfile(filepath=file[0], filter_blender=True,
                                     filemode=8, display_type='FILE_DEFAULTDISPLAY',
                                     load_ui=False, use_scripts=True)
            try:
                if bpy.data.scenes["Scene"]:
                    brush_sizes.append([bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.size,
                                        bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.unprojected_radius])
                else:
                    brush_sizes.append(brush_sizes[-1])
            except KeyError:
                print('modifier not found')
        for el in brush_sizes:
            print(el)

        bs_file = open("../steps/" + self.model_name + "/b_size", "wb+")
        pickle.dump(brush_sizes, bs_file)
        bs_file.close()


    def load_brush_strokes(self, stepmax):
        for k in range(stepmax+1):
            try:
                step_ops = self.steps[str(k)]
                stroke_op = None
                for op in step_ops:
                    if op["op_name"] == "bpy.ops.sculpt.brush_stroke":
                        stroke_op = op
                        break
                if stroke_op:
                    p = self.getPath(stroke_op)
                    if len(p) > 0:
                        self.paths.append(p)
                    else:
                        self.paths.append([[0.0, 0.0, 0.0]])
                else:
                    if k > 0:
                        self.paths.append(self.paths[-1])
                    else:
                        self.paths.append([0.0, 0.0, 0.0])
            except KeyError:
                if k > 0:
                    self.paths.append(self.paths[-1])
                else:
                    self.paths.append([[0.0, 0.0, 0.0]])


    def load_brush_stroke(self, stroke_op):
        paths = []
        for s_op in stroke_op:
            p = self.getPath(s_op)
            if len(p) > 0:
                paths.append(p)
        return paths


    def load_brush_mode(self, stroke_op):
        modes = []
        for s_op in stroke_op:
            if "mode" in s_op:
                modes.append(s_op["mode"])
        return modes


    def load_brush_size(self, step_no):
        blend_file = self.root_blend_files + self.model_name + "/snap" + str(step_no).zfill(6) + ".blend"
        brush_sizes = []

        bpy.ops.wm.open_mainfile(filepath=blend_file, filter_blender=True,
                                 filemode=8, display_type='FILE_DEFAULTDISPLAY',
                                 load_ui=False, use_scripts=True)
        try:
            if bpy.data.scenes["Scene"]:
                brush_sizes = [bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.size,
                               bpy.data.scenes["Scene"].tool_settings.unified_paint_settings.unprojected_radius]
        except KeyError:
            print('brush not found')

        return brush_sizes


    @staticmethod
    def getPath(stroke_op):
        path = numpy.zeros((len(stroke_op["stroke"]), 3), 'f')
        idx = 0
        zeroes = 0
        for point in stroke_op["stroke"]:
            if abs(point["location"][0]) < 200 and abs(point["location"][1]) < 200 and abs(point["location"][2]) < 200:
                path[idx] = [point["location"][0], point["location"][2], -1.0 * point["location"][1]]
                path[idx] = numpy.asarray(path[idx])
                idx += 1
            else:
                zeroes += 1
        if zeroes > 0:
            path = path[:-zeroes]
        return path

    @staticmethod
    def get2DPath(stroke_op):
        path = []
        for point in stroke_op["stroke"]:
            path.append([float(point["mouse"][0]), float(point["mouse"][1])])
        lenght = BrushData.get_path_length(path)
        return path, lenght

    @staticmethod
    def get_path_length(path):
        l = 0.0
        for k in range(len(path) - 1):
            l += numpy.linalg.norm(numpy.array(path[k+1]) - numpy.array(path[k]))
        return l


    @staticmethod
    def get_path_pressure(stroke_op):
        pressures = []
        for point in stroke_op["stroke"]:
            pressures.append(float(point["pressure"]))
        return pressures


    def get_path_aa_bbox(self, points):
        aabb_points = []
        min_ext = numpy.array([1000.0, 1000.0, 1000.0])
        max_ext = numpy.array([-1000.0, -1000.0, -1000.0])

        for p in points:
            min_ext = numpy.minimum(min_ext, numpy.array(p))
            max_ext = numpy.maximum(max_ext, numpy.array(p))

        aabb_points.append([min_ext[0], min_ext[1], min_ext[2]])
        aabb_points.append([max_ext[0], min_ext[1], min_ext[2]])
        aabb_points.append([max_ext[0], max_ext[1], min_ext[2]])
        aabb_points.append([min_ext[0], max_ext[1], min_ext[2]])
        aabb_points.append([min_ext[0], min_ext[1], max_ext[2]])
        aabb_points.append([max_ext[0], min_ext[1], max_ext[2]])
        aabb_points.append([max_ext[0], max_ext[1], max_ext[2]])
        aabb_points.append([min_ext[0], max_ext[1], max_ext[2]])

        vol = (max_ext[0] - min_ext[0]) * (max_ext[1] - min_ext[1]) * (max_ext[2] - min_ext[2])

        return [aabb_points, vol]

    def get_path_bbox(self, points):
        mu = numpy.asarray([0.0, 0.0, 0.0])
        cov_m = numpy.zeros((3, 3), 'f')

        for p in points:
            s = 1.0 / float(points.shape[0])
            r = p * (1.0 / float(points.shape[0]))
            mu += p * (1.0 / float(points.shape[0]))

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

        center, m_pos, m_ext, m_rot = self.build_from_covariance_matrix(cov_m, points)

        vol = BrushData.obb_volume(m_ext)
        obb_points, r, u, f = BrushData.get_bounding_box(m_rot, m_ext, m_pos)

        return [obb_points, vol, m_pos, m_ext, r, u, f]

    def build_from_covariance_matrix(self, cov_m, points):
        eigval, eigvec = numpy.linalg.eig(cov_m)
        m_rot = numpy.zeros((3, 3), 'f')

        r = numpy.asarray([eigvec[0,0], eigvec[1,0], eigvec[2,0]])
        u = numpy.asarray([eigvec[0,1], eigvec[1,1], eigvec[2,1]])
        f = numpy.asarray([eigvec[0,2], eigvec[1,2], eigvec[2,2]])
        r = self.normalize(r)
        u = self.normalize(u)
        f = self.normalize(f)

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

    @staticmethod
    def obb_volume(m_ext):
        return 8 * m_ext[0] * m_ext[1] * m_ext[2]

    @staticmethod
    def get_bounding_box(m_rot, m_ext, m_pos):
        r =  numpy.asarray([m_rot[0][0], m_rot[1][0], m_rot[2][0]])
        u =  numpy.asarray([m_rot[0][1], m_rot[1][1], m_rot[2][1]])
        f =  numpy.asarray([m_rot[0][2], m_rot[1][2], m_rot[2][2]])
        p = numpy.zeros((8, 3), 'f')
        p[0] = m_pos - r * m_ext[0] - u * m_ext[1] - f * m_ext[2]
        p[1] = m_pos + r * m_ext[0] - u * m_ext[1] - f * m_ext[2]
        p[2] = m_pos + r * m_ext[0] - u * m_ext[1] + f * m_ext[2]
        p[3] = m_pos - r * m_ext[0] - u * m_ext[1] + f * m_ext[2]
        p[4] = m_pos - r * m_ext[0] + u * m_ext[1] - f * m_ext[2]
        p[5] = m_pos + r * m_ext[0] + u * m_ext[1] - f * m_ext[2]
        p[6] = m_pos + r * m_ext[0] + u * m_ext[1] + f * m_ext[2]
        p[7] = m_pos - r * m_ext[0] + u * m_ext[1] + f * m_ext[2]
        return p, r, u, f

    @staticmethod
    def normalize(v):
        norm = numpy.linalg.norm(v)
        if norm == 0:
           return v
        return v/norm


    def converto_to_str(self):
        in_file = open("../steps/" + self.model_name + "/b_data", "rb")
        b_data = pickle.load(in_file)

        in_file2 = open("../steps/" + self.model_name + "/b_size", "rb")
        b_size = pickle.load(in_file2)

        out_file = open("../steps/" + self.model_name + "/b_data_nopickle", "w+")
        for bbox_p, l, vol in b_data:
            flat = [" ".join([str(el[0]), str(el[1]), str(el[2])]) for el in bbox_p]
            out_file.write("%s %s %s\n" % (l, vol, ' '.join(flat)))
        out_file.close()

        out_file2 = open("../steps/" + self.model_name + "/b_size_nopickle", "w+")
        for size, unp_size in b_size:
            out_file2.write("%s %s\n" % (size, unp_size))
        out_file2.close()

    def get_path_centroid(self, points):
        temp = [0,0,0]
        for p in points:
            temp[0] += p[0]
            temp[1] += p[1]
            temp[2] += p[2]
        temp[0] /= len(points)
        temp[1] /= len(points)
        temp[2] /= len(points)
        return temp

    def extract_data(self):
        ret_data = {}
        for k in range(self.max_step + 1):
            ret_data[k] = self.extract_data_single(k)
        return ret_data

    def extract_brush_type(self):
        ret_data = {}
        current_brush = "DRAW"
        for k in range(self.max_step + 1):
            print(str(k), '/', str(self.max_step), '--- ')
            if str(k) in self.steps:
                step_ops = self.steps[str(k)]
                for op in step_ops:
                    if "op_name" in op and op["op_name"] == "bpy.ops.paint.brush_select":
                        # we have a new brush
                        current_brush = op["sculpt_tool"]
            ret_data[k] = current_brush
        return ret_data

    def extract_brush_2d_position(self):
        ret_data = {}
        for k in range(self.max_step + 1):
            if str(k) in self.steps:
                print(str(k), '/', str(self.max_step), '--- '),
                step_ops = self.steps[str(k)]
                for op in step_ops:
                    if "op_name" in op and op["op_name"] == "bpy.ops.sculpt.brush_stroke":
                        path, lenght = BrushData.get2DPath(op)
                        ret_data[k] = {"path":path, "lenght":lenght}
        return ret_data

    def extract_data_single(self, step_no):
        '''
        Extracts all brush strokes data for a specific step
        :param step_no: step number
        :return: dictionary with all data
        '''

        return_data = {}

        '''
        brush size
        brush mode
        brush pressure
        path length
        path OBB
        path AABB
        distanza tra i brush path
        '''
        stroke_op = []
        try:
            step_ops = self.steps[str(step_no)]
            for op in step_ops:
                if "op_name" in op and op["op_name"] == "bpy.ops.sculpt.brush_stroke":
                    stroke_op.append(op)
        except KeyError:
            print("No brush found in step %d " % step_no)

        b_size = self.load_brush_size(step_no)
        b_mode = self.load_brush_mode(stroke_op)
        b_paths = self.load_brush_stroke(stroke_op)

        b_lenght = []
        b_pressure = []
        b_obb_boxes = []
        b_aabb_boxes = []
        b_paths_centroid = []
        for idx_p, path in enumerate(b_paths):
            b_lenght.append(self.get_path_length(path))
            b_pressure.append(self.get_path_pressure(stroke_op[idx_p]))
            b_obb_boxes.append(self.get_path_bbox(path))
            b_aabb_boxes.append(self.get_path_aa_bbox(path))
            b_paths_centroid.append(self.get_path_centroid(path))


        if len(b_paths) == 0:
            return_data["valid"] = False
        else:
            return_data["valid"] = True
            return_data["size"] = b_size
            return_data["mode"] = b_mode
            return_data["brush_number"] = len(b_paths)

            list_b_paths = []
            for path in b_paths:
                list_b_paths.append([[float(p[0]), float(p[1]), float(p[2])] for p in path])
            return_data["paths"] = list_b_paths

            list_b_obb_boxes = []
            for bb, volbb, m_pos, m_ext, r, u, f in b_obb_boxes:
                list_b_obb_boxes.append({
                                         "bbox_points": [[float(p[0]), float(p[1]), float(p[2])] for p in bb],
                                         "bbox_volume": volbb,
                                         "bbox_center": [float(m_pos[0]), float(m_pos[1]), float(m_pos[2])],
                                         "bbox_ext": [float(m_ext[0]), float(m_ext[1]), float(m_ext[2])],
                                         "bbox_r": [float(r[0]), float(r[1]), float(r[2])],
                                         "bbox_u": [float(u[0]), float(u[1]), float(u[2])],
                                         "bbox_f": [float(f[0]), float(f[1]), float(f[2])]
                })
            return_data["obboxes"] = list_b_obb_boxes

            list_b_aabb_boxes = []
            for aabb, volaabb in b_aabb_boxes:
                list_b_aabb_boxes.append([[[float(p[0]), float(p[1]), float(p[2])] for p in aabb], volaabb])
            return_data["aabboxes"] = list_b_aabb_boxes

            return_data["centroids"] = b_paths_centroid

            return_data["lenghts"] = b_lenght

            return_data["pressure"] = b_pressure

            temp = numpy.mean(b_pressure, axis=1)
            return_data["pressure_mean"] = temp[0]
            temp = numpy.var(b_pressure, axis=1)
            return_data["pressure_variance"] = temp[0]
            temp = scs.skew(b_pressure, axis=1)
            return_data["pressure_skewness"] = temp[0]
            temp = scs.kurtosis(b_pressure, axis=1)
            return_data["pressure_curtosis"] = temp[0]

            temp = numpy.mean(list_b_paths[0], axis=1)
            return_data["path_mean"] = temp[0]
            temp = numpy.var(list_b_paths[0], axis=1)
            return_data["path_variance"] = temp[0]
            temp = scs.skew(list_b_paths[0], axis=1)
            return_data["path_skewness"] = temp[0]
            temp = scs.kurtosis(list_b_paths[0], axis=1)
            return_data["path_curtosis"] = temp[0]


        return return_data


if __name__ == "__main__":
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

    for model_name, max_step in models:
        bd = BrushData(model_name, max_step)
        #data = bd.extract_brush_type()
        data = bd.extract_brush_2d_position()
        print("Extracting from " + model_name)
        out = open("../steps/" + bd.model_name + "/brush_2d_pos.json", "w")
        json.dump(data, out, indent=2, sort_keys=True)
        out.close()