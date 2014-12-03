__author__ = 'christian'

import math
import numpy
import numpy.random
import pickle
import time
import nearpy
import os.path
from utility import common
from collections import defaultdict

from OpenGL.GL.ARB.vertex_buffer_object import *

color_map = [[0.5, 0.5, 0.5],
             [0.8, 0.7, 0.7],
             [0.55, 0.8, 0.55]]

class mMesh:
    def __init__(self, vbo):
        self.g_fVBOSupported = vbo

        self.name = ''

        self.vertexCount = 0
        self.faceCount = 0
        self.quadCount = 0
        self.trisCount = 0
        self.texCoordCount = 0
        self.normalCount = 0
        self.normalQuadCount = 0
        self.normalTrisCount = 0

        self.vertices = None
        self.verticesAsString = None

        self.seqVertices = []
        self.seqQuadVertices = []
        self.seqTrisVertices = []
        self.seqTrisMap = {}
        self.seqQuadMap = {}

        self.texCoords = None
        self.texCoordsAsString = None

        self.normals = None
        self.quadNormals = None
        self.trisNormals = None

        self.faces = None
        self.faces_n = None
        self.quads = None
        self.tris = None


        self.textureId = None

        self.colors = None
        self.quadColors = None
        self.trisColors = None


        self.VBOVertices = None
        self.VBOQuadVertices = None
        self.VBOTrisVertices = None

        self.VBOTexCoords = None

        self.VBONormals = None
        self.VBOQuadNormals = None
        self.VBOTrisNormals = None

        self.VBOColors = None
        self.VBOQuadColors = None
        self.VBOTrisColors = None

        self.rbp = None
        self.engine = None

        self.mod_vertices = None
        self.mod_quads = None
        self.mod_tris = None
        self.new_vertices = None
        self.new_quads = None
        self.new_tris = None
        self.del_vertices = None
        self.del_quads = None
        self.del_tris = None
        self.mod_faces = None
        self.new_faces = None
        self.del_faces = None
        self.upd_faces = None


    def loadModel(self, path, loadBrushes, isNumpy):
        start = time.time()
        if isNumpy:
            self.loadOBJModelFromNumpy(path)
        else:
            self.loadOBJModel(path)
        print("OBJ loaded in %f" %(time.time() - start))
        print()


    def readOBJFile(self, file_path):
        swapyz = False
        vertices = []
        normals = []
        faces = []
        faces_n = []
        quads = []
        tris = []
        for line in open(file_path, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                vertices.append(v)
            if values[0] == 'vn':
                normals.append(values[1:])
            elif values[0] == 'f':
                face = []
                face_n = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) > 2:
                        face_n.append(int(w[2]))
                    else:
                        face_n.append(0)
                faces.append(face)
                faces_n.append(face_n)
                if len(face) == 4:
                    quads.append(face)
                elif len(face) == 3:
                    tris.append(face)
            else:
                continue
        return (vertices, faces, quads, tris, faces_n, normals)


    def computeNormal(self, temp):
        v0 = self.vertices[temp[0] - 1]
        v1 = self.vertices[temp[1] - 1]
        if len(temp) == 3:
            v2 = self.vertices[temp[2] - 1]
        else:
            v2 = self.vertices[temp[3] - 1]

        edge1 = [v0[0] - v1[0],
                 v0[1] - v1[1],
                 v0[2] - v1[2]]

        edge2 = [v0[0] - v2[0],
                 v0[1] - v2[1],
                 v0[2] - v2[2]]

        normal = [edge1[1] * edge2[2] - edge1[2] * edge2[1],
                  edge1[2] * edge2[0] - edge1[0] * edge2[2],
                  edge1[0] * edge2[1] - edge1[1] * edge2[0]]

        length = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
        if length > 0.0:
            normal = [normal[0] / length, normal[1] / length, normal[2] / length]
        else:
            normal = [0.0, 0.0, 0.0]

        return normal


    def loadOBJModelFromNumpy(self, numpy_path):
        self.__init__(True)

        self.vertices = numpy.load(numpy_path + "verts.npy")
        #self.loadANNEngine()
        #self.loadVertices(self.vertices)

        self.seqQuadVertices = numpy.load(numpy_path + "seqquadverts.npy")
        self.seqTrisVertices = numpy.load(numpy_path + "seqtrisverts.npy")
        self.quadColors = numpy.load(numpy_path + "quadcolors.npy")
        self.trisColors = numpy.load(numpy_path + "triscolors.npy")
        self.quadNormals = numpy.load(numpy_path + "quadnormals.npy")
        self.trisNormals = numpy.load(numpy_path + "trisnormals.npy")

        print("Vertices detected: " + str(len(self.vertices)))
        print("Faces detected: " + str(len(self.seqQuadVertices) / 4 + len(self.seqTrisVertices) / 3))
        print("Quads detected: " + str(len(self.seqQuadVertices) / 4))
        print("Tris detected: " + str(len(self.seqTrisVertices) / 3))


        fSQM = open(numpy_path + "seqquadmap", "rb")
        self.seqQuadMap = pickle.load(fSQM)
        fSQM.close()

        fSTM = open(numpy_path + "seqtrismap", "rb")
        self.seqTrisMap = pickle.load(fSTM)
        fSTM.close()

        print("Done")


    def loadOBJModel(self, path):

        self.__init__(True)

        path_parts = path.split('/')
        self.name = (path_parts[-1].split('.'))[-2]

        self.vertices, self.faces, self.quads, self.tris, self.faces_n, self.normals = self.readOBJFile(path)

        self.quadCount = 0
        self.trisCount = 0
        for f in self.faces:
            if len(f) == 3:
                self.trisCount += 1
            else:
                self.quadCount += 1

        self.vertexCount = len(self.vertices)
        self.texCoordCount = len(self.vertices)

        self.normalQuadCount = self.quadCount * 4
        self.normalTrisCount = self.trisCount * 3

        self.quadNormals = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.trisNormals = numpy.zeros((self.trisCount * 3, 3), 'f')

        self.seqQuadVertices = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')

        self.seqTrisVertices = numpy.zeros((self.trisCount * 3, 3), 'f')
        self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')

        qIndex = 0
        tIndex = 0
        vIndex = 0
        ntIndex = 0
        nqIndex = 0

        vCol = [0, ] * len(self.vertices)

        for idx_f, f in enumerate(self.faces):
            for idx_v, v in enumerate(f):
                vIndex += 1
                v = int(v)
                if len(f) == 3:
                    self.seqTrisVertices[tIndex, 0] = self.vertices[v-1][0]
                    self.seqTrisVertices[tIndex, 1] = self.vertices[v-1][1]
                    self.seqTrisVertices[tIndex, 2] = self.vertices[v-1][2]

                    c = vCol[v-1]
                    self.trisColors[tIndex, 0] = color_map[c][0]
                    self.trisColors[tIndex, 1] = color_map[c][1]
                    self.trisColors[tIndex, 2] = color_map[c][2]
                    tIndex += 1

                    self.trisNormals[ntIndex, 0] = self.normals[self.faces_n[idx_f][idx_v]-1][0]
                    self.trisNormals[ntIndex, 1] = self.normals[self.faces_n[idx_f][idx_v]-1][1]
                    self.trisNormals[ntIndex, 2] = self.normals[self.faces_n[idx_f][idx_v]-1][2]
                    ntIndex += 1

                elif len(f) == 4:
                    self.seqQuadVertices[qIndex, 0] = self.vertices[v-1][0]
                    self.seqQuadVertices[qIndex, 1] = self.vertices[v-1][1]
                    self.seqQuadVertices[qIndex, 2] = self.vertices[v-1][2]

                    c = vCol[v-1]
                    self.quadColors[qIndex, 0] = color_map[c][0]
                    self.quadColors[qIndex, 1] = color_map[c][1]
                    self.quadColors[qIndex, 2] = color_map[c][2]
                    qIndex += 1

                    self.quadNormals[nqIndex, 0] = self.normals[self.faces_n[idx_f][idx_v]-1][0]
                    self.quadNormals[nqIndex, 1] = self.normals[self.faces_n[idx_f][idx_v]-1][1]
                    self.quadNormals[nqIndex, 2] = self.normals[self.faces_n[idx_f][idx_v]-1][2]
                    nqIndex += 1

        self.vertices = numpy.asarray(self.vertices, dtype=numpy.float32)

        print("Done")


    def read_diff_set(self, path, step_no):
        if os.path.isfile(path + "serialized.txt"):
            diff_head = common.load_pickle(path + "diff_" + str(step_no) + "/diff_head")
            if diff_head["valid"]:
                data = {}
                data["valid"] = True
                data["new_verts"] = common.load_pickle(path + "diff_" + str(step_no)  + "/new_verts")
                data["mod_verts"] = common.load_pickle(path + "diff_" + str(step_no)  + "/mod_verts")
                #data["del_verts"] = common.load_pickle(path + "diff_" + str(step_no)  + "/del_verts")
                data["verts_no"] = diff_head["verts_no"]

                data["new_normals"] = common.load_pickle(path + "diff_" + str(step_no)  + "/new_normals")
                data["mod_normals"] = common.load_pickle(path + "diff_" + str(step_no)  + "/mod_normals")
                #data["del_normals"] = common.load_pickle(path + "diff_" + str(step_no)  + "/del_normals")
                data["normals_no"] = diff_head["normals_no"]

                data["new_faces"] = common.load_pickle(path + "diff_" + str(step_no)  + "/new_faces")
                data["mod_faces"] = common.load_pickle(path + "diff_" + str(step_no)  + "/mod_faces")
                #data["del_faces"] = common.load_pickle(path + "diff_" + str(step_no)  + "/del_faces")
                data["faces_no"] = diff_head["faces_no"]
            else:
                data = {}
                data["valid"] = False
        else:
            print("NOT SERIALIZED")
            f = open(path + "diff_" + str(step_no), 'rb')
            data = pickle.load(f)

        v_mod = data["mod_verts"] if "mod_verts" in data else []
        v_add = data["new_verts"] if "new_verts" in data else []
        v_no = data["verts_no"] if "verts_no" in data else []

        n_mod = data["mod_normals"] if "mod_normals" in data else []
        n_add = data["new_normals"] if "new_normals" in data else []
        n_no = data["normals_no"] if "normals_no" in data else []

        f_mod = data["mod_faces"] if "mod_faces" in data else []
        f_add = data["new_faces"] if "new_faces" in data else []
        f_no = data["faces_no"] if "faces_no" in data else []

        return [v_mod, v_add, v_no, n_mod, n_add, n_no, f_mod, f_add, f_no]


    def apply_diff_set(self, current_step, diff_path):
        self.mod_vertices, self.new_vertices, verts_no, \
        self.mod_normals, self.new_normals, normals_no, \
        self.mod_faces, self.new_faces, faces_no = self.read_diff_set(diff_path, current_step)

        if len(self.mod_vertices) + len(self.new_vertices) +\
            len(self.mod_normals) + len(self.new_normals) +\
            len(self.mod_faces) + len(self.new_faces) == 0:
            self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')
            self.quadColors.fill(0.5)
            self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')
            self.trisColors.fill(0.5)
            return False

        print("\t--Diff stats--")
        print("\t\t\t\t\t\tMOD \t\tNEW \t\tNUMBER")
        print("\t\tVerts stats: \t%d, \t\t%d, \t\t%d" % (len(self.mod_vertices), len(self.new_vertices), verts_no))
        print("\t\tNorms stats: \t%d, \t\t%d, \t\t%d" % (len(self.mod_normals), len(self.new_normals), normals_no))
        print("\t\tFaces stats: \t%d, \t\t%d, \t\t%d" % (len(self.mod_faces), len(self.new_faces), faces_no))

        #update vertices and faces list

        # ==========================
        #     UPDATING VERTICES
        # ==========================

        if verts_no > len(self.vertices):
            temp = numpy.zeros((verts_no  - len(self.vertices), 3), self.vertices.dtype)
            self.vertices = numpy.concatenate((self.vertices, temp), axis=0)

        for v_m in self.mod_vertices:
            if v_m[2] == "t":
                self.vertices[int(v_m[1])] = v_m[3]
            else:
                self.vertices[int(v_m[0])] = v_m[3]

        for v_a in self.new_vertices:
            self.vertices[int(v_a[0])] = v_a[1]

        # ==========================
        #     UPDATING NORMALS
        # ==========================

        if normals_no > len(self.normals):
            self.normals = self.normals  + [None, ] * (normals_no - len(self.normals))
        elif normals_no < len(self.normals):
            self.normals = self.normals[:normals_no]

        for n_m in self.mod_normals:
            if n_m[2] == "t":
                self.normals[int(n_m[1])] = n_m[3]
            else:
                self.normals[int(n_m[0])] = n_m[3]

        for n_a in self.new_normals:
            self.normals[int(n_a[0])] = n_a[1]


        # ==========================
        #     UPDATING FACES
        # ==========================

        if faces_no > len(self.faces):
            self.faces = self.faces  + [None, ] * (faces_no  - len(self.faces))
            self.faces_n = self.faces_n  + [None, ] * (faces_no - len(self.faces_n))
        elif faces_no < len(self.faces):
            self.faces = self.faces[:faces_no]

        for f_m in self.mod_faces:
            if f_m[2] == "t":
                verts = []
                verts_n = []
                for v in f_m[3]:
                    v_data = v.split('/')
                    verts.append(int(v_data[0]))
                    verts_n.append(int(v_data[2]))
                self.faces[f_m[1]] = verts
                self.faces_n[f_m[1]] = verts_n
            else:
                verts = []
                verts_n = []
                for v in f_m[3]:
                    v_data = v.split('/')
                    verts.append(int(v_data[0]))
                    verts_n.append(int(v_data[2]))
                self.faces[f_m[0]] = verts
                self.faces_n[f_m[0]] = verts_n


        for f_a in self.new_faces:
            verts = []
            verts_n = []
            for v in f_a[1]:
                v_data = v.split('/')
                verts.append(int(v_data[0]))
                verts_n.append(int(v_data[2]))
            self.faces[f_a[0]] = verts
            self.faces_n[f_a[0]] = verts_n

        # ==========================
        #     PRINTING NULLS
        # ==========================

        for idx, v in enumerate(self.vertices):
            if v == None:
                print("v " + str(idx))
        for idx, n in enumerate(self.normals):
            if n == None:
                print("n " + str(idx))
        for idx, f in enumerate(self.faces):
            if f == None:
                print("f " + str(idx))

        self.quadCount = 0
        self.trisCount = 0
        for f in self.faces:
            if len(f) == 3:
                self.trisCount += 1
            else:
                self.quadCount += 1

        self.vertexCount = len(self.vertices)
        self.texCoordCount = len(self.vertices)
        self.normalQuadCount = self.quadCount * 4
        self.normalTrisCount = self.trisCount * 3

        self.quadNormals = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.trisNormals = numpy.zeros((self.trisCount * 3, 3), 'f')
        self.seqQuadVertices = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.seqTrisVertices = numpy.zeros((self.trisCount * 3, 3), 'f')
        self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')

        qIndex = 0
        tIndex = 0
        vIndex = 0

        vCol = [0, ] * len(self.vertices)
        for idx in self.new_vertices:
            vCol[int(idx[0])] = 2
        idx_f = 0

        for f in self.faces:
            idx_v = 0
            for v in f:
                vIndex += 1
                v = int(v)
                if len(f) == 3:
                    v_u = self.vertices[v-1]
                    self.seqTrisVertices[tIndex, 0] = v_u[0]
                    self.seqTrisVertices[tIndex, 1] = v_u[1]
                    self.seqTrisVertices[tIndex, 2] = v_u[2]

                    c = vCol[v-1]
                    c_u = color_map[c]
                    self.trisColors[tIndex, 0] = c_u[0]
                    self.trisColors[tIndex, 1] = c_u[1]
                    self.trisColors[tIndex, 2] = c_u[2]

                    fn = self.faces_n[idx_f][idx_v]-1
                    n_u = self.normals[fn]
                    self.trisNormals[tIndex, 0] = n_u[0]
                    self.trisNormals[tIndex, 1] = n_u[1]
                    self.trisNormals[tIndex, 2] = n_u[2]
                    tIndex += 1

                elif len(f) == 4:
                    v_u = self.vertices[v-1]
                    self.seqQuadVertices[qIndex, 0] = v_u[0]
                    self.seqQuadVertices[qIndex, 1] = v_u[1]
                    self.seqQuadVertices[qIndex, 2] = v_u[2]

                    c = vCol[v-1]
                    c_u = color_map[c]
                    self.quadColors[qIndex, 0] = c_u[0]
                    self.quadColors[qIndex, 1] = c_u[1]
                    self.quadColors[qIndex, 2] = c_u[2]

                    fn = self.faces_n[idx_f][idx_v]-1
                    n_u = self.normals[fn]
                    self.quadNormals[qIndex, 0] = n_u[0]
                    self.quadNormals[qIndex, 1] = n_u[1]
                    self.quadNormals[qIndex, 2] = n_u[2]
                    qIndex += 1
                idx_v += 1
            idx_f += 1

        self.vertices = numpy.asarray(self.vertices, dtype=numpy.float32)
        return True

if __name__ == "__main__":
    m = mMesh(False)
    m.loadOBJModel('../obj_files/task01/snap000001.obj', False)
    m.apply_diff(1, '../diff/task01/', False)
    #m.read_diff('../diff/task01/diff_1')
