__author__ = 'christian'

import math
import numpy
import numpy.random
import pickle
import time
import nearpy
from collections import defaultdict

from OpenGL.GL.ARB.vertex_buffer_object import *

color_map = [[0.6, 0.7, 0.7],
             [0.8, 0.7, 0.7],
             [0.6, 0.8, 0.6]]

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

    def loadANNEngine(self):
        # Dimension of our vector space
        dimension = 3
        # Create a random binary hash with 10 bits
        self.rbp = nearpy.hashes.RandomBinaryProjections('rbp', 10)
        # Create engine with pipeline configuration
        self.engine = nearpy.Engine(dimension, lshashes=[self.rbp])

    def loadVertices(self, verts):
        for index in range(len(verts)):
            self.engine.store_vector(verts[index], '%d' % index)

    def getNeighbours(self, query):
        N = self.engine.neighbours(query)
        return N


    def loadModel(self, path, loadBrushes, isNumpy):
        start = time.time()
        if isNumpy:
            self.loadOBJModelFromNumpy(path)
        else:
            self.loadOBJModel(path, loadBrushes)
        print("OBJ loaded in %f" %(time.time() - start))
        print()


    def readOBJFile(self, file_path):
        swapyz = False
        vertices = []
        faces = []
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
            elif values[0] == 'f':
                face = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                faces.append(face)
                if len(face) == 4:
                    quads.append(face)
                elif len(face) == 3:
                    tris.append(face)
            else:
                continue
        return (vertices, faces, quads, tris)

    def computeNormal(self, temp):
        edge1 = [temp[0][0] - temp[1][0],
                 temp[0][1] - temp[1][1],
                 temp[0][2] - temp[1][2]]

        if len(temp) == 3:
            edge2 = [temp[0][0] - temp[2][0],
                     temp[0][1] - temp[2][1],
                     temp[0][2] - temp[2][2]]
        else:
            edge2 = [temp[0][0] - temp[3][0],
                     temp[0][1] - temp[3][1],
                     temp[0][2] - temp[3][2]]

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

    def loadOBJModel(self, path, loadBrushes):

        self.__init__(True)

        self.loadANNEngine()

        path_parts = path.split('/')
        self.name = (path_parts[-1].split('.'))[-2]

        self.vertices, self.faces, self.quads, self.tris = self.readOBJFile(path)

        self.vertexCount = len(self.vertices)
        self.quadCount = len(self.quads)
        self.trisCount = len(self.tris)
        self.texCoordCount = len(self.vertices)
        self.normalQuadCount = len(self.quads) * 4
        self.normalTrisCount = len(self.tris) * 3

        self.loadVertices(self.vertices)

        self.quadNormals = numpy.zeros((self.quadCount * 4, 3), 'f')
        #self.quadNormals = numpy.lib.pad(self.quadNormals, ((0, self.quadCount - len(self.quadNormals)), (0,0)), "edge")
        self.trisNormals = numpy.zeros((self.trisCount * 3, 3), 'f')
        #self.trisNormals = numpy.lib.pad(self.trisNormals, ((0, self.trisCount - len(self.trisNormals)), (0,0)), "edge")

        #self.seqTrisMap = numpy.zeros((self.trisCount * 3, 20), 'i')
        #self.seqTrisMapIdx = numpy.zeros((self.trisCount * 3, 1), 'i')

        #self.seqQuadMap = numpy.zeros((self.quadCount * 4, 20), 'i')
        #self.seqQuadMapIdx = numpy.zeros((self.quadCount * 4, 1), 'i')

        #self.texCoords = numpy.zeros((self.faceCount * mult, 2), 'f')

        print("Vertices detected: " + str(self.vertexCount) + " --> " + str(len(self.vertices)))
        print("Faces detected: " + str(self.faceCount))
        print("Quads detected: " + str(self.quadCount))
        print("Tris detected: " + str(self.trisCount))

        fIndex = 0
        qIndex = 0
        tIndex = 0
        vIndex = 0

        ntIndex = 0
        nqIndex = 0

        #Initializing data for seq arrays
        self.seqQuadVertices = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.seqTrisVertices = numpy.zeros((self.trisCount * 3, 3), 'f')

        self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')
        print("Seq vertices: %i" % (len(self.seqVertices)))

        perVertexNormals = {}

        for f in self.faces:
            #Create a sequential array of vertices (for rendering)
            #temp = []
            normal = self.computeNormal([self.vertices[v-1] for v in f])
            for v in f:
                #temp.append(self.vertices[v-1])
                vIndex += 1

                if len(f) == 3:
                    self.seqTrisVertices[tIndex, 0] = self.vertices[v-1][0]
                    self.seqTrisVertices[tIndex, 1] = self.vertices[v-1][1]
                    self.seqTrisVertices[tIndex, 2] = self.vertices[v-1][2]
                    self.trisColors[tIndex, 0] = color_map[0][0]
                    self.trisColors[tIndex, 1] = color_map[0][1]
                    self.trisColors[tIndex, 2] = color_map[0][2]
                    if loadBrushes:
                        if v-1 not in self.seqTrisMap:
                            self.seqTrisMap[v-1] = [tIndex]
                        else:
                            self.seqTrisMap[v-1].append(tIndex)
                    tIndex += 1
                elif len(f) == 4:
                    self.seqQuadVertices[qIndex, 0] = self.vertices[v-1][0]
                    self.seqQuadVertices[qIndex, 1] = self.vertices[v-1][1]
                    self.seqQuadVertices[qIndex, 2] = self.vertices[v-1][2]
                    self.quadColors[qIndex, 0] = color_map[1][0]
                    self.quadColors[qIndex, 1] = color_map[1][1]
                    self.quadColors[qIndex, 2] = color_map[1][2]
                    if loadBrushes:
                        if v-1 not in self.seqQuadMap:
                            self.seqQuadMap[v-1] = [qIndex]
                        else:
                            self.seqQuadMap[v-1].append(qIndex)
                    qIndex += 1

                if v in perVertexNormals:
                    perVertexNormals[v][1] = [(perVertexNormals[v][1][0] * perVertexNormals[v][0] + normal[0]) / (perVertexNormals[v][0] + 1),
                                              (perVertexNormals[v][1][1] * perVertexNormals[v][0] + normal[1]) / (perVertexNormals[v][0] + 1),
                                              (perVertexNormals[v][1][2] * perVertexNormals[v][0] + normal[2]) / (perVertexNormals[v][0] + 1)]
                    perVertexNormals[v][0] += 1
                else:
                    perVertexNormals[v] = [None, None]
                    perVertexNormals[v][1] = normal
                    perVertexNormals[v][0] = 1

        for f in self.faces:
            if len(f) == 3:
                for v in f:
                    self.trisNormals[ntIndex, 0] = perVertexNormals[v][1][0]
                    self.trisNormals[ntIndex, 1] = perVertexNormals[v][1][1]
                    self.trisNormals[ntIndex, 2] = perVertexNormals[v][1][2]
                    ntIndex += 1
            elif len(f) == 4:
                for v in f:
                    self.quadNormals[nqIndex, 0] = perVertexNormals[v][1][0]
                    self.quadNormals[nqIndex, 1] = perVertexNormals[v][1][1]
                    self.quadNormals[nqIndex, 2] = perVertexNormals[v][1][2]
                    nqIndex += 1
            fIndex += 1

        self.vertices = numpy.asarray(self.vertices, dtype=numpy.float32)

        print("Done")

    def read_diff(self, path):
        f = open(path, 'rb')
        data = pickle.load(f)

        v_mod = []
        v_add = []
        v_del = []

        f_mod = []
        f_add = []
        f_del = []
        f_upd = []

        q_mod = []
        q_add = []
        q_del = []

        t_mod = []
        t_add = []
        t_del = []

        # type, index, el_from, el_to
        for line in data:
            if line[0] == 'vm':
                v_mod.append(line[1:])
            elif line[0] == 'va':
                v_add.append(line[1:])
            elif line[0] == 'vd':
                v_del.append(line[1:])
            elif line[0] == 'fm':
                f_mod.append(line[1:])
                if len(line[-2]) == 4:
                    q_mod.append(line[1:])
                elif len(line[-2]) == 3:
                    t_mod.append(line[1:])
            elif line[0] == 'fa':
                f_add.append(line[1:])
                if len(line[-1]) == 4:
                    q_add.append(line[1:])
                elif len(line[-1]) == 3:
                    t_add.append(line[1:])
            elif line[0] == 'fd':
                f_del.append(line[1:])
                if len(line[-2]) == 4:
                    q_del.append(line[1:])
                elif len(line[-2]) == 3:
                    t_del.append(line[1:])
            elif line[0] == 'fu':
                f_upd.append(line[1:])

        return [v_mod, q_mod, t_mod, v_add, q_add, t_add, v_del, q_del, t_del, f_mod, f_add, f_del, f_upd]

    def apply_diff(self, current_step, diff_path, reverse=False):
        print("\tApplying diff " + str(current_step))
        start = time.time()
        self.mod_vertices, self.mod_quads, self.mod_tris, \
        self.new_vertices, self.new_quads, self.new_tris, \
        self.del_vertices, self.del_quads, self.del_tris,\
        self.mod_faces, self.new_faces, self.del_faces, self.upd_faces = self.read_diff(diff_path + "diff_" + str(current_step))
        print("Read diff took %f" % (time.time() - start))
        if len(self.mod_vertices) + len(self.mod_faces) + len(self.new_vertices) + len(self.new_faces) == 0:
            self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')
            self.quadColors.fill(0.5)
            self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')
            self.trisColors.fill(0.5)
            return False

        print("\t--Diff stats--")
        print("\t\t\t\t\t\tMOD \t\tNEW \t\tDEL")
        print("\t\tVerts stats: \t%d, \t\t%d, \t\t%d" % (len(self.mod_vertices), len(self.new_vertices), len(self.del_vertices)))
        print("\t\tQuads stats: \t%d, \t\t%d, \t\t%d" % (len(self.mod_quads), len(self.new_quads), len(self.del_quads)))
        print("\t\tTris stats: \t%d, \t\t%d, \t\t%d" % (len(self.mod_tris), len(self.new_tris), len(self.del_tris)))
        print("\t\tTris stats: \t%d, \t\t%d, \t\t%d" % (len(self.mod_faces), len(self.new_faces), len(self.del_faces)))

        #update vertices and faces list
        start = time.time()
        for vm in self.mod_vertices:
            self.vertices[int(vm[0]), 0] = float(vm[-1][0])
            self.vertices[int(vm[0]), 1] = float(vm[-1][1])
            self.vertices[int(vm[0]), 2] = float(vm[-1][2])
        if len(self.new_vertices) > 0:
            temp = numpy.zeros((len(self.new_vertices), 3), self.vertices.dtype)
            self.vertices = numpy.concatenate((self.vertices, temp), axis=0)
            for va in self.new_vertices:
                self.vertices[int(va[0]) - 1, 0] = float(va[-1][0])
                self.vertices[int(va[0]) - 1, 1] = float(va[-1][1])
                self.vertices[int(va[0]) - 1, 2] = float(va[-1][2])

        for fm in self.mod_faces:
            self.faces[int(fm[0])] = [int(v) for v in fm[-1]]
        if len(self.new_faces) > 0:
            self.faces = self.faces + [None, ] * (self.new_faces[-1][0] - len(self.faces))
            for fa in self.new_faces:
                self.faces[int(fa[0]) - 1] = [int(v) for v in fa[-1]]

        from_tri_to_quad = [x for x in self.mod_tris if len(x[-1]) == 4]
        from_quad_to_tri = [x for x in self.mod_quads if len(x[-1]) == 3]
        print("Update verts and faces took %f" % (time.time() - start))

        self.vertexCount = len(self.vertices)
        self.quadCount += len(self.new_quads) + len(from_tri_to_quad) - len(from_quad_to_tri)
        self.trisCount += len(self.new_tris) + len(from_quad_to_tri) - len(from_tri_to_quad)

        reset_quads = False
        reset_tris = False
        if (len(self.new_tris) + len(from_quad_to_tri) - len(from_tri_to_quad) < 0) or (len(from_tri_to_quad) > 0):
            reset_tris = True
        if (len(self.new_quads) + len(from_tri_to_quad) - len(from_quad_to_tri) < 0) or (len(from_quad_to_tri) > 0):
            reset_quads = True

        self.texCoordCount = len(self.vertices)
        self.normalQuadCount = self.quadCount * 4
        self.normalTrisCount = self.trisCount * 3
        #self.quadNormals = numpy.zeros((self.quadCount * 4, 3), 'f')
        #self.trisNormals = numpy.zeros((self.trisCount * 3, 3), 'f')

        #self.quadNormals = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.quadNormals = numpy.lib.pad(self.quadNormals, ((0, max(0, self.normalQuadCount - len(self.quadNormals))), (0,0)), "edge")
        #self.trisNormals = numpy.zeros((self.trisCount * 3, 3), 'f')
        self.trisNormals = numpy.lib.pad(self.trisNormals, ((0, max(0, self.normalTrisCount - len(self.trisNormals))), (0,0)), "edge")

        fIndex = 0
        qIndex = 0
        tIndex = 0
        vIndex = 0
        ntIndex = 0
        nqIndex = 0

        if not (reset_quads or reset_tris):
            start = time.time()
            if len(self.new_quads) > 0:
                temp = numpy.zeros((self.new_quads[-1][0] * 4 + 4 - len(self.seqQuadVertices), 3), self.seqQuadVertices.dtype)
                self.seqQuadVertices = numpy.concatenate((self.seqQuadVertices, temp), axis=0)
                self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')
            if len(self.new_tris) > 0:
                temp = numpy.zeros((self.new_tris[-1][0] * 3 + 3 - len(self.seqTrisVertices), 3), self.seqTrisVertices.dtype)
                self.seqTrisVertices = numpy.concatenate((self.seqTrisVertices, temp), axis=0)
                self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')
            self.quadColors.fill(0.5)
            self.trisColors.fill(0.5)

            to_change = [int(el[0]) for el in self.mod_vertices] + [int(el[0]) - 1 for el in self.new_vertices]
            for idx_f, f in enumerate(self.faces):
                k = 0
                for v in f:
                    v = int(v)
                    if len(f) == 3:
                        if v-1 in to_change:
                            self.seqTrisVertices[idx_f * 3 + k, 0] = self.vertices[v-1][0]
                            self.seqTrisVertices[idx_f * 3 + k, 1] = self.vertices[v-1][1]
                            self.seqTrisVertices[idx_f * 3 + k, 2] = self.vertices[v-1][2]
                            self.trisColors[idx_f * 3 + k, 0] = color_map[2][0]
                            self.trisColors[idx_f * 3 + k, 1] = color_map[2][1]
                            self.trisColors[idx_f * 3 + k, 2] = color_map[2][2]
                    if len(f) == 4:
                        if v-1 in to_change:
                            self.seqQuadVertices[idx_f * 4 + k, 0] = self.vertices[v-1][0]
                            self.seqQuadVertices[idx_f * 4 + k, 1] = self.vertices[v-1][1]
                            self.seqQuadVertices[idx_f * 4 + k, 2] = self.vertices[v-1][2]
                            self.quadColors[idx_f * 4 + k, 0] = color_map[2][0]
                            self.quadColors[idx_f * 4 + k, 1] = color_map[2][1]
                            self.quadColors[idx_f * 4 + k, 2] = color_map[2][2]
                    k += 1
            print("Update structures took %f" % (time.time() - start))
        else:
            start = time.time()
            self.seqQuadVertices = numpy.zeros((self.quadCount * 4, 3), 'f')
            self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')

            self.seqTrisVertices = numpy.zeros((self.trisCount * 3, 3), 'f')
            self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')

            all_idx = [el[0] for el in self.new_faces] + [el[0] for el in self.mod_faces]
            all_idx.sort()

            idx = 0
            for f in self.faces:
                for v in f:
                    vIndex += 1

                    if len(f) == 3:
                        self.seqTrisVertices[tIndex, 0] = self.vertices[v-1][0]
                        self.seqTrisVertices[tIndex, 1] = self.vertices[v-1][1]
                        self.seqTrisVertices[tIndex, 2] = self.vertices[v-1][2]
                        self.trisColors[tIndex, 0] = color_map[0 if idx in all_idx else 2][0]
                        self.trisColors[tIndex, 1] = color_map[0 if idx in all_idx else 2][1]
                        self.trisColors[tIndex, 2] = color_map[0 if idx in all_idx else 2][2]
                        tIndex += 1
                    elif len(f) == 4:
                        self.seqQuadVertices[qIndex, 0] = self.vertices[v-1][0]
                        self.seqQuadVertices[qIndex, 1] = self.vertices[v-1][1]
                        self.seqQuadVertices[qIndex, 2] = self.vertices[v-1][2]
                        self.quadColors[qIndex, 0] = color_map[0 if idx in all_idx else 2][0]
                        self.quadColors[qIndex, 1] = color_map[0 if idx in all_idx else 2][1]
                        self.quadColors[qIndex, 2] = color_map[0 if idx in all_idx else 2][2]
                        qIndex += 1
            print("Total reset took %f" % (time.time() - start))


        start = time.time()
        #perVertexNormals = defaultdict(lambda: [0, [0, 0, 0]])
        perVertexNormals = {}

        for f_idx in self.upd_faces:
            f = self.faces[f_idx[0]]
            normal = self.computeNormal([self.vertices[v-1] for v in f])
            for v in f:
                if v in perVertexNormals:
                    perVertexNormals[v][1] = [(perVertexNormals[v][1][0] * perVertexNormals[v][0] + normal[0]) / (perVertexNormals[v][0] + 1),
                                              (perVertexNormals[v][1][1] * perVertexNormals[v][0] + normal[1]) / (perVertexNormals[v][0] + 1),
                                              (perVertexNormals[v][1][2] * perVertexNormals[v][0] + normal[2]) / (perVertexNormals[v][0] + 1)]
                    perVertexNormals[v][0] += 1
                else:
                    perVertexNormals[v] = [None, None]
                    perVertexNormals[v][1] = normal
                    perVertexNormals[v][0] = 1
                    '''
                    perVertexNormals[v][1] = [(perVertexNormals[v][1][0] * perVertexNormals[v][0] + normal[0]) / (perVertexNormals[v][0] + 1),
                                              (perVertexNormals[v][1][1] * perVertexNormals[v][0] + normal[1]) / (perVertexNormals[v][0] + 1),
                                              (perVertexNormals[v][1][2] * perVertexNormals[v][0] + normal[2]) / (perVertexNormals[v][0] + 1)]
                    perVertexNormals[v][0] += 1
                    '''

        for f_idx in self.upd_faces:
            f = self.faces[f_idx[0]]
            k = 0
            if len(f) == 3:
                for v in f:
                    self.trisNormals[3 * f_idx[0] + k, 0] = perVertexNormals[v][1][0]
                    self.trisNormals[3 * f_idx[0] + k, 1] = perVertexNormals[v][1][1]
                    self.trisNormals[3 * f_idx[0] + k, 2] = perVertexNormals[v][1][2]
                    k += 1
            elif len(f) == 4:
                for v in f:
                    self.quadNormals[4 * f_idx[0] + k, 0] = perVertexNormals[v][1][0]
                    self.quadNormals[4 * f_idx[0] + k, 1] = perVertexNormals[v][1][1]
                    self.quadNormals[4 * f_idx[0] + k, 2] = perVertexNormals[v][1][2]
                    k += 1
        print("Update normals took %f" % (time.time() - start))

        start = time.time()
        self.vertices = numpy.asarray(self.vertices, dtype=numpy.float32)
        print("Verts to numpy took %f" % (time.time() - start))

        return True



if __name__ == "__main__":
    m = mMesh(False)
    m.loadOBJModel('../obj_files/task01/snap000001.obj', False)
    m.apply_diff(1, '../diff/task01/', False)
    #m.read_diff('../diff/task01/diff_1')
