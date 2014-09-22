__author__ = 'christian'

import math
import numpy
import numpy.random
import pickle
import time
import nearpy

from OpenGL.GL.ARB.vertex_buffer_object import *

g_fVBOSupported = False  # ARB_vertex_buffer_object supported?

color_map = [[0.6, 0.7, 0.7],
             [0.8, 0.7, 0.7]]

class mMesh:
    def __init__(self, vbo):

        global g_fVBOSupported
        g_fVBOSupported = vbo

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
        print("Loading an OBJ model")
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
        self.trisNormals = numpy.zeros((self.trisCount * 3, 3), 'f')

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


    def buildVBOs (self):
        global g_fVBOSupported

        ''' Generate And Bind The Vertex Buffer '''
        if g_fVBOSupported:
            '''
            # Generate And Bind The Texture Coordinate Buffer
            #self.VBOTexCoords = int(glGenBuffersARB( 1))
            #glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOTexCoords )
            # Load The Data
            #glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.texCoords, GL_STATIC_DRAW_ARB )
            '''

            self.VBOQuadVertices = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOQuadVertices )
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.seqQuadVertices, GL_STATIC_DRAW_ARB )

            self.VBOQuadNormals = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOQuadNormals )
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.quadNormals, GL_STATIC_DRAW_ARB )

            self.VBOQuadColors = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOQuadColors )
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.quadColors, GL_STATIC_DRAW_ARB )

            self.VBOTrisVertices = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOTrisVertices )
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.seqTrisVertices, GL_STATIC_DRAW_ARB )

            self.VBOTrisNormals = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOTrisNormals )
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.trisNormals, GL_STATIC_DRAW_ARB )

            self.VBOTrisColors = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOTrisColors )
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.trisColors, GL_STATIC_DRAW_ARB )

if __name__ == "__main__":
    m = mMesh(True)
    m.loadModel(m)