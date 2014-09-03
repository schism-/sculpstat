import math
import numpy

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

    def loadModel(self, path):
        path_parts = path.split('.')
        ext = path_parts[-1]
        if (ext == 'obj' ) or (ext == 'OBJ'):
            print("Loading an OBJ model")
            self.loadOBJModel(path)
        elif (ext == 'off' ) or (ext == 'OFF'):
            print("Loading an OFF model")
            #self.loadOFFModel(path, '')

    def readOBJFile(self, file_path, quad):
        swapyz = False
        vertices = []
        normals = []
        texcoords = []
        faces = []
        quads = []
        tris = []
        material = None

        for line in open(file_path, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                vertices.append(v)
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)

                #self.faces.append((face, norms, texcoords, material))
                faces.append(face)

                if len(face) == 4:
                    quads.append(face)
                elif len(face) == 3:
                    tris.append(face)
            else:
                pass
            '''
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                normals.append(v)
            elif values[0] == 'vt':
                texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                pass
            elif values[0] == 'mtllib':
                pass
            '''
        return (vertices, faces, quads, tris)

    def computeNormal(self, temp):
        edge1 = [temp[0][0] - temp[1][0],
                 temp[0][1] - temp[1][1],
                 temp[0][2] - temp[1][2]]

        edge2 = [temp[0][0] - temp[2][0],
                 temp[0][1] - temp[2][1],
                 temp[0][2] - temp[2][2]]

        normal = [edge1[1] * edge2[2] - edge1[2] * edge2[1],
                  edge1[2] * edge2[0] - edge1[0] * edge2[2],
                  edge1[0] * edge2[1] - edge1[1] * edge2[0]]

        length = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
        if length > 0.0:
            normal = [normal[0] / length, normal[1] / length, normal[2] / length]
        else:
            normal = [0.0, 0.0, 0.0]

        return normal

    def loadOBJModel(self, path):

        self.__init__(True)

        quad = False
        if quad:
            mult = 4
        else:
            mult = 3

        path_parts = path.split('/')
        self.name = (path_parts[-1].split('.'))[-2]

        self.vertices, self.faces, self.quads, self.tris = self.readOBJFile(path, quad)

        self.vertexCount = len(self.vertices)
        #self.faceCount = len(self.faces)
        self.quadCount = len(self.quads)
        self.trisCount = len(self.tris)
        self.texCoordCount = len(self.vertices)
        #self.normalCount = len(self.vertices)
        self.normalQuadCount = len(self.quads) * 4
        self.normalTrisCount = len(self.tris) * 3


        #self.normals = numpy.zeros((self.faceCount * mult, 3), 'f')
        self.quadNormals = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.trisNormals = numpy.zeros((self.trisCount * 3, 3), 'f')
        self.texCoords = numpy.zeros((self.faceCount * mult, 2), 'f')

        print("Vertices detected: " + str(self.vertexCount) + " --> " + str(len(self.vertices)))
        print("Faces detected: " + str(self.faceCount))
        print("Quads detected: " + str(self.quadCount))
        print("Tris detected: " + str(self.trisCount))

        fIndex = 0
        qIndex = 0
        tIndex = 0
        vIndex = 0

        nIndex = 0
        ntIndex = 0
        nqIndex = 0

        #Initializing data for seq arrays
        #self.seqVertices = numpy.zeros((self.faceCount * 4, 3), 'f')
        self.seqQuadVertices = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.seqTrisVertices = numpy.zeros((self.trisCount * 3, 3), 'f')

        #self.colors = numpy.zeros((self.faceCount * 4, 3), 'f')
        self.quadColors = numpy.zeros((self.quadCount * 4, 3), 'f')
        self.trisColors = numpy.zeros((self.trisCount * 3, 3), 'f')

        print("Seq vertices: %i" % (len(self.seqVertices)))

        for f in self.faces:
            #Create a sequential array of vertices (for rendering)
            temp = []
            for v in f:
                #self.seqVertices[vIndex, 0] = self.vertices[v-1][0]
                #self.seqVertices[vIndex, 1] = self.vertices[v-1][1]
                #self.seqVertices[vIndex, 2] = self.vertices[v-1][2]

                #self.colors[vIndex, 0] = color_map[0][0]
                #self.colors[vIndex, 1] = color_map[0][1]
                #self.colors[vIndex, 2] = color_map[0][2]
                temp.append(self.vertices[v-1])
                vIndex += 1

                if len(f) == 3:
                    self.seqTrisVertices[tIndex, 0] = self.vertices[v-1][0]
                    self.seqTrisVertices[tIndex, 1] = self.vertices[v-1][1]
                    self.seqTrisVertices[tIndex, 2] = self.vertices[v-1][2]

                    self.trisColors[tIndex, 0] = color_map[0][0]
                    self.trisColors[tIndex, 1] = color_map[0][1]
                    self.trisColors[tIndex, 2] = color_map[0][2]

                    tIndex += 1
                elif len(f) == 4:
                    self.seqQuadVertices[qIndex, 0] = self.vertices[v-1][0]
                    self.seqQuadVertices[qIndex, 1] = self.vertices[v-1][1]
                    self.seqQuadVertices[qIndex, 2] = self.vertices[v-1][2]

                    self.quadColors[qIndex, 0] = color_map[1][0]
                    self.quadColors[qIndex, 1] = color_map[1][1]
                    self.quadColors[qIndex, 2] = color_map[1][2]

                    qIndex += 1

            normal = self.computeNormal(temp)

            #for _ in range(mult):
            #    self.normals[nIndex, 0] = normal[0]
            #    self.normals[nIndex, 1] = normal[1]
            #    self.normals[nIndex, 2] = normal[2]
            #    nIndex += 1

            if len(f) == 3:
                for _ in range(3):
                    self.trisNormals[ntIndex, 0] = normal[0]
                    self.trisNormals[ntIndex, 1] = normal[1]
                    self.trisNormals[ntIndex, 2] = normal[2]
                    ntIndex += 1
            elif len(f) == 4:
                for _ in range(4):
                    self.quadNormals[nqIndex, 0] = normal[0]
                    self.quadNormals[nqIndex, 1] = normal[1]
                    self.quadNormals[nqIndex, 2] = normal[2]
                    nqIndex += 1
            fIndex += 1

        #print("Normals: %d" % (len(self.normals)))
        print("Done")


    def buildVBOs (self):
        global g_fVBOSupported

        ''' Generate And Bind The Vertex Buffer '''
        if g_fVBOSupported:
            '''
            #self.VBOVertices = int(glGenBuffersARB( 1))                    # Get A Valid Name
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOVertices )       # Bind The Buffer
            # Load The Data
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.seqVertices, GL_STATIC_DRAW_ARB )

            # Generate And Bind The Texture Coordinate Buffer
            #self.VBOTexCoords = int(glGenBuffersARB( 1))
            #glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOTexCoords )
            # Load The Data
            #glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.texCoords, GL_STATIC_DRAW_ARB )

            self.VBONormals = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBONormals )
            # Load The Data
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.normals, GL_STATIC_DRAW_ARB )

            self.VBOColors = int(glGenBuffersARB( 1))
            glBindBufferARB( GL_ARRAY_BUFFER_ARB, self.VBOColors )
            # Load The Data
            glBufferDataARB( GL_ARRAY_BUFFER_ARB, self.colors, GL_STATIC_DRAW_ARB )

            #Our Copy Of The Data Is No Longer Necessary, It Is Safe In The Graphics Card
            #self.vertices = None
            #self.texCoords = None
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