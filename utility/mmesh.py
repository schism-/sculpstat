import math
import numpy

from OpenGL.GL.ARB.vertex_buffer_object import *

g_fVBOSupported = False  # ARB_vertex_buffer_object supported?

color_map = [[0.7, 0.7, 0.7]]

class mMesh:
    def __init__(self, vbo):

        global g_fVBOSupported
        g_fVBOSupported = vbo

        self.name = ''

        self.vertexCount = 0
        self.faceCount = 0
        self.texCoordCount = 0
        self.normalCount = 0

        self.vertices = None
        self.verticesAsString = None
        self.seqVertices = []

        self.texCoords = None
        self.texCoordsAsString = None

        self.normals = None
        self.faces = None
        self.textureId = None
        self.colors = None

        self.segments = {}              # Array of mSegment
        self.components = {}
        self.adjacency_matrix = {}

        self.VBOVertices = None
        self.VBOTexCoords = None
        self.VBONormals = None
        self.VBOColors = None

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
                if quad and len(face) == 4:
                    faces.append(face)
                elif not quad and len(face) == 3:
                    faces.append(face)
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
        return (vertices, faces)

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

        self.vertices, self.faces = self.readOBJFile(path, quad)

        self.vertexCount = len(self.vertices)
        self.faceCount = len(self.faces)
        self.texCoordCount = len(self.vertices)
        self.normalCount = len(self.vertices)

        self.normals = numpy.zeros((self.faceCount * mult, 3), 'f')
        self.texCoords = numpy.zeros((self.faceCount * mult, 2), 'f')

        print("Vertices detected: " + str(self.vertexCount) + " --> " + str(len(self.vertices)))
        print("Faces detected: " + str(self.faceCount))

        fIndex = 0
        vIndex = 0
        nIndex = 0

        #Initializing data for seq arrays
        self.seqVertices = numpy.zeros((self.faceCount * mult, 3), 'f')
        self.colors = numpy.zeros((self.faceCount * mult, 3), 'f')
        self.normals = numpy.zeros((self.faceCount * mult, 3), 'f')

        print("Seq vertices: %i" % (len(self.seqVertices)))

        for f in self.faces:
            #Create a sequential array of vertices (for rendering)
            temp = []
            for v in f:
                self.seqVertices[vIndex, 0] = self.vertices[v-1][0]
                self.seqVertices[vIndex, 1] = self.vertices[v-1][1]
                self.seqVertices[vIndex, 2] = self.vertices[v-1][2]

                self.colors[vIndex, 0] = color_map[0][0]
                self.colors[vIndex, 1] = color_map[0][1]
                self.colors[vIndex, 2] = color_map[0][2]
                temp.append(self.vertices[v-1])
                vIndex += 1

            normal = self.computeNormal(temp)

            for _ in range(mult):
                self.normals[nIndex, 0] = normal[0]
                self.normals[nIndex, 1] = normal[1]
                self.normals[nIndex, 2] = normal[2]
                nIndex += 1

            fIndex += 1
        print("Normals: %d" % (len(self.normals)))
        print("Done")


    def buildVBOs (self):
        global g_fVBOSupported

        ''' Generate And Bind The Vertex Buffer '''
        if g_fVBOSupported:
            self.VBOVertices = int(glGenBuffersARB( 1))                    # Get A Valid Name
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

if __name__ == "__main__":
    m = mMesh(True)
    m.loadModel(m)