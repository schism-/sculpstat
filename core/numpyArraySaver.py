__author__ = 'christian'

import os
import math
import numpy
import pickle
import os.path
import utility.mmesh as mmesh
import utility.common as common

color_map = [[0.6, 0.7, 0.7],
             [0.8, 0.7, 0.7]]

def readOBJFile(file_path):
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

def loadOBJModel(path, loadBrushes):
    vertices, faces, quads, tris = readOBJFile(path)

    vertexCount = len(vertices)
    faceCount = len(faces)
    quadCount = len(quads)
    trisCount = len(tris)

    print("Vertices detected: " + str(vertexCount) + " --> " + str(len(vertices)))
    print("Faces detected: " + str(faceCount))
    print("Quads detected: " + str(quadCount))
    print("Tris detected: " + str(trisCount))

    fIndex = 0
    qIndex = 0
    tIndex = 0
    vIndex = 0
    ntIndex = 0
    nqIndex = 0

    #Initializing data for seq arrays
    seqQuadVertices = numpy.zeros((quadCount * 4, 3), 'f')
    seqTrisVertices = numpy.zeros((trisCount * 3, 3), 'f')

    quadColors = numpy.zeros((quadCount * 4, 3), 'f')
    trisColors = numpy.zeros((trisCount * 3, 3), 'f')

    quadNormals = numpy.zeros((quadCount * 4, 3), 'f')
    trisNormals = numpy.zeros((trisCount * 3, 3), 'f')

    seqTrisMap = {}
    seqQuadMap = {}

    for f in faces:
        #Create a sequential array of vertices (for rendering)
        temp = []
        for v in f:
            temp.append(vertices[v-1])
            vIndex += 1

            if len(f) == 3:
                seqTrisVertices[tIndex, 0] = vertices[v-1][0]
                seqTrisVertices[tIndex, 1] = vertices[v-1][1]
                seqTrisVertices[tIndex, 2] = vertices[v-1][2]

                trisColors[tIndex, 0] = color_map[0][0]
                trisColors[tIndex, 1] = color_map[0][1]
                trisColors[tIndex, 2] = color_map[0][2]

                if loadBrushes:
                    if v-1 not in seqTrisMap:
                        seqTrisMap[v-1] = [tIndex]
                    else:
                        seqTrisMap[v-1].append(tIndex)
                tIndex += 1
            elif len(f) == 4:
                seqQuadVertices[qIndex, 0] = vertices[v-1][0]
                seqQuadVertices[qIndex, 1] = vertices[v-1][1]
                seqQuadVertices[qIndex, 2] = vertices[v-1][2]

                quadColors[qIndex, 0] = color_map[1][0]
                quadColors[qIndex, 1] = color_map[1][1]
                quadColors[qIndex, 2] = color_map[1][2]

                if loadBrushes:
                    if v-1 not in seqQuadMap:
                        seqQuadMap[v-1] = [qIndex]
                    else:
                        seqQuadMap[v-1].append(qIndex)

                qIndex += 1

        normal = computeNormal(temp)
        if len(f) == 3:
            for _ in range(3):
                trisNormals[ntIndex, 0] = normal[0]
                trisNormals[ntIndex, 1] = normal[1]
                trisNormals[ntIndex, 2] = normal[2]
                ntIndex += 1
        elif len(f) == 4:
            for _ in range(4):
                quadNormals[nqIndex, 0] = normal[0]
                quadNormals[nqIndex, 1] = normal[1]
                quadNormals[nqIndex, 2] = normal[2]
                nqIndex += 1
        fIndex += 1

    saveData(path, seqQuadVertices, seqTrisVertices, seqQuadMap, seqTrisMap, quadColors, quadNormals, trisColors, trisNormals)

    print("Done")

def computeNormal(temp):
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

def saveData(path, seqQuadVertices, seqTrisVertices, seqQuadMap, seqTrisMap, quadColors, quadNormals, trisColors, trisNormals):
    print("Saving numpy data for: %s" % path)

    path_parts = path.split('/')
    model_name = path_parts[-2]
    model_step_name = (path_parts[-1].split('.'))[-2]

    print("Saving numpy data for: %s" % path)
    print("Model: %s " % model_name)
    print("Step: %s" % model_step_name)

    save_dir = "../numpy_data/" + model_name + "/" + model_step_name + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    numpy.save(save_dir + "seqquadverts", seqQuadVertices)
    numpy.save(save_dir + "seqtrisverts", seqTrisVertices)
    numpy.save(save_dir + "quadcolors", quadColors)
    numpy.save(save_dir + "triscolors", trisColors)
    numpy.save(save_dir + "quadnormals", quadNormals)
    numpy.save(save_dir + "trisnormals", trisNormals)

    fSQM = open(save_dir + "seqquadmap", "wb+")
    pickle.dump(seqQuadMap, fSQM)
    fSQM.close()

    fSTM = open(save_dir + "seqtrismap", "wb+")
    pickle.dump(seqTrisMap, fSTM)
    fSTM.close()


def parse_dir(files_path):
    onlyfiles = common.get_files_from_directory(files_path, ['obj'])

    for file in onlyfiles:
        print(file)
        loadOBJModel(file[0], True)

if __name__ == "__main__":
    obj_files_path = "../obj_files/monster"

    parse_dir(obj_files_path)