__author__ = 'christian'

import time
import json
from numpy import *
from utility.mmesh import *
from OpenGL.arrays import vbo
from scipy.spatial import KDTree
from utility.drawfunctions import *
from utility.mouseInteractor import MouseInteractor

'''
    ========== GLOBAL VARIABLES & CONSTANTS ==========
'''
color_map = [[0.7, 0.7, 0.7]]

ESCAPE = '\033'  # Octal value for 'esc' key
SCREEN_SIZE = (800, 600)
SHAPE = ''
lastx=0
lasty=0

meshes = []

# Rendered objects
meshes_loaded = []
gui_objects = []

brush_paths = []
brush_paths_colors = []

#GUI Variables
thread = None
buttonThread = None

#Render variables
drawContactPoints = False
drawContactTriangles = False
draw_gui = False

loadBrushes = True

'''
    =========== MAIN FUNCTIONS ============
'''

def loadComponent(m, path, segment_number):
    m.loadAsComponent(path, segment_number)

def loadVBO(m):
    m.VBOVertices = vbo.VBO(m.seqVertices)
    m.VBONormals = vbo.VBO(m.normals)
    m.VBOColors = vbo.VBO(m.colors)

def loadModel(m, path, loadBrushes, isNumpy):
    m.loadModel(path, loadBrushes, isNumpy)
    m.VBOQuadVertices = vbo.VBO(m.seqQuadVertices)
    m.VBOTrisVertices = vbo.VBO(m.seqTrisVertices)

    m.VBOQuadNormals = vbo.VBO(m.quadNormals)
    m.VBOTrisNormals = vbo.VBO(m.trisNormals)

    m.VBOQuadColors = vbo.VBO(m.quadColors)
    m.VBOTrisColors = vbo.VBO(m.trisColors)


def drawModel(m):
    if m.VBOQuadVertices is not None:
        m.VBOQuadVertices.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, m.VBOQuadVertices)

        m.VBOQuadNormals.bind()
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, m.VBOQuadNormals)

        if (m.VBOQuadColors is not None):
            m.VBOQuadColors.bind()
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, m.VBOQuadColors)
        else:
            glDisableClientState(GL_COLOR_ARRAY)
            pass

        glDrawArrays(GL_QUADS, 0, len(m.seqQuadVertices))

    if m.VBOTrisVertices is not None:
        m.VBOTrisVertices.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, m.VBOTrisVertices)

        m.VBOTrisNormals.bind()
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, m.VBOTrisNormals)

        if (m.VBOTrisColors is not None):
            m.VBOTrisColors.bind()
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, m.VBOTrisColors)
        else:
            glDisableClientState(GL_COLOR_ARRAY)
            pass

        glDrawArrays(GL_TRIANGLES, 0, len(m.seqTrisVertices))

def drawBrushPath(path, idx):
    global brush_paths_colors
    glColor3f(*brush_paths_colors[idx])
    glLineWidth(4.0)
    glBegin(GL_LINES)
    for k in range(len(path) - 1):
        glVertex3f(path[k][0], path[k][1], path[k][2])
        glVertex3f(path[k+1][0], path[k+1][1], path[k+1][2])
    glEnd()

    glColor3f(0.0, 0.0, 0.0)
    glLineWidth(1.0)

def drawBBoxes(m):
    drawBBox(m.bbox)

def initLightning():
    lP = 7
    lA = 0.1
    lD = 0.2
    lS = 0.3
    glEnable( GL_LIGHTING )
    glEnable( GL_LIGHT0 )
    glLightModelfv( GL_LIGHT_MODEL_AMBIENT, [0.8, 0.8, 0.8, 1] )
    glLightfv( GL_LIGHT0, GL_POSITION, [lP, lP, lP, 1] )
    glLightfv( GL_LIGHT0, GL_AMBIENT, [lA, lA, lA, 1] )
    glLightfv( GL_LIGHT0, GL_DIFFUSE, [lD, lD, lD, 1] )
    glLightfv( GL_LIGHT0, GL_SPECULAR, [lS, lS, lS, 1] )

    glEnable( GL_LIGHT1 )
    glLightModelfv( GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1] )
    glLightfv( GL_LIGHT1, GL_POSITION, [-lP, lP, lP, 1] )
    glLightfv( GL_LIGHT1, GL_AMBIENT, [lA, lA, lA, 1] )
    glLightfv( GL_LIGHT1, GL_DIFFUSE, [lD, lD, lD, 1] )
    glLightfv( GL_LIGHT1, GL_SPECULAR, [lS, lS, lS, 1] )

    glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, [lA, lA, lA, 1] )
    glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, [lA, lA, lA, 1] )
    glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, [0.8, 0.8, 0.8, 1] )
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 20 )

def loadFinalCandidate(mesh_path, loadBrushes, isNumpy):
    global meshes
    meshes = [None,] * 1
    meshes[0] = mMesh(g_fVBOSupported)
    loadModel(meshes[0], mesh_path, loadBrushes, isNumpy)
    meshes = [m for m in meshes if m is not None ]


def changeCand():
    global meshes, meshes_loaded, gui_objects
    gui_objects[1].setIndices(indices)
    loadFinalCandidate()
    for x in range(len(meshes)):
        loadVBO(meshes[x])
    meshes_loaded = []
    for x in range(len(meshes)):
        meshes_loaded.append(meshes[x])

def init(model_name, stepno, window=None, isNumpy=False):

    global g_fVBOSupported, meshes_loaded, gui_objects, mouseInteractor, quadric, loadBrushes

    if isNumpy:
        obj_path = "../numpy_data/" + model_name + "/snap" + str(stepno).zfill(6) + "/"
    else:
        obj_path = "../obj_files/" + model_name + "/snap" + str(stepno).zfill(6) + ".obj"

    step_path = "../steps/" + model_name + "/steps.json"

    glClearColor(0.1, 0.1, 0.2, 0.0)

    #Check for VBOs Support
    g_fVBOSupported = False
    quadric = gluNewQuadric()
    #Define openGL rendering behaviours
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    initLightning()

    mouseInteractor = MouseInteractor( .01, 1 , gui_objects)

    #LOAD MODEL
    start = time.time()

    start_lfc = time.time()
    loadFinalCandidate(obj_path, loadBrushes, isNumpy)
    print("Models loaded in %f" %(time.time() - start_lfc))

    if loadBrushes:
        start_bs =time.time()
        loadBrushStrokes(step_path, stepno, window)
        print("Brush loaded in %f" %(time.time() - start_bs))

    for x in range(len(meshes)):
        meshes_loaded.append(meshes[x])

    print("Models loaded in %f" %(time.time() - start))
    print()


def loadBrushStrokes(step_path, stepno, window=None):

    global brush_paths, brush_paths_colors, meshes

    f = open(step_path, 'r')
    step_file = json.load(f)

    w = 1 if not window else window
    start_outer = time.time()
    for k in range(w):
        try:
            step_ops = step_file[str(stepno - k)]
            stroke_op = None
            for op in step_ops:
                if op["op_name"] == "bpy.ops.sculpt.brush_stroke":
                    stroke_op = op
                    break
            if stroke_op:
                start_path = time.time()
                path = getPath(stroke_op)
                #print("path got in %f: " % (time.time() - start_path))
                col = [random.random(), random.random(), random.random()]
                for p in path:
                    start_n = time.time()
                    neighbours = getNeighbours(meshes[0], p)
                    start_c = time.time()
                    updateColors(neighbours, col)
                    #print("\t path loop in (%f, %f): " % (time.time() - start_n, time.time() - start_c))

                brush_paths.append(path)
                brush_paths_colors.append(col)
        except KeyError as e:
            print("Step not found")
            print(e)
        except TypeError as e:
            print("ERROR")
            print(e)
    print("outer loop in %f: " % (time.time() - start_outer))

def getPath(stroke_op):
    path = numpy.zeros((len(stroke_op["stroke"]), 3), 'f')
    idx = 0
    zeroes = 0
    for point in stroke_op["stroke"]:
        if abs(point["location"][0]) < 200 and abs(point["location"][1]) < 200 and abs(point["location"][2]) < 200:
            path[idx] = [point["location"][0], point["location"][2], -1.0 * point["location"][1]]
            idx += 1
        else:
            zeroes += 1
    if zeroes > 0:
        path = path[:-zeroes]
    return path

def getNeighbours(mesh, point):
    n = []
    if False:
        n = mesh.getNeighbours(point)
    else:
        NDIM = 3
        a = mesh.vertices
        a.shape = a.size / NDIM, NDIM
        tree = KDTree(a, leafsize=a.shape[0]+1)
        distances, ndx = tree.query([point], k = 50)
        for k in range(len(ndx[0])):
            n.append([a[ndx[0][k]], ndx[0][k], distances[0][k]])
    return n

def updateColors(neighbours, col):
    global meshes
    for n in neighbours:
        if n[2] < 0.2:
            try:
                idx = meshes[0].seqTrisMap[int(n[1])]
                for i in idx:
                    meshes[0].trisColors[i, 0] = col[0]
                    meshes[0].trisColors[i, 1] = col[1]
                    meshes[0].trisColors[i, 2] = col[2]
            except KeyError:
                pass

            try:
                idx = meshes[0].seqQuadMap[int(n[1])]
                for i in idx:
                    meshes[0].quadColors[i, 0] = col[0]
                    meshes[0].quadColors[i, 1] = col[1]
                    meshes[0].quadColors[i, 2] = col[2]
            except KeyError:
                pass

def debugStuff():
    pass


def drawScene():
    global gui_objects, meshes_loaded, quadric, draw_gui, brush_paths, mouseInteractor

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)

    glShadeModel(GL_SMOOTH)

    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    xSize, ySize = glutGet( GLUT_WINDOW_WIDTH ), glutGet( GLUT_WINDOW_HEIGHT )
    gluPerspective(60, float(xSize) / float(ySize), 0.1, 1000)
    glMatrixMode( GL_MODELVIEW )
    glLoadIdentity()

    glTranslatef( 0, 0, -5 )
    mouseInteractor.applyTransformation()

    #Draw axis (for reference)
    drawAxis()

    #Draw all the stuff here
    if mouseInteractor.drawMeshes:
        for m in meshes_loaded:
            glPushMatrix()
            drawModel(m)
            if (False):
                drawBBoxes(m)
            glPopMatrix()

    p_idx = 0
    for p in brush_paths:
        glPushMatrix()
        drawBrushPath(p, p_idx)
        glPopMatrix()
        p_idx += 1

    #Draw all the interface here
    if draw_gui:
        glDisable( GL_LIGHTING )
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, float(xSize), float(ySize), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        for obj in gui_objects:
            obj.draw()
        glEnable( GL_LIGHTING )
    glutSwapBuffers()

def resizeWindow(width, height):
    if height == 0:
        height = 1

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0 , float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def mainLoop(model_name, stepno, stepwindow=None, loadB=True, isNumpy=False):
    global loadBrushes
    if loadB:
        loadBrushes = True
    else:
        loadBrushes = False

    glutInit(sys.argv)
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH )
    glutInitWindowSize(*SCREEN_SIZE)
    glutInitWindowPosition(1000, 200)
    window = glutCreateWindow("obj viewer 0.1")
    init(model_name, stepno, stepwindow, isNumpy)
    mouseInteractor.registerCallbacks()
    glutDisplayFunc(drawScene)
    glutIdleFunc(drawScene)
    glutReshapeFunc(resizeWindow)

    glutMainLoop()

if __name__ == "__main__":
    #mainLoop(model_name = "task01", stepno = 1520, stepwindow = 5, loadB = True, isNumpy = True)
    #mainLoop(model_name = "task02", stepno = 2619, stepwindow = None, loadB = False, isNumpy = False)
    #mainLoop(model_name = "gargoyle2", stepno = 1058, stepwindow = None, loadB = False, isNumpy = False)
    mainLoop(model_name = "monster", stepno = 925, stepwindow = 2, loadB = True, isNumpy = False)