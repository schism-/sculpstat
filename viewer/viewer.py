__author__ = 'christian'

from OpenGL.arrays import vbo
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from utility.mouseInteractor import MouseInteractor
from utility.drawfunctions import *
from utility.mmesh import *
from numpy import *
from time import time


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

#GUI Variables
thread = None
buttonThread = None

#Render variables
drawContactPoints = False
drawContactTriangles = False
draw_gui = False

'''
    =========== MAIN FUNCTIONS ============
'''

def loadComponent(m, path, segment_number):
    m.loadAsComponent(path, segment_number)

def loadVBO(m):
    m.VBOVertices = vbo.VBO(m.seqVertices)
    m.VBONormals = vbo.VBO(m.normals)
    m.VBOColors = vbo.VBO(m.colors)

def loadModel(m, path):
    m.loadModel(path)
    m.VBOVertices = vbo.VBO(m.seqVertices)
    m.VBONormals = vbo.VBO(m.normals)
    m.VBOColors = vbo.VBO(m.colors)

def drawModel(m, quadric):
    if m.VBOVertices is not None:
        m.VBOVertices.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, m.VBOVertices)

        m.VBONormals.bind()
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, m.VBONormals)

        if (m.VBOColors is not None):
            m.VBOColors.bind()
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, 0, m.VBOColors)
        else:
            glDisableClientState(GL_COLOR_ARRAY)
            pass

        glDrawArrays(GL_TRIANGLES, 0, len(m.seqVertices))
        #glDrawArrays(GL_QUADS, 0, len(m.seqVertices))

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

def loadFinalCandidate(mesh_path):
    global meshes
    meshes = [None, ] * 1
    meshes[0] = mMesh(g_fVBOSupported)
    loadModel(meshes[0], mesh_path)
    meshes = [ m for m in meshes if m is not None ]


def changeCand():
    global meshes, meshes_loaded, gui_objects

    gui_objects[1].setIndices(indices)
    loadFinalCandidate()

    for x in range(len(meshes)):
        loadVBO(meshes[x])
    meshes_loaded = []
    for x in range(len(meshes)):
        meshes_loaded.append(meshes[x])

def init(filepath):

    global g_fVBOSupported, meshes_loaded, gui_objects, mouseInteractor, quadric

    #Check for VBOs Support
    g_fVBOSupported = False
    quadric = gluNewQuadric()

    glClearColor(0.1, 0.1, 0.2, 0.0)

    #Define openGL rendering behaviours
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)

    initLightning()

    mouseInteractor = MouseInteractor( .01, 1 , gui_objects)

    #LOAD MODEL
    start = time()

    loadFinalCandidate(filepath)

    for x in range(len(meshes)):
        loadVBO(meshes[x])

    for x in range(len(meshes)):
        meshes_loaded.append(meshes[x])

    print("Models loaded in %f" %(time() - start))
    print()


def debugStuff():
    pass


def drawScene():

    global gui_objects, meshes_loaded, quadric, draw_gui

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)

    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    xSize, ySize = glutGet( GLUT_WINDOW_WIDTH ), glutGet( GLUT_WINDOW_HEIGHT )
    gluPerspective(60, float(xSize) / float(ySize), 0.1, 1000)
    glMatrixMode( GL_MODELVIEW )
    glLoadIdentity()

    glTranslatef( 0, 0, -10 )
    mouseInteractor.applyTransformation()

    #Draw axis (for reference)
    drawAxis()

    #Draw all the stuff here
    for m in meshes_loaded:
        glPushMatrix()
        drawModel(m, quadric)
        if (False):
            drawBBoxes(m)
        glPopMatrix()

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

def mainLoop():
    glutInit(sys.argv)
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH )
    glutInitWindowSize(*SCREEN_SIZE)
    glutInitWindowPosition(1000, 200)

    window = glutCreateWindow("obj viewer 0.1")

    init("../obj_files/gargoyle2/snap001058.obj")
    mouseInteractor.registerCallbacks()

    glutDisplayFunc(drawScene)
    glutIdleFunc(drawScene)
    glutReshapeFunc(resizeWindow)

    glutMainLoop()

if __name__ == "__main__":
    mainLoop()
