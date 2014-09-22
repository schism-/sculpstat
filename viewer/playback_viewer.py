__author__ = 'christian'

import json

from numpy import *
from OpenGL.arrays import vbo
from scipy.spatial import KDTree

from utility.mmesh import *
from utility.drawfunctions import *
from utility.mouseInteractor import MouseInteractor

class Viewer(object):

    def __init__(self, model_name):
        self.color_map = [[0.7, 0.7, 0.7]]

        self.g_fVBOSupported = True
        self.ESCAPE = '\033'  # Octal value for 'esc' key
        self.SCREEN_SIZE = (800, 600)
        self.SHAPE = ''
        self.lastx=0
        self.lasty=0

        self.meshes = []

        self.mouseInteractor = None

        # Rendered objects
        self.meshes_loaded = []
        self.gui_objects = []

        self.brush_paths = []
        self.brush_paths_colors = []

        #GUI Variables
        self.window = None
        self.thread = None
        self.buttonThread = None

        #Render variables
        self.draw_gui = False
        self.load_brushes = True
        self.is_numpy = False

        self.obj_path = "../obj_files/" + model_name + "/snap" + str(0).zfill(6) + ".obj"
        self.blend_path = "../blend_files/" + model_name + "/snap" + str(0).zfill(6) + ".blend"
        self.numpy_path = "../nupy_data/" + model_name + "/snap" + str(0).zfill(6) + "/"
        self.step_path = "../steps/" + model_name + "/steps.json"

        self.current_step = 0

        self.brush_paths = []
        self.brush_paths_colors = []

    def init(self):
        glClearColor(0.1, 0.1, 0.2, 0.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        self.initLightning()

        self.mouseInteractor = MouseInteractor( .01, 1 , self.gui_objects)

        #LOAD MODEL
        start = time.time()
        start_lfc = time.time()
        self.loadFinalCandidate(self.obj_path)
        print("Models loaded in %f" %(time.time() - start_lfc))

        if self.load_brushes:
            start_bs =time.time()
            self.loadBrushStrokes(self.step_path, self.current_step)
            print("Brush loaded in %f" %(time.time() - start_bs))

        for x in range(len(self.meshes)):
            self.meshes_loaded.append(self.meshes[x])

        print("Models loaded in %f" %(time.time() - start))
        print()

    def initLightning(self):
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

    def mainLoop(self):
        glutInit(sys.argv)
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH )
        glutInitWindowSize(*self.SCREEN_SIZE)
        glutInitWindowPosition(1000, 200)
        self.window = glutCreateWindow("obj viewer 0.1")
        self.init()
        self.mouseInteractor.registerCallbacks()
        glutDisplayFunc(self.drawScene)
        glutIdleFunc(self.drawScene)
        glutReshapeFunc(self.resizeWindow)
        glutKeyboardFunc(self.keyboardPressed)
        glutKeyboardUpFunc(self.keyboardUp)
        glutMainLoop()

    def loadFinalCandidate(self, mesh_path):
        self.meshes = [None,] * 1
        self.meshes[0] = mMesh(self.g_fVBOSupported)
        self.loadModel(self.meshes[0], mesh_path)
        self.meshes = [m for m in self.meshes if m is not None ]


    def loadModel(self, m, path):
        m.loadModel(path, self.load_brushes, self.is_numpy)
        m.VBOQuadVertices = vbo.VBO(m.seqQuadVertices)
        m.VBOTrisVertices = vbo.VBO(m.seqTrisVertices)

        m.VBOQuadNormals = vbo.VBO(m.quadNormals)
        m.VBOTrisNormals = vbo.VBO(m.trisNormals)

        m.VBOQuadColors = vbo.VBO(m.quadColors)
        m.VBOTrisColors = vbo.VBO(m.trisColors)


    def loadBrushStrokes(self, step_path, stepno, window=None):
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
                    path = getPath(stroke_op)
                    col = [random.random(), random.random(), random.random()]
                    self.brush_paths.append(path)
                    self.brush_paths_colors.append(col)
            except KeyError as e:
                print("Step not found")
                print(e)
            except TypeError as e:
                print("ERROR")
                print(e)
        print("outer loop in %f: " % (time.time() - start_outer))


    def getPath(self, stroke_op):
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

    def drawScene(self):
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
        self.mouseInteractor.applyTransformation()

        #Draw all the stuff here
        if self.mouseInteractor.drawMeshes:
            for m in self.meshes_loaded:
                glPushMatrix()
                self.drawModel(m)
                if (False):
                    self.drawBBoxes(m)
                glPopMatrix()

        p_idx = 0
        for p in self.brush_paths:
            glPushMatrix()
            self.drawBrushPath(p, p_idx)
            glPopMatrix()
            p_idx += 1

        #Draw all the interface here
        if self.draw_gui:
            glDisable( GL_LIGHTING )
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, float(xSize), float(ySize), 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            for obj in self.gui_objects:
                obj.draw()
            glEnable( GL_LIGHTING )
        glutSwapBuffers()

    def resizeWindow(self, width, height):
        if height == 0:
            height = 1

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0 , float(width)/float(height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)


    def loadNextModel(self):
        pass

    def loadPrevModel(self):
        pass


    def keyboardPressed(self, key, x, y):
        if key == b'z' and not self.mouseInteractor.zooming:
            self.mouseInteractor.zooming = True
        if key == b'a' and self.mouseInteractor.drawMeshes:
            self.mouseInteractor.drawMeshes = False
        if key == b'q':
            self.loadPrevModel()
        if key == b'q':
            self.loadNextModel()

    def keyboardUp(self, key, x, y):
        if key == b'z' and self.mouseInteractor.zooming:
            self.mouseInteractor.zooming = False
        if key == b'a' and not self.mouseInteractor.drawMeshes:
            self.mouseInteractor.drawMeshes = True

    def drawModel(self, m):
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

    def drawBrushPath(self, path, idx):
        glColor3f(*self.brush_paths_colors[idx])
        glLineWidth(4.0)
        glBegin(GL_LINES)
        for k in range(len(path) - 1):
            glVertex3f(path[k][0], path[k][1], path[k][2])
            glVertex3f(path[k+1][0], path[k+1][1], path[k+1][2])
        glEnd()

        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(1.0)


    def drawBBoxes(self, m):
        drawBBox(m.bbox)


    def loadVBO(self, m):
        m.VBOVertices = vbo.VBO(m.seqVertices)
        m.VBONormals = vbo.VBO(m.normals)
        m.VBOColors = vbo.VBO(m.colors)


    def changeCand(self):
        self.gui_objects[1].setIndices(indices)
        self.loadFinalCandidate()
        for x in range(len(self.meshes)):
            self.loadVBO(meshes[x])
        self.meshes_loaded = []
        for x in range(len(self.meshes)):
            self.meshes_loaded.append(self.meshes[x])



if __name__ == "__main__":
    v = Viewer("task01")
    v.mainLoop()
    #mainLoop(model_name = "task02", stepno = 2619, stepwindow = None, loadB = False, isNumpy = False)
    #mainLoop(model_name = "gargoyle2", stepno = 1058, stepwindow = None, loadB = False, isNumpy = False)
    #mainLoop(model_name = "monster", stepno = 925, stepwindow = 10, loadB = True, isNumpy = False)