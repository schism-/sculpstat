__author__ = 'christian'

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

def load_vertex_pointer(pointer):
    pointer.bind()
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, pointer)

def load_normal_pointer(pointer):
    pointer.bind()
    glEnableClientState(GL_NORMAL_ARRAY)
    glNormalPointer(GL_FLOAT, 0, pointer)

def load_color_pointer(pointer):
    pointer.bind()
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, pointer)

def disable_quad(m):
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    if m.VBOQuadColors:
        glDisableClientState(GL_COLOR_ARRAY)

def disable_tris(m):
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    if m.VBOTrisColors:
        glDisableClientState(GL_COLOR_ARRAY)