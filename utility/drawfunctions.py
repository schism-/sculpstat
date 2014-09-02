from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

def drawAxis():
    glColor(1.0, 0.0, 0.0)
    glBegin(GL_LINE_STRIP)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.5, 0.0, 0.0)
    glEnd()
    glBegin(GL_LINE_STRIP)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.5, 0.0)
    glEnd()
    glBegin(GL_LINE_STRIP)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.5)
    glEnd()
    glColor(0.0, 0.0, 0.0)

def drawBBox(bbox):
    bbox_idx = [
                [0, 3, 2, 1],
                [4, 5, 6, 7],
                [0, 4, 7, 3],
                [1, 2, 6, 5],
                [0, 1, 5, 4],
                [2, 3, 7, 6]
               ]
    
    for face in bbox_idx:
        glBegin(GL_LINE_STRIP)
        for idx in face:
            glVertex3f(*bbox[idx])
        glVertex3f(*bbox[face[0]])   
        glEnd()

def drawCube():
    glBegin(GL_QUADS)

    # Front Face (note that the texture's corners have to match the quad's corners)
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0,  1.0)	# Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0,  1.0)	# Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0,  1.0)	# Top Right Of The Texture and Quad
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0,  1.0)	# Top Left Of The Texture and Quad
	
    # Back Face
    glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, -1.0)	# Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0, 1.0); glVertex3f(-1.0,  1.0, -1.0)	# Top Right Of The Texture and Quad
    glTexCoord2f(0.0, 1.0); glVertex3f( 1.0,  1.0, -1.0)	# Top Left Of The Texture and Quad
    glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0, -1.0)	# Bottom Left Of The Texture and Quad
	
    # Top Face
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -1.0)	# Top Left Of The Texture and Quad
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0,  1.0)	# Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0, 0.0); glVertex3f( 1.0,  1.0,  1.0)	# Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -1.0)	# Top Right Of The Texture and Quad
	
    # Bottom Face	   
    glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, -1.0, -1.0)	# Top Right Of The Texture and Quad
    glTexCoord2f(0.0, 1.0); glVertex3f( 1.0, -1.0, -1.0)	# Top Left Of The Texture and Quad
    glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0)	# Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0,  1.0)	# Bottom Right Of The Texture and Quad
	
    # Right face
    glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, -1.0)	# Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -1.0)	# Top Right Of The Texture and Quad
    glTexCoord2f(0.0, 1.0); glVertex3f( 1.0,  1.0,  1.0)	# Top Left Of The Texture and Quad
    glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0)	# Bottom Left Of The Texture and Quad
	
    # Left Face
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0)	# Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0,  1.0)	# Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0, 1.0); glVertex3f(-1.0,  1.0,  1.0)	# Top Right Of The Texture and Quad
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -1.0)	# Top Left Of The Texture and Quad
	
    glEnd();				# Done Drawing The Cube

def drawTriangle():
    glBegin(GL_POLYGON)
    glVertex3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, 0.0)
    glVertex3f(-1.0, -1.0, 0.0)
    glEnd()

def drawSquare():
    glBegin(GL_QUADS)
    glVertex3f(-1.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, 0.0)
    glVertex3f(-1.0, -1.0, 0.0)
    glEnd()

