__author__ = 'christian'

from OpenGL.GL import *

class InteractionMatrix ( object ):
    def __init__( self ):
        self.__currentMatrix = None
        self.reset( )

    def reset( self ):
        glPushMatrix( )
        glLoadIdentity( )
        self.__currentMatrix = glGetFloatv( GL_MODELVIEW_MATRIX )
        glPopMatrix( )

    def addTranslation( self, tx, ty, tz ):
        glPushMatrix( )
        glLoadIdentity( )
        glTranslatef(tx, ty, tz)
        glMultMatrixf( self.__currentMatrix )
        self.__currentMatrix = glGetFloatv( GL_MODELVIEW_MATRIX )
        glPopMatrix( )

    def addRotation( self, ang, rx, ry, rz ):
        glPushMatrix( )
        glLoadIdentity( )
        glRotatef(ang, rx, ry, rz)
        glMultMatrixf( self.__currentMatrix )
        self.__currentMatrix = glGetFloatv( GL_MODELVIEW_MATRIX )
        glPopMatrix( )

    def getCurrentMatrix( self ):
        return self.__currentMatrix

    def setCurrentMatrix(self, matrix ):
        self.__currentMatrix = matrix