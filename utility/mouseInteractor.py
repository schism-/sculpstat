__author__ = 'christian'

import threading
import numpy
from OpenGL.GLUT import *
from utility.intmatrix import *

thread = None
buttonThread = None

class MouseInteractor(object):

    def __init__( self, translationScale=0.1, rotationScale=.2, interface=[]):
        self.scalingFactorRotation = rotationScale
        self.scalingFactorTranslation = translationScale
        self.rotationMatrix = InteractionMatrix( )
        self.translationMatrix = InteractionMatrix( )
        self.mouseButtonPressed = None
        self.oldMousePos = [ 0, 0 ]
        
        self.interface = interface
        self.zoomSlider = False
        self.zooming = False
        self.drawMeshes = True

    def mouseButton( self, button, mode, x, y ):

        if mode == GLUT_DOWN:
            guiPressed, bPressed = self.checkGUI(x, y, True)
            if (not guiPressed):
                self.mouseButtonPressed = button
                print("button pressed is: " + str(button))
                self.zoomSlider = False
            else:
                #Some button of the GUI was pressed (saved in bPressed)
                #Don't register the button click as a simple mouse action (to disable model re-rendering)
                self.mouseButtonPressed = None
                
                if (bPressed.enableCallback and bPressed.disabled == False):
                    print('button pressed')
                    bPressed.disabled = True
                    
                    if (bPressed.type == "MultiButton"):
                        bPressed.newConf(x,y)
                        
                    global thread, buttonThread
                    thread = threading.Thread(target=bPressed.callback)
                    thread.start()
                    
                    #Saving in a global variable, so other methods know what button to enable when the thread is done
                    buttonThread = bPressed

                if (bPressed.type == "Slider"):
                    self.zoomSlider = True
        else:
            self.mouseButtonPressed = None
            self.zoomSlider = False
    
        self.oldMousePos[0], self.oldMousePos[1] = x, y
        '''
        self.rotationMatrix.setCurrentMatrix(numpy.array([[ 0.9136188 , -0.02043674,  0.40605885,  0.        ],
                                                           [-0.03868171,  0.98983645,  0.13685055,  0.        ],
                                                           [-0.40472841, -0.14073625,  0.90354288,  0.        ],
                                                           [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=numpy.float32))
        self.translationMatrix.setCurrentMatrix(numpy.array([[ 1.,  0.,  0.,  0.],
                                                               [ 0.,  1.,  0.,  0.],
                                                               [ 0.,  0.,  1.,  0.],
                                                               [ 0.,  0.,  0.,  1.]], dtype=numpy.float32))
        '''
        if (not self.checkGUI(x, y, True)[0]):
            glutPostRedisplay( )

    def mouseMotion( self, x, y ):
        deltaX = x - self.oldMousePos[ 0 ]
        deltaY = y - self.oldMousePos[ 1 ]
        if self.mouseButtonPressed == GLUT_RIGHT_BUTTON:
            tX = deltaX * self.scalingFactorTranslation
            tY = deltaY * self.scalingFactorTranslation
            self.translationMatrix.addTranslation(tX, -tY, 0)
        elif self.mouseButtonPressed == GLUT_LEFT_BUTTON and not self.zooming:
            rY = deltaX * self.scalingFactorRotation
            self.rotationMatrix.addRotation(rY, 0, 1, 0)
            rX = deltaY * self.scalingFactorRotation
            self.rotationMatrix.addRotation(rX, 1, 0, 0)
            # print("self.rotationMatrix = " + repr(self.rotationMatrix.getCurrentMatrix()))
            # print("self.translationMatrix = " + repr(self.translationMatrix.getCurrentMatrix()))
        elif self.mouseButtonPressed == GLUT_LEFT_BUTTON and self.zooming:
            if (not self.checkGUI(x, y, True)[0]):
                tZ = deltaY * self.scalingFactorTranslation
                self.translationMatrix.addTranslation(0, 0, tZ)
                
        self.oldMousePos[0], self.oldMousePos[1] = x, y
        glutPostRedisplay( )

    def mousePassiveMotion(self, x, y):
        for b in self.interface:
            if (b.enableHighlighting):
                if (b.checkHit(x, y, False)):
                    b.highlighted = True
                else:
                    b.highlighted = False
            else:
                b.mouseUp()
                    
    def applyTransformation( self ):
        global thread, buttonThread
        if(not(thread == None)):
            if (not thread.is_alive()):
                buttonThread.disabled = False
                thread = None
                  
        if self.zoomSlider:
            sl = None
            for b in self.interface:
                if b.type == "Slider":
                    sl = b
                    break
            if (sl.checkHit(self.oldMousePos[0], self.oldMousePos[1], True)):
                tZ = sl.offset * 0.001
                self.translationMatrix.addTranslation(0, 0, tZ)          
        else:
            for b in self.interface:
                b.mouseUp()

        glMultMatrixf( self.translationMatrix.getCurrentMatrix() )
        glMultMatrixf( self.rotationMatrix.getCurrentMatrix() )

    def registerCallbacks( self ):
        """Initialise glut callback functions."""
        glutMouseFunc( self.mouseButton )
        glutMotionFunc( self.mouseMotion )
        glutPassiveMotionFunc( self.mousePassiveMotion )

    def checkGUI(self, x, y, click):
        somethingPressed = False
        buttonPressed = None
        for b in self.interface:
            if (b.checkHit(x, y, click)):
                somethingPressed = True
                buttonPressed = b
        
        return (somethingPressed, buttonPressed)