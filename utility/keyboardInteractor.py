__author__ = 'christian'

from OpenGL.GLUT import glutTimerFunc

class KeyboardInteractor(object):

    def __init__(self, viewer, mouse_interactor):
        global v
        self.viewer = viewer
        v = viewer
        self.mouse_interactor = mouse_interactor
        self.last_x = 0.0
        self.last_y = 0.0


    def keyboardPressed(self, key, x, y):
        self.last_x = x
        self.last_y = y
        if key == b'z' and not self.mouse_interactor.zooming:
            self.mouse_interactor.zooming = True
        if key == b'a' and self.mouse_interactor.drawMeshes:
            self.mouse_interactor.drawMeshes = False
        if key == b'q':
            self.viewer.loadPrevModel()
        if key == b'e':
            self.viewer.loadNextModel()
        if key == b'p':
            KeyboardInteractor.playback(1)

    def keyboardUp(self, key, x, y):
        self.last_x = x
        self.last_y = y
        if key == b'z' and self.mouse_interactor.zooming:
            self.mouse_interactor.zooming = False
        if key == b'a' and not self.mouse_interactor.drawMeshes:
            self.mouse_interactor.drawMeshes = True

    @staticmethod
    def playback(value=1):
        global v
        v.loadNextModel()
        glutTimerFunc(100, KeyboardInteractor.playback, value + 1)