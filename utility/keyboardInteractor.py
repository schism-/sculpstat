__author__ = 'christian'


class KeyboardInteractor(object):

    def __init__(self, viewer, mouse_interactor):
        self.viewer = viewer
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

    def keyboardUp(self, key, x, y):
        self.last_x = x
        self.last_y = y
        if key == b'z' and self.mouse_interactor.zooming:
            self.mouse_interactor.zooming = False
        if key == b'a' and not self.mouse_interactor.drawMeshes:
            self.mouse_interactor.drawMeshes = True