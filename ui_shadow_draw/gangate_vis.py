from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

import numpy as np
import time
import cv2

from .ui_sketch import UISketch
from .ui_recorder import UIRecorder


class GANGATEVis(QWidget):
    def __init__( self, win_size = 256 , img_size = 256, interactive = False, disable_browser=False ):
        QWidget.__init__(self)
        self.win_size = win_size
        self.img_size = img_size
        self.nps = win_size
        self.vis_results=None
        self.interactive = interactive
        self.show()
        self.setMouseTracking(True)
        self.pos = QPoint(0,0)
        self.disable_browser = disable_browser

    def paintEvent(self,event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(),Qt.white)

        if self.vis_results is not None:
            qImg = QImage(self.vis_results.tostring(), self.img_size, self.img_size, QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        painter.end()

    def update_vis(self,img_path):
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        destRGB = cv2.resize(destRGB, (self.img_size,self.img_size))
        self.vis_results=destRGB

        self.update()

    def update_vis_cv2(self,cv2_img):
        destRGB = cv2_img
        destRGB = cv2.resize(destRGB, (self.img_size,self.img_size))
        self.vis_results=destRGB

        self.update()



    def reset(self):
        self.vis_results = None
        self.update()

    def round_point(self, pnt):
        x = int(np.round(pnt.x()))
        y = int(np.round(pnt.y()))
        return QPoint(x, y)



    def mouseMoveEvent(self,event):
        self.pos = self.round_point(event.pos())
        if not self.disable_browser:
            self.parent().parent().browse(self.pos.x(),self.pos.y())
