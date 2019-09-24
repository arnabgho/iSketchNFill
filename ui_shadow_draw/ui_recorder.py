import copy
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2

class UIRecorder:
    def __init__(self):
        self.strokes = []
        self.colors = []
        self.widths = []
        self.patches = []

    def save_record(self, stroke, color, width, patch=None):
        self.strokes.append(copy.deepcopy(stroke))
        self.colors.append(color)
        self.widths.append(width)
        self.patches.append(patch)

    def draw(self, painter):
        for points, color, width, patch in zip(self.strokes, self.colors, self.widths, self.patches):
            painter.setPen(QPen(color, width, cap=Qt.RoundCap, join=Qt.RoundJoin))

            npnts = len(points)
            for i in range(0, npnts - 5, 5):
                painter.drawLine(points[i], points[i + 5])
            #if t is 'patch':
            #    w=patch.shape[0]
            #    h=patch.shape[1]
            #    qImg = QImage(patch.tostring(), w, h, QImage.Format_RGB888)
            #    painter.drawImage(points.x()-w/2, points.y()-h/2, qImg)


    def reset(self):
        del self.strokes[:]
        del self.colors[:]
        del self.widths[:]
        del self.patches[:]
