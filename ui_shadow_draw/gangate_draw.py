from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

import numpy as np
import time
import cv2

from .ui_sketch import UISketch
from .ui_recorder import UIRecorder

class GANGATEDraw(QWidget):
    def __init__( self, win_size = 512 , img_size = 512,interactive=True ):
        QWidget.__init__(self)
        self.points = []
        self.moving_points = []
        self.nps = win_size
        self.scale = win_size /  float(img_size)
        self.brushWidth = int(2 * self.scale)
        self.cutWidth = 96
        self.warpWidth = int(4*self.scale)

        nc = 3
        self.uiSketch = UISketch(img_size=img_size,scale=self.scale,nc=nc,width=self.brushWidth)
        self.uir = UIRecorder()
        self.img_size = img_size
        self.interactive=interactive
        self.color = QColor(0,0,0)

        self.rgb_color = 0

        self.setMouseTracking(True)
        self.frame_id = -1
        self.image_id = 0
        self.shadow_image=None
        self.move(win_size,win_size)
        self.pos = QPoint(0,0)
        self.isPressed = False
        self.show_ui=False #True
        self.moving = False
        self.warping = False
        self.warp_start = None
        self.warp_end = None
        self.warp_control_points = []
        self.prev_brushWidth = None
        self.scribbling = True
        self.selecting_patch = False
        self.cycling_shadows=False
        self.show()

    def update_control_points(self,warp_start,warp_end):
        new_control_points = []
        for control_point in self.warp_control_points:
            if warp_start.x() == control_point.x() and warp_start.y() == control_point.y():
                pass
            else:
                new_control_points.append(control_point)
        new_control_points.append(warp_end)
        return new_control_points

    def remove_control_point(self,pos):
        new_control_points = []
        for control_point in self.warp_control_points:
            if pos.x() == control_point.x() and pos.y() == control_point.y():
                pass
            else:
                new_control_points.append(control_point)
        return new_control_points

    def update_ui(self):
        if self.moving:
            self.uiSketch.move_img(self.moving_points[-2],self.moving_points[-1],self.cutWidth)
        elif self.warping and self.warp_start is not None:
            self.uiSketch.warp_img(self.warp_start,self.warp_end,self.warp_control_points)
        else:
            self.uiSketch.update(self.points,self.rgb_color)
        self.update()

    def set_image_id(self, image_id):
        if self.image_id != image_id:
            self.image_id = image_id
            self.update()

    def set_frame_id(self, frame_id):
        if self.frame_id != frame_id:
            self.frame_id = frame_id
            self.update()


    def reset(self):
        self.isPressed = False
        self.points = []
        self.moving_points = []
        self.warp_control_points = []
        self.lastDraw = 0
        self.uir.reset()
        self.uiSketch.reset()
        self.frame_id = -1
        self.image_id = 0
        self.shadow_image=None
        self.warp_start = None
        self.warp_end = None
        self.brushWidth = int(1 * self.scale)
        self.scribbling = True
        self.scribble()
        self.selecting_patch=False
        self.cycling_shadows=False
        self.update()

    def erase(self):
        if self.scribbling:
            self.prev_brushWidth = self.brushWidth
            self.scribbling = False
        self.brushWidth = int(24 * self.scale)
        self.uiSketch.update_brushwidth(self.brushWidth)
        self.color = QColor(255,255,255)
        self.rgb_color = 255

    def scribble(self):
        if self.prev_brushWidth is None:
            self.brushWidth = int(2 * self.scale)
        else:
            self.brushWidth = self.prev_brushWidth
        self.scribbling = True
        self.uiSketch.update_brushwidth(self.brushWidth)
        self.color = QColor(0,0,0)
        self.rgb_color = 0
    def move_stroke(self):
        self.moving = True
        self.warping = False
        self.selecting_patch=False
        self.moving_points = []
        self.moving_points.append(self.pos)

    def warp_stroke(self):
        self.warping = True
        self.moving = False
        self.selecting_patch=False
        self.brushWidth = int(2 * self.scale)
        print("warping")


    def draw_stroke(self):
        self.warping = False
        self.moving = False
        self.selecting_patch=False
        self.scribbling=True
        self.warp_control_points = []


    def select_patch(self):
        self.warping = False
        self.moving = False
        self.scribbling=False
        self.selecting_patch=True


    def round_point(self, pnt):
        x = int(np.round(pnt.x()))
        y = int(np.round(pnt.y()))
        return QPoint(x, y)

    def get_image_id(self):
        return self.image_id

    def get_frame_id(self):
        return self.frame_id

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), Qt.white)
        painter.setRenderHint(QPainter.Antialiasing)
        self.uiSketch.set_shadow_img(self.shadow_image)
        im =self.uiSketch.get_img()

        if im is not None:
            bigim = cv2.resize(im, (self.nps, self.nps))
            qImg = QImage(bigim.tostring(), self.nps, self.nps, QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        # draw cursor
        if self.pos is not None:
            w = self.brushWidth
            c = self.color
            ca = QColor(255, 255, 255, 127)
            pnt = QPointF(self.pos.x(), self.pos.y())
            ca = QColor(0, 0, 0, 127)

            painter.setPen(QPen(ca, 1))
            painter.setBrush(ca)
            if self.moving or self.selecting_patch :
                painter.drawRect(int(self.pos.x()),int(self.pos.y()), self.cutWidth, self.cutWidth)
            elif self.warping and self.warp_start is not None:
                point_activated_color = QColor(0, 255, 0, 127)
                painter.setPen(QPen(point_activated_color, 1, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
                pnt1f = QPointF(self.warp_start)
                pnt2f = QPointF(self.pos.x(), self.pos.y())
                painter.drawEllipse(pnt1f, self.warpWidth, self.warpWidth)
                painter.drawEllipse(pnt2f, self.warpWidth, self.warpWidth)
                painter.drawLine(pnt1f, pnt2f)
            elif self.warping and self.warp_start is None:
                for point in self.warp_control_points:
                    painter.drawRect(point.x(),point.y(),self.warpWidth,self.warpWidth)
            else:
                painter.drawEllipse(pnt,w,w)


        if self.show_ui:
            self.uir.draw(painter)

    def shadow_image(self, img, pos):
        if img is None:
            return None
        weighted_img = np.ones((img.shape[0], img.shape[1]), np.uint8)
        x = int(pos.x() / self.scale)
        y = int(pos.y() / self.scale)

        weighted_img[y, x] = 0
        dist_img = cv2.distanceTransform(weighted_img, distanceType=cv2.cv.CV_DIST_L2, maskSize=5).astype(np.float32)
        dist_sigma = self.img_size/2.0
        dist_img_f = np.exp(-dist_img / dist_sigma)
        dist_img_f = np.tile(dist_img_f[..., np.newaxis], [1,1,3])
        l = 0.25
        img_f = img.astype(np.float32)
        rst_f = (img_f * l + (1-l) * (img_f * dist_img_f + (1-dist_img_f)*255.0))
        rst = rst_f.astype(np.uint8)
        return rst



    def wheelEvent(self, event):
        if not self.moving:
            d = event.angleDelta() / 120
            self.brushWidth = self.uiSketch.update_width(d)
        else:
            d= event.angleDelta()
            self.cutWidth = self.cutWidth + d.y()/10

        if self.selecting_patch:
            if self.cycling_shadows:
                self.parent().parent().cycle_shadow_image()
            else:
                d= event.angleDelta()
                self.cutWidth = self.cutWidth + d.y()/10

        if self.scribbling:
            self.prev_brushWidth = self.brushWidth

        #if self.cycling_shadows and self.selecting_patch:
        #    self.parent().parent().cycle_shadow_image()
        self.update()


    def findNearestControlPoint(self,pos,epsilon=10):
        for control_point in self.warp_control_points:
            d = abs(pos.x()-control_point.x()) + abs(pos.y()-control_point.y())
            if d < epsilon:
                return control_point

        return None

    def mousePressEvent(self, event):
        self.pos = self.round_point(event.pos())

        if event.button() == Qt.LeftButton:
            self.isPressed = True
            if self.moving:
                self.moving_points.append(self.pos)
            elif self.selecting_patch:
                paste_img=self.parent().parent().get_paste_image()
                self.uiSketch.paste_patch(self.pos,self.cutWidth,paste_img)
            elif self.warping:
                self.warp_start = self.findNearestControlPoint(self.pos)
            else:
                self.points.append(self.pos)
            self.update()

        if event.button() == Qt.RightButton and self.warping:
            nearest_point = self.findNearestControlPoint(self.pos,8)
            if nearest_point == None:
                self.warp_control_points.append(self.pos)
            else:
                self.warp_control_points = self.remove_control_point(nearest_point)
            self.update_ui()
            self.update()

        elif event.button() == Qt.RightButton:
            self.erase()
            self.isPressed = True
            self.points.append(self.pos)
            self.update_ui()
            self.update()





    def mouseMoveEvent(self, event):
        self.pos = self.round_point(event.pos())
        if self.isPressed:
            if self.moving:
                self.moving_points.append(self.pos)
            if self.warping:
                self.warp_end = self.pos
                if self.warp_start is not None:
                    self.update_ui()
            else:
                self.points.append(self.pos)
                self.update_ui()
            #self.update()
        if self.interactive:
            self.parent().parent().generate()
    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.LeftButton or event.button() == Qt.RightButton) and self.isPressed:
            self.uir.save_record(self.points, self.color, self.brushWidth)

            del self.points[:]
            self.isPressed = False
            del self.moving_points[:]
            self.lastDraw = 0
            self.scribble()
            if self.warping:
                self.warp_end = self.pos
                if self.warp_start is not None:
                    self.update_ui()
                    self.warp_control_points = self.update_control_points(self.warp_start,self.warp_end)
                    self.warp_start = None
        if self.interactive:
            self.parent().parent().generate()
    def getImage(self):
        return self.uiSketch.get_img()

    def getDrawImage(self):
        return self.uiSketch.get_draw_img()


    def setShadowImage(self,cv2_img):
        self.shadow_image= cv2.resize(cv2_img,(self.img_size,self.img_size))
        self.update()

    def cycleShadows(self):
        self.cycling_shadows=True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GANGATEDraw()
    sys.exit(app.exec_())
