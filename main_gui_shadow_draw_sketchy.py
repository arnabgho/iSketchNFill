from PyQt5.QtWidgets import * #QWidget, QApplication
from PyQt5.QtGui import * #QPainter, QPainterPath
from PyQt5.QtCore import * #Qt
import sys
from PIL import Image
import numpy as np
import time
import cv2
import torch
from ui_shadow_draw.ui_sketch import UISketch
from ui_shadow_draw.ui_recorder import UIRecorder
import qdarkstyle
from ui_shadow_draw.gangate_draw import GANGATEDraw
from ui_shadow_draw.gangate_vis import GANGATEVis
from data.base_dataset import get_transform


import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util import util

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.loadSize=256

transform = get_transform(opt)
model = create_model(opt)

class GANGATEGui(QWidget):
    def __init__(self,win_size= 384 ,img_size = 384):
        QWidget.__init__(self)

        self.win_size = win_size
        self.img_size = img_size

        self.drawWidget = GANGATEDraw(win_size=self.win_size,img_size=self.img_size)
        self.drawWidget.setFixedSize(win_size,win_size)

        self.visWidget = GANGATEVis(win_size=self.win_size,img_size=self.img_size)
        self.visWidget.setFixedSize(win_size,win_size)

        vbox = QVBoxLayout()

        self.drawWidgetBox = QGroupBox()
        self.drawWidgetBox.setTitle('Drawing Pad')
        vbox_t = QVBoxLayout()
        vbox_t.addWidget(self.drawWidget)
        self.drawWidgetBox.setLayout(vbox_t)
        vbox.addWidget(self.drawWidgetBox)

        self.labelId=0

        self.bBicycle = QRadioButton("Bicycle")
        self.bBicycle.setToolTip("This button enables generation of a Bicycle")

        self.bCat = QRadioButton("Cat")
        self.bCat.setToolTip("This button enables generation of a Cat")

        self.bChair = QRadioButton("Chair")
        self.bChair.setToolTip("This button enables generation of a Chair")

        self.bHamburger = QRadioButton("Hamburger")
        self.bHamburger.setToolTip("This button enables generation of a Hamburger")

        self.bPizza = QRadioButton("Pizza")
        self.bPizza.setToolTip("This button enables generation of a Pizza")

        self.bTeddy = QRadioButton("Teddy")
        self.bTeddy.setToolTip("This button enables generation of a Teddy")


        bhbox = QGridLayout()

        bhbox.addWidget(self.bBicycle,0,0)
        bhbox.addWidget(self.bCat,1,0)
        bhbox.addWidget(self.bChair,2,0)
        bhbox.addWidget(self.bHamburger,0,1)
        bhbox.addWidget(self.bPizza,1,1)
        bhbox.addWidget(self.bTeddy,2,1)



        self.bGenerate = QPushButton('Generate !')
        self.bGenerate.setToolTip("This button generates the final image to render")


        self.bReset = QPushButton('Reset !')
        self.bReset.setToolTip("This button resets the drawing pad !")



        self.bRandomize = QPushButton('Dice')
        self.bRandomize.setToolTip("This button generates new set of generations the drawing pad !")


        self.bMoveStroke = QRadioButton('Move Stroke')
        self.bMoveStroke.setToolTip("This button moves the selected stroke on the drawing pad !")

        self.bWarpStroke = QRadioButton('Warp Stroke')
        self.bWarpStroke.setToolTip("This button warps the selected stroke on the drawing pad !")

        self.bDrawStroke = QRadioButton('Draw Stroke')
        self.bDrawStroke.setToolTip("This button reverts back to the drawing mode on the drawing pad !")

        self.bSelectPatch = QRadioButton('Select Patch')
        self.bSelectPatch.setToolTip("This button selects patches from the shadows!")

        self.bEnableShadows = QCheckBox('Enable Shadows')
        self.bEnableShadows.toggle()

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)

        vbox3 = QVBoxLayout()
        self.visWidgetBox = QGroupBox()
        self.visWidgetBox.setTitle('Generations')

        vbox_t3 = QVBoxLayout()
        vbox_t3.addWidget(self.visWidget)
        self.visWidgetBox.setLayout(vbox_t3)
        vbox3.addWidget(self.visWidgetBox)



        bhbox_controls = QGridLayout()

        bhbox_controls.addWidget(self.bReset,0,0)
        bhbox_controls.addWidget(self.bRandomize,0,1)
        bhbox_controls.addWidget(self.bDrawStroke,0,2)
        bhbox_controls.addWidget(self.bMoveStroke,0,3)
        bhbox_controls.addWidget(self.bWarpStroke,0,4)
        bhbox_controls.addWidget(self.bSelectPatch,0,5)
        bhbox_controls.addWidget(self.bEnableShadows,0,6)


        hbox.addLayout(vbox3)
        hbox.addLayout(bhbox)

        controlBox = QGroupBox()
        controlBox.setTitle('Controls')

        controlBox.setLayout(bhbox_controls)

        vbox_final = QVBoxLayout()
        vbox_final.addLayout(hbox)
        vbox_final.addWidget(controlBox)
        self.setLayout(vbox_final)

        self.bTeddy.setChecked(True)
        self.labelId=5
        self.bDrawStroke.setChecked(True)

        self.enable_shadow = True
        self.which_shadow_img = 0

        self.bBicycle.clicked.connect(self.Bicycle)
        self.bCat.clicked.connect(self.Cat)
        self.bChair.clicked.connect(self.Chair)
        self.bHamburger.clicked.connect(self.Hamburger)
        self.bPizza.clicked.connect(self.Pizza)
        self.bTeddy.clicked.connect(self.Teddy)

        self.bGenerate.clicked.connect(self.generate)
        self.bReset.clicked.connect(self.reset)
        self.bRandomize.clicked.connect(self.randomize)
        self.bMoveStroke.clicked.connect(self.move_stroke)
        self.bWarpStroke.clicked.connect(self.warp_stroke)
        self.bDrawStroke.clicked.connect(self.draw_stroke)
        self.bSelectPatch.clicked.connect(self.select_patch)
        self.bEnableShadows.stateChanged.connect(self.toggle_shadow)


    def toggle_shadow(self,state):
        if state == Qt.Checked:
            self.enable_shadow=True
        else:
            self.enable_shadow=False
            self.drawWidget.cycleShadows()
        self.generate()


    def Bicycle(self):
        self.labelId = 0

    def Cat(self):
        self.labelId = 1

    def Chair(self):
        self.labelId = 2

    def Hamburger(self):
        self.labelId = 3

    def Pizza(self):
        self.labelId = 4

    def Teddy(self):
        self.labelId = 5

    def get_network_input(self):
        cv2_scribble = self.drawWidget.getDrawImage()
        cv2_scribble = cv2.cvtColor(cv2_scribble,cv2.COLOR_BGR2RGB)
        cv2.imwrite('./imgs/current_scribble.jpg',cv2_scribble)
        pil_scribble = Image.fromarray(cv2_scribble)

        A = transform(pil_scribble)
        A=A.resize_(1,opt.input_nc,128,128)
        A=A.expand(opt.num_interpolate,opt.input_nc,128,128)
        B = A
        label = torch.LongTensor([self.labelId])
        label = label.expand(opt.num_interpolate)
        data = {'A': A,'A_sparse':A,'A_mask':A,
                'B': B,'A_paths': '', 'B_paths': '', 'label': label }
        return data

    def browse(self,pos_y,pos_x):
        num_rows = int(opt.num_interpolate/2)
        num_cols = 2
        div_rows = int(self.img_size/num_rows)
        div_cols = int(self.img_size/num_cols)

        which_row = int(pos_x / div_rows)
        which_col = int(pos_y / div_cols)


        cv2_gallery = cv2.imread('imgs/fake_B_gallery.png')
        cv2_gallery = cv2.resize(cv2_gallery,(self.img_size,self.img_size))

        cv2_gallery = cv2.rectangle(cv2_gallery, ( which_col * div_cols , which_row * div_rows  ) , ( (which_col + 1) * div_cols , (which_row + 1) * div_rows  ) , (0,255,0) , 8)
        self.visWidget.update_vis_cv2(cv2_gallery)

        cv2_img = cv2.imread('imgs/test_fake_B_shadow.png')
        cv2_img = cv2.resize(cv2_img,(self.img_size,self.img_size))

        which_highlight = which_row * 2 + which_col
        img_gray = cv2.imread('imgs/test_%d_L_fake_B_inter.png' % (which_highlight),cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.resize(img_gray,(self.img_size,self.img_size))
        (thresh,im_bw)=cv2.threshold(img_gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2_img[np.where(im_bw==[0])] = [0,255,0]


        self.drawWidget.setShadowImage(cv2_img)

    def generate(self):
        #pass
        data=self.get_network_input()
        model.set_input(data)
        visuals=model.get_latent_noise_visualization()

        image_dir='./imgs'
        for label,image_numpy in visuals.items():
            image_name='test_%s.png' % (label)
            save_path=os.path.join(image_dir,image_name)
            util.save_image(image_numpy,save_path)
        ## convert back from pil image to cv2 image


        if self.enable_shadow:
            cv2_img = cv2.imread('imgs/test_fake_B_shadow.png')
        else:
            cv2_img = cv2.imread('imgs/test_%d_L_fake_B_inter.png'%(self.which_shadow_img))

        self.drawWidget.setShadowImage(cv2_img)

        self.visWidget.update_vis('imgs/fake_B_gallery.png')

    def cycle_shadow_image(self):
        self.which_shadow_img+=1
        self.which_shadow_img = self.which_shadow_img%opt.num_interpolate
        self.generate()
    def get_paste_image(self):
        cv2_img = cv2.imread('imgs/test_%d_L_fake_B_inter.png'%(self.which_shadow_img))
        cv2_img = cv2.resize(cv2_img,(self.img_size,self.img_size))

        return cv2_img

    def reset(self):
        self.drawWidget.reset()

    def move_stroke(self):
        self.drawWidget.move_stroke()

    def warp_stroke(self):
        self.drawWidget.warp_stroke()

    def draw_stroke(self):
        self.drawWidget.draw_stroke()

    def select_patch(self):
        self.drawWidget.select_patch()

    def randomize(self):
        model.randomize_noise()
        self.generate()
    def scribble(self):
        self.drawWidget.scribble()

    def erase(self):
        self.drawWidget.erase()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GANGATEGui()
    window.setWindowTitle('iSketchNFill')
    window.setWindowFlags(window.windowFlags() & ~Qt.WindowMaximizeButtonHint)   # fix window siz
    window.show()
    sys.exit(app.exec_())
