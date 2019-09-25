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
import copy
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


opt2 = copy.deepcopy(opt)

opt2.name = opt.name_pix2pix
opt2.model = opt.model_pix2pix
opt2.ndisc_out_filters = 1
opt2.ndres_down = 4
opt2.ngres = 16
opt2.ndres = 16
opt2.ngf = 64
opt2.ndf = 64
opt2.spectral_G = True
opt2.spectral_D = True
opt2.norm_G = 'instance'
opt2.norm_D = 'instance'
opt2.checkpoints_dir = opt2.checkpoints_dir_pix2pix
opt2.res_op = 'add'
opt2.which_epoch = opt.which_epoch_pix2pix
opt2.shadow = True
opt2.nz = 8
opt2.input_nc = 3
opt2.loadSize = 256
opt2.fineSize = 256


transform_color = get_transform(opt2)
model_color = create_model(opt2)
class GANGATEGui(QWidget):
    def __init__(self,win_size= 256 ,img_size = 256):
        QWidget.__init__(self)

        self.win_size = win_size
        self.img_size = img_size

        self.drawWidget = GANGATEDraw(win_size=self.win_size,img_size=self.img_size)
        self.drawWidget.setFixedSize(win_size,win_size)


        self.visWidget_color = GANGATEVis(win_size=self.win_size,img_size=self.img_size,disable_browser=True)
        self.visWidget_color.setFixedSize(win_size,win_size)



        vbox = QVBoxLayout()

        self.drawWidgetBox = QGroupBox()
        self.drawWidgetBox.setTitle('Drawing Pad')
        vbox_t = QVBoxLayout()
        vbox_t.addWidget(self.drawWidget)
        self.drawWidgetBox.setLayout(vbox_t)
        vbox.addWidget(self.drawWidgetBox)

        self.labelId=6

        self.bBasketball = QRadioButton("Basketball")
        self.bBasketball.setToolTip("This button enables generation of a Basketball")

        self.bSoccer = QRadioButton("Soccer")
        self.bSoccer.setToolTip("This button enables generation of a Soccer")

        self.bWatermelon = QRadioButton("Watermelon")
        self.bWatermelon.setToolTip("This button enables generation of a Watermelon")

        self.bOrange = QRadioButton("Orange")
        self.bOrange.setToolTip("This button enables generation of a Orange")

        self.bCookie = QRadioButton("Cookie")
        self.bCookie.setToolTip("This button enables generation of a Cookie")

        self.bMoon = QRadioButton("Moon")
        self.bMoon.setToolTip("This button enables generation of a Moon")

        self.bStrawberry = QRadioButton("Strawberry")
        self.bStrawberry.setToolTip("This button enables generation of a Strawberry")

        self.bPineapple = QRadioButton("Pineapple")
        self.bPineapple.setToolTip("This button enables generation of a Pineapple")

        self.bCupcake = QRadioButton("Cupcake")
        self.bCupcake.setToolTip("This button enables generation of a Cupcake")

        self.bChicken = QRadioButton("Fried Chicken")
        self.bChicken.setToolTip("This button enables generation of a Chicken")

        bhbox = QGridLayout()  #QHBoxLayout()
        bGroup = QButtonGroup(self)

        bGroup.addButton(self.bBasketball)
        bGroup.addButton(self.bSoccer)
        bGroup.addButton(self.bWatermelon)
        bGroup.addButton(self.bOrange)
        bGroup.addButton(self.bCookie)
        bGroup.addButton(self.bMoon)
        bGroup.addButton(self.bStrawberry)
        bGroup.addButton(self.bPineapple)
        bGroup.addButton(self.bCupcake)
        bGroup.addButton(self.bChicken)

        bhbox.addWidget(self.bBasketball,0,0)
        bhbox.addWidget(self.bSoccer,1,0)
        bhbox.addWidget(self.bWatermelon,2,0)
        bhbox.addWidget(self.bOrange,3,0)
        bhbox.addWidget(self.bCookie,4,0)
        bhbox.addWidget(self.bMoon,0,1)
        bhbox.addWidget(self.bStrawberry,1,1)
        bhbox.addWidget(self.bPineapple,2,1)
        bhbox.addWidget(self.bCupcake,3,1)
        bhbox.addWidget(self.bChicken,4,1)


        self.bGenerate = QPushButton('Generate !')
        self.bGenerate.setToolTip("This button generates the final image to render")


        self.bReset = QPushButton('Reset !')
        self.bReset.setToolTip("This button resets the drawing pad !")



        self.bRandomize = QPushButton('Dice')
        self.bRandomize.setToolTip("This button generates new set of generations the drawing pad !")


        self.bMoveStroke = QRadioButton('Move Stroke')
        self.bMoveStroke.setToolTip("This button resets the drawing pad !")

        self.bWarpStroke = QRadioButton('Warp Stroke')
        self.bWarpStroke.setToolTip("This button resets the drawing pad !")

        self.bDrawStroke = QRadioButton('Draw Stroke')
        self.bDrawStroke.setToolTip("This button resets the drawing pad !")


        self.bEnableShadows = QCheckBox('Enable Shadows')
        self.bEnableShadows.toggle()



        hbox = QHBoxLayout()
        hbox.addLayout(vbox)


        vbox4 = QVBoxLayout()
        self.visWidgetBox = QGroupBox()
        self.visWidgetBox.setTitle('Generations')

        vbox_t4 = QVBoxLayout()
        vbox_t4.addWidget(self.visWidget_color)
        self.visWidgetBox.setLayout(vbox_t4)
        vbox4.addWidget(self.visWidgetBox)




        hbox.addLayout(vbox4)
        hbox.addLayout(bhbox)


        bhbox_controls = QGridLayout()
        bGroup_controls = QButtonGroup(self)

        bGroup_controls.addButton(self.bReset)
        bGroup_controls.addButton(self.bDrawStroke)
        bGroup_controls.addButton(self.bMoveStroke)
        bGroup_controls.addButton(self.bWarpStroke)


        bhbox_controls.addWidget(self.bReset,0,0)
        bhbox_controls.addWidget(self.bRandomize,0,1)
        bhbox_controls.addWidget(self.bDrawStroke,0,2)
        bhbox_controls.addWidget(self.bMoveStroke,0,3)
        bhbox_controls.addWidget(self.bWarpStroke,0,4)
        bhbox_controls.addWidget(self.bEnableShadows,0,5)

        hbox.addLayout(bhbox)

        controlBox = QGroupBox()
        controlBox.setTitle('Controls')

        controlBox.setLayout(bhbox_controls)

        vbox_final = QVBoxLayout()
        vbox_final.addLayout(hbox)
        vbox_final.addWidget(controlBox)
        self.setLayout(vbox_final)

        self.bPineapple.setChecked(True)

        self.bDrawStroke.setChecked(True)


        self.enable_shadow = True



        self.bBasketball.clicked.connect(self.Basketball)
        self.bSoccer.clicked.connect(self.Soccer)
        self.bWatermelon.clicked.connect(self.Watermelon)
        self.bOrange.clicked.connect(self.Orange)
        self.bCookie.clicked.connect(self.Cookie)
        self.bMoon.clicked.connect(self.Moon)
        self.bStrawberry.clicked.connect(self.Strawberry)
        self.bPineapple.clicked.connect(self.Pineapple)
        self.bCupcake.clicked.connect(self.Cupcake)
        self.bChicken.clicked.connect(self.Chicken)
        self.bGenerate.clicked.connect(self.generate)
        self.bReset.clicked.connect(self.reset)
        self.bRandomize.clicked.connect(self.randomize)
        self.bMoveStroke.clicked.connect(self.move_stroke)
        self.bWarpStroke.clicked.connect(self.warp_stroke)
        self.bDrawStroke.clicked.connect(self.draw_stroke)
        self.bEnableShadows.stateChanged.connect(self.toggle_shadow)

    def toggle_shadow(self,state):
        if state == Qt.Checked:
            self.enable_shadow=True
        else:
            self.enable_shadow=False
        self.generate()



    def Basketball(self):
        self.labelId = 0

    def Soccer(self):
        self.labelId = 7

    def Watermelon(self):
        self.labelId = 9

    def Orange(self):
        self.labelId = 5

    def Cookie(self):
        self.labelId = 2

    def Moon(self):
        self.labelId = 4

    def Strawberry(self):
        self.labelId = 8

    def Pineapple(self):
        self.labelId = 6

    def Cupcake(self):
        self.labelId = 3

    def Chicken(self):
        self.labelId = 1

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

    def get_network_input_color(self):
        pil_scribble = Image.open('imgs/test_0_L_fake_B_inter.png').convert('RGB')
        pil_scribble = pil_scribble.resize((256,256),Image.ANTIALIAS)
        A = transform_color(pil_scribble)
        A=A.resize_(1,3,256,256)
        B = A
        label = torch.LongTensor([[self.labelId]])
        data = {'A': A,'A_sparse':A,'A_mask':A,
                'B': B,'A_paths': '', 'B_paths': '', 'label': label }
        return data



    def generate(self):
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
            cv2_img = cv2.imread('imgs/test_0_L_fake_B_inter.png')

        cv2_img = cv2.resize(cv2_img,(self.img_size,self.img_size))
        self.drawWidget.setShadowImage(cv2_img)


        data = self.get_network_input_color()
        model_color.set_input(data)
        model_color.test()
        visuals = model_color.get_current_visuals()

        image_dir = './imgs'

        for label,image_numpy in visuals.items():
            image_name = 'test_color_%s.png' % (label)
            save_path = os.path.join(image_dir,image_name)
            util.save_image(image_numpy,save_path)

        self.visWidget_color.update_vis('imgs/test_color_fake_B.png')


    def reset(self):
        self.drawWidget.reset()

    def move_stroke(self):
        self.drawWidget.move_stroke()

    def warp_stroke(self):
        self.drawWidget.warp_stroke()

    def draw_stroke(self):
        self.drawWidget.draw_stroke()

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
