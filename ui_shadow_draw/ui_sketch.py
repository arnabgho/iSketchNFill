import numpy as np
import cv2
import ui_shadow_draw.thinplate as tps
from ui_shadow_draw.img_utils import *

def warp_image_cv(img,c_src,c_dst,dshape=None):
    transformed_image = mls_rigid_deformation_inv(img,c_src,c_dst,alpha=1,density=1)
    transformed_image = 255 * transformed_image
    transformed_image = transformed_image.astype(img.dtype)
    return transformed_image

class UISketch:
    def __init__(self, img_size, scale, accu=True, nc=3,width=10):
        self.img_size = img_size
        self.scale = scale
        self.nc = nc
        self.img = np.zeros((img_size, img_size, self.nc), np.uint8)
        self.shadow_img = np.ones((img_size, img_size, self.nc), np.uint8)
        self.mask = np.zeros((img_size, img_size, 1), np.uint8)
        if self.nc == 1:  # [hack]
            self.width = width
        else:
            self.width = width
        self.img.fill(255)
        self.mask.fill(255)
        self.shadow_img.fill(255)
        self.background_img = self.img
        self.foreground_img = self.img

    def update(self, points,color):
        num_pnts = len(points)
        c = color #0#255

        for i in range(0, num_pnts - 1):
            pnt1 = (int(points[i].x()/self.scale), int(points[i].y()/self.scale))
            pnt2 = (int(points[i + 1].x()/self.scale), int(points[i + 1].y()/self.scale))
            if self.nc == 3:
                cv2.line(self.img, pnt1, pnt2, (c,c,c), self.width)
            else:
                cv2.line(self.img, pnt1, pnt2, c, self.width)

    def warp_img(self,pos,move_pos,warp_control_points):
        pos_x = int(pos.x()/self.scale)
        pos_y = int(pos.y()/self.scale)
        move_pos_x = int(move_pos.x()/self.scale)
        move_pos_y = int(move_pos.y()/self.scale)

        c_src = np.array([
            [pos_x, pos_y],
            ])

        c_dst = np.array([
            [move_pos_x, move_pos_y],
            ])

        for warp_control_point in warp_control_points:
            x = int(warp_control_point.x()/self.scale)
            y = int(warp_control_point.y()/self.scale)
            arr = np.array([[x,y]])
            c_src = np.vstack((c_src,arr))
            c_dst = np.vstack((c_dst,arr))
        self.img = warp_image_cv(self.img, c_src, c_dst, dshape=(self.img_size, self.img_size))

    def paste_patch(self,pos,w,paste_img):
        pos_x = int(pos.x()/self.scale)
        pos_y = int(pos.y()/self.scale)
        w = int(w)

        paste_img = cv2.resize(paste_img,(self.img_size,self.img_size))

        mask = np.full((self.img_size, self.img_size), 0 ,dtype = np.uint8)
        mask = cv2.rectangle(mask,(pos_x,pos_y),( pos_x + w , pos_y + w  ),(255,255,255),thickness=-1)

        img = self.img
        foreground_img = cv2.bitwise_or(paste_img,paste_img,mask=mask)
        back_mask = cv2.bitwise_not(mask)
        background_img =  cv2.bitwise_or(img, img, mask = back_mask)

        self.img = cv2.bitwise_or(foreground_img,background_img)



    def move_img(self,pos,move_pos,w):
        pos_x = int(pos.x()/self.scale)
        pos_y = int(pos.y()/self.scale)
        move_pos_x = int(move_pos.x()/self.scale)
        move_pos_y = int(move_pos.y()/self.scale)

        w = int(w)

        mask = np.full((self.img_size, self.img_size), 0 ,dtype = np.uint8)
        mask = cv2.rectangle(mask,(pos_x,pos_y),( pos_x + w , pos_y + w  ),(255,255,255),thickness=-1)

        img = self.img
        foreground_img =  cv2.bitwise_or(img,img,mask=mask)
        back_mask = cv2.bitwise_not(mask)
        background_img =  cv2.bitwise_or(img, img, mask = back_mask)

        white = np.zeros((self.img_size,self.img_size,3),np.uint8)
        white.fill(255)
        white_masked = cv2.bitwise_or(white,white,mask=mask)

        background_img = cv2.bitwise_or(background_img,white_masked)

        shift_x = move_pos_x-pos_x
        shift_y = move_pos_y-pos_y
        M = np.float32([[1,0,shift_x],[0,1,shift_y]])

        mask =  cv2.warpAffine(mask,M,(self.img_size,self.img_size))
        foreground_img =  cv2.warpAffine(foreground_img,M,(self.img_size,self.img_size))

        final_back_mask = cv2.bitwise_not(mask)
        background_img=cv2.bitwise_or(background_img,background_img,mask=final_back_mask)

        self.img =cv2.bitwise_or(foreground_img,background_img)


    def update_width(self, d):
        self.width = min(20, max(1, self.width+ d.y()))
        return self.width

    def update_brushwidth(self, width):
        self.width = width
        return self.width



    def get_constraints(self):
        return self.img, self.mask

    def get_img(self):
        final_image= cv2.addWeighted(self.img, 0.25, self.shadow_img, 0.75, 0)
        return final_image #self.img

    def get_draw_img(self):
        return self.img


    def set_shadow_img(self,cv2_img):
        if cv2_img is not None:
            self.shadow_img= cv2_img

    def show_img(self):
        cv2.imshow('ImageWindow', self.img)
        cv2.waitKey()

    def get_mask(self):
        return self.mask

    def reset(self):
        self.img = np.zeros((self.img_size, self.img_size, self.nc), np.uint8)
        self.mask = np.zeros((self.img_size, self.img_size, 1), np.uint8)
        self.shadow_img = np.ones((self.img_size, self.img_size, self.nc), np.uint8)
        self.img.fill(255)
        self.mask.fill(255)
        self.shadow_img.fill(255)
