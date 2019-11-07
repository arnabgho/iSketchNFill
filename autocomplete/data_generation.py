import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Input Scribble Directory!')
parser.add_argument('--scribble_dir', type=str, help='Scribble Directory')
parser.add_argument('--how_many_per_size', type=int, default =1, help='Number of images per size')
args = parser.parse_args()

def fill(im,points):
    filler = cv2.convexHull(points)
    im=cv2.fillConvexPoly(im, filler, (255,255,255))
    return im

scribble_dir = args.scribble_dir #'../data/cartoonset10k/scribbles/'
occ_sizes=[64,128,192]
how_many_per_size = args.how_many_per_size
img_size = 256
for scribble_class in os.listdir(scribble_dir):
    out_dir = os.path.join(os.getcwd(),'autocomplete',scribble_class)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for scribble_img in tqdm(os.listdir(os.path.join(scribble_dir,scribble_class))):
        scribble_path = os.path.join(scribble_dir,scribble_class,scribble_img)
        if not (scribble_path.endswith('.png') or scribble_path.endswith('.jpg')):
            continue
        img = cv2.imread(scribble_path)
        img = cv2.resize(img,(img_size,img_size))
        rows,cols,ch = img.shape
        img_copy = img.copy()

        for occ_size in occ_sizes:
            max_start=rows-occ_size

    # start in range (0,191)
            for i in range(how_many_per_size):
                start= np.random.randint(max_start, size=2)
                points=np.array(((start[0],start[1]),(start[0]+occ_size,start[1]),(start[0],start[1]+occ_size),(start[0]+occ_size,start[1]+occ_size)))
                img=img_copy.copy()
                im = fill(img,points)
                scribble_img_id = scribble_img.split('.')[0]
                save_path=os.path.join(out_dir,scribble_img_id + '_' + str(occ_size) + '_' + str(i)+'.png')
                cv2.imwrite(save_path,im)
