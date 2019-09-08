import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform , get_sparse_transform , get_mask_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import os
import numpy as np
class LabeledDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_scribbles = os.path.join(opt.dataroot, 'scribbles')  #'pix2pix') #'scribbles' )  #'masks')
        self.dir_images = os.path.join(opt.dataroot, 'images') #os.path.join(opt.dataroot, 'images')

        self.classes = sorted(os.listdir(self.dir_images)) # sorted so that the same order in all cases; check if you've to change this with other models
        self.num_classes = len(self.classes)

        self.scribble_paths = []
        self.images_paths = []
        for cl in self.classes:
            self.scribble_paths.append(sorted( make_dataset( os.path.join( self.dir_scribbles , cl  )  )  ) )
            self.images_paths.append( sorted(  make_dataset( os.path.join( self.dir_images , cl  )  )  ) )

        self.cum_sizes = []
        self.sizes = []
        size =0
        for i in range(self.num_classes):
            size += len(self.scribble_paths[i])
            self.cum_sizes.append(size)
            self.sizes.append(size)

        self.transform = get_transform(opt)
        self.sparse_transform = get_sparse_transform(opt)
        self.mask_transform =  get_mask_transform(opt)
    def find_label(self,index):
        sub=0
        for i in range(self.num_classes):
            if index < self.cum_sizes[i]:
                return i,(index-sub)
            sub= self.cum_sizes[i]

    def __getitem__(self, index):
        index = index % self.cum_sizes[ self.num_classes -1  ]
        label , relative_index = self.find_label(index)
        if self.opt.sketchy_dataset:
            A_path = self.scribble_paths[label][ relative_index  ]
            B_path = A_path.replace('scribbles','images').split('-')[0]+'.jpg'

        elif self.opt.autocomplete_dataset_outline:
            A_path = self.scribble_paths[label][ relative_index  ]
            B_path = A_path.replace('scribbles','images').split('_')[0]+'.png'

        elif self.opt.autocomplete_dataset_edges:
            A_path = self.scribble_paths[label][ relative_index  ]
            B_path = A_path.replace('scribbles','images').split('_')[0]+'_AB.jpg'
        elif self.opt.edges_outlines_dataset:
            A_path = self.scribble_paths[label][ relative_index  ]
            B_path = A_path.replace('scribbles','images')
            if np.random.multinomial(1, [1.0 / 3, 2.0 / 3])[0]==1:
                A_path=A_path.replace('scribbles','edges_outlines')
        else :
            A_path = self.scribble_paths[label][ relative_index  ]
            B_path = self.images_paths[label][relative_index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')



        A = self.transform(A_img)
        B = self.transform(B_img)
        A_mask = self.mask_transform(A_img)
        A_sparse = self.sparse_transform(A_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if input_nc == 1:  # RGB to gray
            tmp = A_sparse[0, ...] * 0.299 + A_sparse[1, ...] * 0.587 + A_sparse[2, ...] * 0.114
            A_sparse = tmp.unsqueeze(0)



        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A,'A_sparse':A_sparse, 'A_mask':A_mask, 'B': B,
                'A_paths': A_path, 'B_paths': B_path, 'label': label }

    def __len__(self):
        return self.cum_sizes[ self.num_classes - 1  ]

    def get_transform(self):
        return self.transform

    def get_root(self):
        return self.root

    def get_classes(self):
        return self.classes

    def get_num_classes(self):
        return len(self.classes)

    def name(self):
        return 'LabeledDataset'
