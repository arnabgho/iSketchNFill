import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageOps
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    if opt.model == "label_pix2pix" or opt.dataset_mode=='labeled' :
        transform_list = []
        osize = [opt.fineSize, opt.fineSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))

        transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]


    return transforms.Compose(transform_list)

def get_sparse_transform(opt):
    transform_list=[]
    if opt.model == "label_pix2pix" or opt.dataset_mode=='labeled' :
        transform_list = []
        osize = [opt.sparseSize, opt.sparseSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))

        transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]


    return transforms.Compose(transform_list)

def get_mask_transform(opt):
    transform_list = []
    osize = [opt.fineSize, opt.fineSize]
    transform_list.append(transforms.Scale(osize, Image.BICUBIC))

    transform_list += [transforms.Lambda(lambda img: __binarize(img)),
                                transforms.ToTensor()]
    return transforms.Compose(transform_list)

def __binarize(img):
    img = ImageOps.invert(img)
    img = img.convert('1')
    #img_np = np.array(img)
    ##img_np_inverted = np.invert(img_np)
    ##final_img=Image.fromarray(img_np_inverted)
    #final_img=Image.fromarray(img_np)
    return img



def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
