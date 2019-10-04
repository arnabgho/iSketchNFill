from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import visdom
from .common_net import *
from torch.nn import functional as F
import torch.utils.data.distributed
from torch.nn.utils import spectral_norm


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[] , opt={} ):
    netG = None
    use_gpu = len(gpu_ids) > 0
    #norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'GAN_stability_Generator':
        netG = GAN_stability_Generator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[],opt={}):
    netD = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'GAN_stability_Discriminator':
        netD = GAN_stability_Discriminator(opt)
        print(netD)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    return netD




# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm(planes,norm_type='batch',num_groups=4):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d(planes, affine=True)
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d(planes, affine=False)
    elif norm_type == 'group':
        norm_layer = nn.GroupNorm(num_groups,planes)
    elif norm_type == 'adain':
        norm_layer = AdaptiveInstanceNorm2d(planes)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class GatedResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(GatedResnetBlock,self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        norm_layer='instance'
        # Submodules
        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias))

        if self.learned_shortcut:
            self.conv_s = spectral_norm( nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False))



    def forward(self, x,alpha=1.0,beta=0.0):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        #dx = self.norm(dx)
        if type(alpha)!=float:
            alpha=alpha.expand_as(x_s)
        if type(beta)!=float:
            beta=beta.expand_as(x_s)
        out = x_s + alpha*dx + beta   #x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s




class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


########################################################
########################################################
### Networks from original GAN_stability paper
########################################################

class GAN_stability_Generator(nn.Module):
    def __init__(self, opt , embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.opt = opt
        size = opt.fineSize
        nlabels = opt.n_classes
        s0 = self.s0 = size // 32
        nf = self.nf = opt.ngf
        self.z_dim = z_dim = opt.nz
        nc = opt.input_nc
        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)

        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)

        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, opt.output_nc, 3, padding=1)

        sparse_processor_blocks = []

        # 8x8
        sparse_processor_blocks += [GatedResnetBlock(nc,16*nf)]
        # 16x16
        sparse_processor_blocks += [GatedResnetBlock(nc,16*nf)]
        # 32x32
        sparse_processor_blocks += [GatedResnetBlock(nc,8*nf)]
        # 64x64
        sparse_processor_blocks += [GatedResnetBlock(nc,4*nf)]
        # 128x128
        sparse_processor_blocks += [GatedResnetBlock(nc,2*nf)]

        self.num_sparse_blocks = len(sparse_processor_blocks)

        self.sparse_processor = nn.Sequential(*sparse_processor_blocks)


    def forward(self, sparse_input , y, z):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        scale_factor = 1.0/32.0

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[0](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[1](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[2](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[3](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[4](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        if self.opt.no_sparse_add:
            out = self.conv_img(actvn(out))
        else:
            out = sparse_input + self.conv_img(actvn(out))
        out = F.tanh(out)

        return out


class GAN_stability_Discriminator(nn.Module):
    def __init__(self,opt, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.opt = opt
        size = opt.fineSize
        self.embed_size = embed_size
        nlabels = opt.n_classes
        s0 = self.s0 = size // 32
        nf = self.nf = opt.ndf
        ny = nlabels
        if opt.img_conditional_D:
            nc = opt.input_nc + opt.output_nc
        else:
            nc = opt.output_nc


        # Submodules
        self.conv_img = nn.Conv2d(nc, 1*nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_0_1 = ResnetBlock(1*nf, 2*nf)

        self.resnet_1_0 = ResnetBlock(2*nf, 2*nf)
        self.resnet_1_1 = ResnetBlock(2*nf, 4*nf)

        self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_2_1 = ResnetBlock(4*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 8*nf)
        self.resnet_3_1 = ResnetBlock(8*nf, 16*nf)

        self.resnet_4_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_4_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_5_1 = ResnetBlock(16*nf, 16*nf)

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)

        sparse_processor_blocks = []
        # 128x128
        sparse_processor_blocks += [GatedResnetBlock(nc,2*nf)]
        # 64x64
        sparse_processor_blocks += [GatedResnetBlock(nc,4*nf)]
        # 32x32
        sparse_processor_blocks += [GatedResnetBlock(nc,8*nf)]
        # 16x16
        sparse_processor_blocks += [GatedResnetBlock(nc,16*nf)]
        # 8x8
        sparse_processor_blocks += [GatedResnetBlock(nc,16*nf)]

        self.num_sparse_blocks = len(sparse_processor_blocks)

        self.sparse_processor = nn.Sequential(*sparse_processor_blocks)


    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)

        scale_factor = 1.0/2.0

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        sparse = F.interpolate(x,scale_factor=scale_factor)
        sparse = self.sparse_processor[0](sparse)
        out += sparse
        scale_factor /= 2.0
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        sparse = F.interpolate(x,scale_factor=scale_factor)
        sparse = self.sparse_processor[1](sparse)
        out += sparse
        scale_factor /= 2.0
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        sparse = F.interpolate(x,scale_factor=scale_factor)
        sparse = self.sparse_processor[2](sparse)
        out += sparse
        scale_factor /= 2.0
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        sparse = F.interpolate(x,scale_factor=scale_factor)
        sparse = self.sparse_processor[3](sparse)
        out += sparse
        scale_factor /= 2.0
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        sparse = F.interpolate(x,scale_factor=scale_factor)
        sparse = self.sparse_processor[4](sparse)
        out += sparse
        scale_factor /= 2.0
        out = self.resnet_5_1(out)

        out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out



class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True,use_sn=False):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        if use_sn:
            # Submodules
            self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1))
            self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias))
            if self.learned_shortcut:
                self.conv_s = spectral_norm(nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False))
        else:
            # Submodules
            self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
            self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
            if self.learned_shortcut:
                self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out




