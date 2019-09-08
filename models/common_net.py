import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
##########################################################
# Adaptive Instance Normalization
##########################################################

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

##############################################################


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class BATCHResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(BATCHResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.BatchNorm1d(num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.BatchNorm1d(num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out


class INSResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(INSResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.InstanceNorm1d(num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.InstanceNorm1d(num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out

class ResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(ResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        #model += [nn.InstanceNorm1d(num_neurons)] # Just testing might be removed
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        #model += [nn.InstanceNorm1d(num_neurons)] # Just testing might be removed
        model += [nn.ReLU()]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out

class ResBlock1D(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(ResBlock1D, self).__init__()

        model = []
        model += [ nn.Conv1d( num_neurons , num_neurons  , kernel_size=3, stride=1,padding=1)  ]
        model += [nn.BatchNorm1d(num_neurons)] # Just testing might be removed
        model += [nn.ReLU(inplace=True)]
        model += [ nn.Conv1d( num_neurons , num_neurons  , kernel_size=3, stride=1,padding=1)  ]
        model += [nn.BatchNorm1d(num_neurons)] # Just testing might be removed
        model += [nn.ReLU()]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out


class GatedResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(GatedResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x,alpha):
        residual=x
        out=alpha*self.model(x)
        out+=residual
        return out

#class UpGatedConvResBlock(nn.Module):
#  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
#    if use_sn:
#        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
#    else:
#        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)
#
#
#  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False):
#    super(UpGatedConvResBlock, self).__init__()
#    self.upsample = nn.Sequential(*upsampleLayer(inplanes , planes , upsample='nearest' , use_sn=use_sn))
#    model = []
#    #model += upsampleLayer(inplanes , planes , upsample='subpixel' , use_sn=use_sn)
#    #model += [nn.Upsample( scale_factor=2, mode='nearest')]#mode='bilinear',align_corners=True)]
#    #model += [nn.Upsample( scale_factor=2, mode='bilinear',align_corners=True)]
#    #model += [self.conv3x3(inplanes, planes, stride,use_sn)]
#    model += [nn.BatchNorm2d(planes)]
#    model += [nn.ReLU(inplace=True)]
#    model += [self.conv3x3(planes, planes,stride,use_sn)]
#    model += [nn.BatchNorm2d(planes)]
#    model += [nn.ReLU(inplace=True)]
#    if dropout > 0:
#      model += [nn.Dropout(p=dropout)]
#    self.model = nn.Sequential(*model)
#
#    residual_block = []
#    #residual_block += upsampleLayer(inplanes , planes , upsample='nearest' , use_sn=use_sn)
#    #residual_block += [nn.Upsample(scale_factor=2, mode='nearest')]  #mode='nearest',align_corners=True)]
#    #residual_block += [nn.Upsample( scale_factor=2, mode='bilinear' ,align_corners=True)]
#    #residual_block += [self.conv3x3(inplanes,planes,stride,use_sn)]
#    #self.model.apply(gaussian_weights_init)
#    self.residual_block=nn.Sequential(*residual_block)
#
#  def forward(self, x, alpha):
#    x = self.upsample(x)
#    residual = self.residual_block(x)
#    out = alpha * self.model(x)
#    out += residual
#    return out



def upsampleLayer(inplanes, outplanes, upsample='basic', use_sn=True):
    # padding_type = 'zero'
    if upsample == 'basic' and not use_sn:
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=3, stride=2,padding=1, output_padding=1)]
    elif upsample == 'bilinear' and not use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif upsample == 'nearest' and not use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif upsample == 'subpixel' and not use_sn:
        upconv = [ nn.Conv2d(inplanes,outplanes*4,kernel_size=3 , stride=1 , padding=1),
                   nn.PixelShuffle(2)]
    elif upsample == 'basic' and use_sn :
        upconv = [spectral_norm(nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=3, stride=2,padding=1, output_padding=1))]
    elif upsample == 'bilinear' and use_sn :
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif upsample == 'nearest' and use_sn :
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif upsample == 'subpixel' and use_sn:
        upconv = [ spectral_norm(nn.Conv2d(inplanes,outplanes*4,kernel_size=3 , stride=1 , padding=1)),
                   nn.PixelShuffle(2)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv

def downsampleLayer(inplanes, outplanes, downsample='basic', use_sn=True):
    # padding_type = 'zero'
    if downsample == 'basic' and not use_sn:
        downconv = [nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1)]
    elif downsample == 'avgpool' and not use_sn:
        downconv = [nn.AvgPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif downsample == 'maxpool' and not use_sn:
        downconv = [nn.MaxPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]

    elif downsample == 'basic' and use_sn :
        downconv = [spectral_norm(nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1))]
    elif downsample == 'avgpool' and use_sn :
        downconv = [nn.AvgPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif downsample == 'maxpool' and use_sn :
        downconv = [nn.MaxPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]

    else:
        raise NotImplementedError(
            'downsample layer [%s] not implemented' % downsample)
    return downconv

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

##############################################################
## Simple Gated Operations (Affine) and (Multiplicative)
##############################################################

def residual_op(out,residual,op='max'):
    if op == 'add':
        return out + residual
    elif op=='sub':
        return out - residual
    elif op == 'max':
        return torch.max(out,residual)
    elif op == 'min':
        return torch.min(out,residual)

class UpGatedConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8,res_op='add'):
    super(UpGatedConvResBlock, self).__init__()
    model = []
    model += upsampleLayer(inplanes , planes , upsample='nearest' , use_sn=use_sn)
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)] #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block +=  upsampleLayer(inplanes , planes , upsample='bilinear' , use_sn=use_sn)
    self.residual_block=nn.Sequential(*residual_block)
    self.res_op = res_op
  def forward(self, x, alpha=1.0,beta=0.0):
    residual = self.residual_block(x)
    f_x=self.model(x)
    if type(alpha)!=float:
        alpha=alpha.expand_as(f_x)
    if type(beta)!=float:
        beta=beta.expand_as(f_x)
    out = alpha * f_x + beta
    out = residual_op(out,residual,self.res_op)  #out += residual
    return out

class DownGatedConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8,res_op='add'):
    super(DownGatedConvResBlock, self).__init__()
    model = []
    model += downsampleLayer(inplanes,planes,downsample='avgpool',use_sn=use_sn)
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block += downsampleLayer(inplanes,planes,downsample='avgpool',use_sn=use_sn)
    self.residual_block=nn.Sequential(*residual_block)
    self.res_op = res_op

  def forward(self, x,alpha=1.0,beta=0.0):
    residual = self.residual_block(x)
    f_x = self.model(x)
    if type(alpha)!=float:
        alpha=alpha.expand_as(f_x)
    if type(beta)!=float:
        beta=beta.expand_as(f_x)
    out = alpha * f_x + beta
    out = residual_op(out,residual,self.res_op) #out += residual
    return out

class GatedConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=False):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1,dilation=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1,dilation=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8,res_op='add'):
    super(GatedConvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride,use_sn=use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes,affine=True)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride , use_sn=use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups) ] #[nn.BatchNorm2d(planes,affine=True)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.res_op = res_op
  def forward(self, x,alpha=1.0,beta=0.0):
    residual = x
    if type(alpha)!=float:
        alpha=alpha.expand_as(x)
    if type(beta)!=float:
        beta= beta.expand_as(x)
    out = alpha*self.model(x) + beta
    out= residual_op(out,residual,self.res_op) #out += residual
    return out

###########################################################################
##### Versions of the Conv, Upconv and DownConv Resblocks without gating
###########################################################################

class UpConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8):
    super(UpConvResBlock, self).__init__()
    model = []
    model += upsampleLayer(inplanes , planes , upsample='nearest' , use_sn=use_sn)
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)] #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block +=  upsampleLayer(inplanes , planes , upsample='bilinear' , use_sn=use_sn)
    self.residual_block=nn.Sequential(*residual_block)

  def forward(self, x):
    residual = self.residual_block(x)
    out = self.model(x)
    out += residual
    return out



class DownConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8):
    super(DownConvResBlock, self).__init__()
    model = []
    model += downsampleLayer(inplanes,planes,downsample='avgpool',use_sn=use_sn)
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block += downsampleLayer(inplanes,planes,downsample='avgpool',use_sn=use_sn)
    self.residual_block=nn.Sequential(*residual_block)

  def forward(self, x):
    residual = self.residual_block(x)
    out = self.model(x)
    out += residual
    return out

class ConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=False):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1,dilation=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1,dilation=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8):
    super(ConvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride,use_sn=use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes,affine=True)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride , use_sn=use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups) ] #[nn.BatchNorm2d(planes,affine=True)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out


#####################################################

#class UpConvResBlock(nn.Module):
#  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
#    if use_sn:
#        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
#    else:
#        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)
#
#
#  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False):
#    super(UpConvResBlock, self).__init__()
#    model = []
#    model += [nn.Upsample( scale_factor=2,mode='nearest')]
#    model += [self.conv3x3(inplanes, planes, stride,use_sn)]
#    model += [nn.BatchNorm2d(planes)]
#    model += [nn.ReLU(inplace=True)]
#    model += [self.conv3x3(planes, planes,stride,use_sn)]
#    model += [nn.BatchNorm2d(planes)]
#    model += [nn.ReLU(inplace=True)]
#    if dropout > 0:
#      model += [nn.Dropout(p=dropout)]
#    self.model = nn.Sequential(*model)
#
#    residual_block = []
#    residual_block += [nn.Upsample(scale_factor=2,mode='nearest')]
#    residual_block += [self.conv3x3(inplanes,planes,stride,use_sn)]
#    #self.model.apply(gaussian_weights_init)
#    self.residual_block=nn.Sequential(*residual_block)
#
#  def forward(self, x):
#    residual = self.residual_block(x)
#    out = self.model(x)
#    out += residual
#    return out
#
#class DownConvResBlock(nn.Module):
#  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
#    if use_sn:
#        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
#    else:
#        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)
#
#
#  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False):
#    super(DownConvResBlock, self).__init__()
#    model = []
#    model += [nn.AvgPool2d(2, stride=2)]
#    model += [self.conv3x3(inplanes, planes, stride,use_sn)]
#    model += [nn.BatchNorm2d(planes)]
#    model += [nn.ReLU(inplace=True)]
#    model += [self.conv3x3(planes, planes,stride,use_sn)]
#    model += [nn.BatchNorm2d(planes)]
#    model += [nn.ReLU(inplace=True)]
#    if dropout > 0:
#      model += [nn.Dropout(p=dropout)]
#    self.model = nn.Sequential(*model)
#
#    residual_block = []
#    residual_block += [self.conv3x3(inplanes,planes,stride,use_sn)]
#    residual_block += [nn.AvgPool2d(2, stride=2)]
#    #self.model.apply(gaussian_weights_init)
#    self.residual_block=nn.Sequential(*residual_block)
#
#  def forward(self, x):
#    residual = self.residual_block(x)
#    out = self.model(x)
#    out += residual
#    return out


#class ConvResBlock(nn.Module):
#  def conv3x3(self, inplanes, out_planes, stride=1):
#    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)
#
#  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
#    super(ConvResBlock, self).__init__()
#    model = []
#    model += [self.conv3x3(inplanes, planes, stride)]
#    #model += [nn.InstanceNorm2d(planes)]
#    model += [nn.ReLU(inplace=True)]
#    model += [self.conv3x3(planes, planes)]
#    #model += [nn.InstanceNorm2d(planes)]
#    if dropout > 0:
#      model += [nn.Dropout(p=dropout)]
#    self.model = nn.Sequential(*model)
#    #self.model.apply(gaussian_weights_init)
#
#  def forward(self, x):
#    residual = x
#    out = self.model(x)
#    out += residual
#    return out

class BATCHConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(BATCHConvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    #self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out


class MAX_SELECTResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(MAX_SELECTResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out=torch.max(out,residual)
        return out

class MAX_PARALLELResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(MAX_PARALLELResBlock, self).__init__()

        model_1 = []
        model_1 += [nn.Linear(num_neurons,num_neurons)]
        model_1 += [nn.ReLU(inplace=True)]
        model_1 += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model_1 += [nn.Dropout(p=dropout)]
        self.model_1 = nn.Sequential(*model_1)

        model_2 = []
        model_2 += [nn.Linear(num_neurons,num_neurons)]
        model_2 += [nn.ReLU(inplace=True)]
        model_2 += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model_2 += [nn.Dropout(p=dropout)]
        self.model_2 = nn.Sequential(*model_2)


    def forward(self,x):
        residual=x
        out_1=self.model_1(x)
        out_2=self.model_2(x)
        out_max=torch.max(out_1,out_2)
        out = residual + out_max
        return out


class RELUResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(RELUResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU()]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out

class LinearRELUBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(LinearRELUBlock,self).__init__()
        model=[]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU()]
        if dropout>0:
            model+= [nn.Dropout(p=dropout)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
       out=self.model(x)
       return out
