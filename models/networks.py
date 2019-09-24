import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from .common_net import *
from torch.nn.utils import spectral_norm
import copy
from .gumbelmodule import GumbleSoftmax
try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range

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



###############################################################################
# Functions
###############################################################################

class log_gaussian:

  def __call__(self, x, mu, var):
    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    return logli.sum(1).mean().mul(-1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True) #affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[] , opt={} ):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netG == 'StochasticLabelBetaChannelGatedResnetConvResnetG' :
        netG = StochasticLabelBetaChannelGatedResnetConvResnetG(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    #init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[],opt={}):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'LabelChannelGatedResnetConvResnetD':
        netD = LabelChannelGatedResnetConvResnetD(opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    #init_weights(netD, init_type=init_type)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class WGANLoss(nn.Module):
    def __init__(self, use_wgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(WGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def loss(self,d_out,target):
        loss = (2*target - 1) * d_out.mean()
        return loss.mean()

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)




###################################################################
###### Channel-Wise Gating techniques (Multiplicative and Affine)
###################################################################



class StochasticLabelBetaChannelGatedResnetConvResnetG(nn.Module):
    def __init__(self,opt):
        super(StochasticLabelBetaChannelGatedResnetConvResnetG, self).__init__()
        self.opt=opt
        opt.nsalient = max(10,opt.n_classes)
        self.label_embedding = nn.Embedding(opt.n_classes, opt.nsalient)
        self.main_initial = nn.Sequential( nn.Conv2d(3,opt.ngf,kernel_size=3,stride=1,padding=1) ,
                            get_norm(opt.ngf,opt.norm_G,opt.num_groups),
                            nn.ReLU(True)
                            )
        self.label_noise = nn.Linear(opt.nz,opt.nsalient)
        main_block=[]
        #Input is z going to series of rsidual blocks

        # Sets of residual blocks start
        for i in range(3):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op)]


        for i in range(opt.ngres_up_down):
            main_block += [ DownGatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op) ]

        for i in range(int(opt.ngres/2-opt.ngres_up_down-3)):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op)]


        for i in range(int(opt.ngres/2-opt.ngres_up_down-3)):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op)]



        for i in range(opt.ngres_up_down):
            main_block += [ UpGatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op ) ]

        for i in range(3):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G , num_groups = opt.num_groups,res_op=opt.res_op )]



        # Final layer to map to 3 channel
        if opt.spectral_G:
            main_block+=[spectral_norm(nn.Conv2d(opt.ngf,opt.nc,kernel_size=3,stride=1,padding=1)) ]
        else:
            main_block+=[nn.Conv2d(opt.ngf,opt.nc,kernel_size=3,stride=1,padding=1) ]
        main_block+=[nn.Tanh()]
        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ Reshape( -1, 1 ,opt.nsalient)  ]
        gate_block+=[ nn.Conv1d(1,opt.ngf_gate,kernel_size=3,stride=1,padding=1)  ]
        gate_block+=[ nn.ReLU()]
        for i in range(opt.ngres_gate):
            gate_block+=[ResBlock1D(opt.ngf_gate,opt.dropout_gate)]
        # state size (opt.batchSize, opt.ngf_gate, opt.nsalient)
        gate_block+=[Reshape(-1,opt.ngf_gate*opt.nsalient)]

        self.gate=nn.Sequential(*gate_block)

        gate_block_mult = []
        gate_block_mult+=[ nn.Linear(opt.ngf_gate*opt.nsalient,opt.ngres*opt.ngf) ]
        gate_block_mult+= [ nn.Sigmoid()]

        self.gate_mult = nn.Sequential(*gate_block_mult)


        gate_block_add = gate_block
        gate_block_add+=[ nn.Linear(opt.ngf_gate*opt.nsalient,opt.ngres*opt.ngf) ]
        gate_block_add+= [nn.Hardtanh()]
        self.gate_add = nn.Sequential(*gate_block_add)


    def forward(self, input,labels,noise=None):
        input_gate = self.label_embedding(labels)
        input_noise=self.label_noise(noise)

        # Things are just flipped here
        output_gate = self.gate(input_noise)
        output_gate_mult = self.gate_mult(output_gate)
        output_gate_add = self.gate_add(input_gate)
        output = self.main_initial(input)
        for i in range(self.opt.ngres):
            alpha = output_gate_mult[:,i*self.opt.ngf:(i+1)*self.opt.ngf]
            alpha = alpha.resize(self.opt.batchSize,self.opt.ngf,1,1)
            beta=output_gate_add[:,i*self.opt.ngf:(i+1)*self.opt.ngf]
            beta=beta.resize(self.opt.batchSize,self.opt.ngf,1,1)
            output=self.main[i](output,alpha,beta)

        output=self.main[self.opt.ngres](output)
        output=self.main[self.opt.ngres+1](output)
        return output


class LabelChannelGatedResnetConvResnetD(nn.Module):
    def __init__(self,opt,input_nc=6, ndf=32, n_layers=0, norm_layer=nn.BatchNorm2d, use_sigmoid=True, gpu_ids=[],use_sn=False):
        super(LabelChannelGatedResnetConvResnetD, self).__init__()
        self.opt=opt
        opt.nsalient = max(10,opt.n_classes)
        self.label_embedding = nn.Embedding(opt.n_classes, opt.nsalient)
        use_sn = opt.spectral_D
        use_sigmoid = opt.no_lsgan
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ndf= opt.ndf
        kw = 4
        padw = 1

        sequence = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            #nf_mult = min(2**n, 8)
            if use_sn:
                sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    get_norm(ndf*nf_mult,opt.norm_D,opt.num_groups),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    get_norm(ndf*nf_mult,opt.norm_D,opt.num_groups),
                    nn.LeakyReLU(0.2, True)
                ]

        if use_sn:
            sequence += [spectral_norm( nn.Conv2d(ndf * nf_mult, opt.ndisc_out_filters, kernel_size=kw, stride=1, padding=padw) ) ]
        else:
            sequence += [ nn.Conv2d(ndf * nf_mult, opt.ndisc_out_filters, kernel_size=kw, stride=1, padding=padw) ]


        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.main_latter = nn.Sequential(*sequence)


        main_block=[]
        #Input is z going to series of rsidual blocks
        # First layer to map to ndf channel
        if opt.spectral_D:
            main_block+=[spectral_norm(nn.Conv2d(opt.input_nc + opt.output_nc,opt.ndf,kernel_size=3,stride=1,padding=1))]
        else:
            main_block+=[nn.Conv2d(opt.input_nc + opt.output_nc ,opt.ndf,kernel_size=3,stride=1,padding=1)]
        # Sets of residual blocks start

        for i in range(3):
            main_block+= [GatedConvResBlock(opt.ndf,opt.ndf,dropout=opt.dropout,use_sn=opt.spectral_D,norm_layer=opt.norm_D,num_groups=opt.num_groups,res_op=opt.res_op)]

        for i in range(opt.ndres_down):
            main_block+= [DownGatedConvResBlock(opt.ndf,opt.ndf,dropout=opt.dropout_D,use_sn=opt.spectral_D,norm_layer=opt.norm_D,num_groups=opt.num_groups,res_op=opt.res_op)]

        for i in range(opt.ndres - opt.ndres_down-3  ):
            main_block+= [GatedConvResBlock(opt.ndf,opt.ndf,dropout=opt.dropout_D,use_sn=opt.spectral_D,norm_layer=opt.norm_D , num_groups=opt.num_groups ,res_op=opt.res_op)]


        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ Reshape( -1, 1 ,opt.nsalient)  ]
        gate_block+=[ nn.Conv1d(1,opt.ngf_gate,kernel_size=3,stride=1,padding=1)  ]


        gate_block+=[ nn.ReLU()]
        for i in range(opt.ndres_gate):
            gate_block+=[ResBlock1D(opt.ndf_gate,opt.dropout_gate)]
        # state_size (opt.batchSize,opt.ndf_gate,opt.nsalient)
        gate_block+= [Reshape(-1,opt.ndf_gate*opt.nsalient)]

        self.gate = nn.Sequential(*gate_block)

        gate_block_mult=[]
        gate_block_mult+=[ nn.Linear(opt.ndf_gate*opt.nsalient,opt.ndres*opt.ndf) ]
        gate_block_mult+= [nn.Sigmoid()]

        self.gate_mult = nn.Sequential(*gate_block_mult)

        if opt.gate_affine:
            gate_block_add = []
            gate_block_add+=[ nn.Linear(opt.ndf_gate*opt.nsalient,opt.ndres*opt.ndf) ]
            gate_block_add+=[nn.Tanh()]
            self.gate_add=nn.Sequential(*gate_block_add)

    def forward(self, img, labels):
        batchSize=labels.size(0)
        input_gate = self.label_embedding(labels)
        input_main = img

        output_gate = self.gate(input_gate)
        output = self.main[0](img)
        output_gate_mult = self.gate_mult(output_gate)
        if self.opt.gate_affine:
            output_gate_add = self.gate_add(output_gate)
        for i in xrange(1,1+self.opt.ndres):
            alpha = output_gate_mult[:,(i-1)*self.opt.ndf:i*self.opt.ndf]
            alpha = alpha.resize(batchSize,self.opt.ndf,1,1)
            if self.opt.gate_affine:
                beta=output_gate_add[:,(i-1)*self.opt.ndf:i*self.opt.ndf]
                beta=beta.resize(batchSize,self.opt.ndf,1,1)
                output=self.main[i](output,alpha,beta)
            else:
                output=self.main[i](output,alpha)

        output = self.main_latter(output)
        return output


