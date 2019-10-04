import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import torchvision
from torch.nn import functional as F
from . import networks_sparse
from . import networks
import torch.autograd as autograd
import torch.nn as nn
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg



class SparseWGANGPPix2PixModel(BaseModel):
    def name(self):
        return 'SparseWGANGPPix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.sparse_input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.sparseSize, opt.sparseSize)
        self.mask_input_A = self.Tensor(opt.batchSize, 1,
                                   opt.fineSize, opt.fineSize)


        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.label = self.Tensor(opt.batchSize,1)
        if opt.nz>0:
            self.noise=self.Tensor(opt.batchSize,opt.nz)
            self.test_noise= self.get_z_random(opt.num_interpolate,opt.nz)
            self.test_noise.normal_(0,0.2)
        # load/define networks
        opt.which_model_netG = 'GAN_stability_Generator'


        self.netG = networks_sparse.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            opt.which_model_netD = 'GAN_stability_Discriminator'
            self.netD = networks_sparse.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,opt)

            if self.isTrain:
                self.netD = nn.DataParallel(self.netD)
                self.netD.to(device)
            else:
                self.netD.cuda()

        if self.isTrain:
            self.netG = nn.DataParallel(self.netG)
            self.netG.to(device)
        else:
            self.netG.cuda()

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.WGANLoss(tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr_g, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        z= z.cuda()
        return z

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        sparse_input_A = input['A_sparse']
        mask_input_A = input['A_mask']
        input_B = input['B' if AtoB else 'A']
        self.label = input['label']
        self.label = self.label.cuda()
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.sparse_input_A.resize_(sparse_input_A.size()).copy_(sparse_input_A)
        self.mask_input_A.resize_(mask_input_A.size()).copy_(mask_input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_A = Variable(self.input_A)
        self.sparse_real_A = Variable(self.input_A)
        self.real_A = self.sparse_real_A
        if self.opt.nz>0:
            self.noise = self.get_z_random(self.real_A.size(0),self.opt.nz)#self.noise.normal_(0,1)

    def forward(self):
        self.sparse_real_A = Variable(self.input_A)
        if self.opt.nz>0:
            self.fake_B = self.netG(self.real_A,self.label,self.noise)
        else:
            self.fake_B = self.netG(self.real_A,self.label)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.sparse_real_A = Variable(self.input_A, volatile=True)
        if self.opt.nz>0:
            self.noise.fill_(0)
            self.fake_B = self.netG(self.real_A,self.label,self.noise)
        else:
            self.fake_B = self.netG(self.real_A,self.label)
        self.real_B = Variable(self.input_B, volatile=True)


    def get_latent_space_visualization(self,num_interpolate=20,label_1=-1,label_2=-1):
        rand_perm = np.random.permutation( self.opt.n_classes  )
        if label_1 == -1:
            label_1 = self.label[0] #rand_perm[0]
        if label_2 == -1:
            label_2 = self.opt.target_label #rand_perm[1]
        alpha_blends = np.linspace(0,1,num_interpolate)
        self.label[0] = label_1
        output_gate_1 = self.netG.forward_gate(self.label)
        self.label[0] = label_2
        output_gate_2 = self.netG.forward_gate(self.label)
        results={}
        results['latent_real_A']=util.tensor2im(self.real_A.data)
        results['latent_real_B']=util.tensor2im(self.real_B.data)

        for i in range(num_interpolate):
            alpha_blend = alpha_blends[i]
            output_gate = output_gate_1*alpha_blend + output_gate_2*(1-alpha_blend)
            self.fake_B = self.netG.forward_main( self.real_A,output_gate)

            results['%d_L_fake_B_inter'%(i)]=util.tensor2im(self.fake_B.data)

        return OrderedDict(results)

    def get_latent_noise_visualization(self,num_interpolate=20):
        alpha_blends = np.linspace(0,1,num_interpolate)
        noise_1 = self.noise.clone()
        noise_1.normal_(0,1)
        noise_2 = self.noise.clone()
        noise_2.normal_(0,1)

        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)


        results={}
        results['latent_real_A']=util.tensor2im(self.real_A.data)
        results['latent_real_B']=util.tensor2im(self.real_B.data)


        shadow = None
        #self.fake_B = self.netG(self.real_A,self.label,self.test_noise)
        #shadow = self.fake_B.data.sum(0)
        self.fake_B = self.netG(self.real_A,self.label,self.test_noise)
        torchvision.utils.save_image(self.fake_B,'./imgs/fake_B_gallery.png',nrow=2)
        for i in range(self.opt.num_interpolate):
            alpha_blend = alpha_blends[i]
            #self.noise = self.test_noise #self.get_z_random(self.real_A.size(0),self.opt.nz)#noise_1 * alpha_blend + noise_2 * (1-alpha_blend)
            results['%d_L_fake_B_inter'%(i)]=util.tensor2im(self.fake_B.data[i].unsqueeze(0))
            if i==0:
                shadow = self.fake_B.data[i]
            else:
                shadow += self.fake_B.data[i]
        results['fake_B_shadow']=util.tensor2im(shadow.unsqueeze(0),normalize=True)
        return OrderedDict(results)

    def randomize_noise(self):
        #self.test_noise= self.get_z_random(self.opt.num_interpolate,self.opt.nz)
        self.test_noise.normal_(0,self.opt.test_std) # truncation trick to obtain better samples
    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def get_gate_activations_G(self,label):
        self.label[0]=label
        gate_act = self.netG.forward_gate(self.label)
        return gate_act.data.cpu().numpy()

    def get_gate_activations_D(self,label):
        self.label[0]=label
        gate_act = self.netD.forward_gate(self.label)
        return gate_act.data.cpu().numpy()



    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.img_conditional_D:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB.detach(),self.label)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        if self.opt.img_conditional_D:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = self.real_B

        pred_real = self.netD(real_AB,self.label)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()
        self.reg = self.opt.wgan_gp_lambda * self.wgan_gp_reg(real_AB,fake_AB,self.label,center= self.opt.wgan_gp_center)
        self.reg.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.img_conditional_D:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB,self.label)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # Second, G(A) = B
        mask_A_resized = self.mask_input_A.expand_as(self.fake_B)

        self.loss_G_L1 = self.criterionL1(self.fake_B*mask_A_resized, self.real_A) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.netD(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg




    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D_real', self.loss_D_real.data.item()),
                            ('D_fake', self.loss_D_fake.data.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        sparse_real_A = util.tensor2im(self.sparse_real_A.data)
        mask_real_A = util.tensor2im(self.mask_input_A)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('sparse_real_A', sparse_real_A),  ('mask_real_A', mask_real_A)   , ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
