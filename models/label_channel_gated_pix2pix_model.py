import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn

class LabelChannelGatedPix2PixModel(BaseModel):
    def name(self):
        return 'LabelChannelGatedPix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.label = self.Tensor(opt.batchSize,1)
        if opt.nz>0:
            self.noise=self.Tensor(opt.batchSize,opt.nz)
        # load/define networks
        if opt.which_model_gated_netG=='pix2pix':
            opt.which_model_netG = 'ChannelGatedResnetGenerator'
        else :
            opt.which_model_netG = 'LabelChannelGatedResnetConvResnetG'


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            opt.which_model_netD = 'LabelChannelGatedResnetConvResnetD'
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,opt)
            if torch.cuda.device_count()>1:
                self.netD = nn.DataParallel(self.netD)
                self.netD.to(device)
            else:
                self.netD.cuda()
        if torch.cuda.device_count()>1:
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
            self.criterionGAN = networks.GANLoss(use_lsgan=opt.lsgan, tensor=self.Tensor)
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

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.label = input['label']
        self.label = self.label.cuda()
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.nz>0:
            self.noise.normal_(0,1)

    def forward(self):
        self.real_A = Variable(self.input_A)
        if self.opt.nz>0:
            self.fake_B = self.netG(self.real_A,self.label,self.noise)
        else:
            self.fake_B = self.netG(self.real_A,self.label)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        if self.opt.nz>0:
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


        for i in range(num_interpolate):
            alpha_blend = alpha_blends[i]
            self.noise = noise_1 * alpha_blend + noise_2 * (1-alpha_blend)
            self.fake_B = self.netG(self.real_A,self.label,self.noise)
            output_gate = self.netG.forward_gate(self.label,self.noise)
            print(output_gate)
            results['%d_L_fake_B_inter'%(i)]=util.tensor2im(self.fake_B.data)

        return OrderedDict(results)


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
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach(),self.label)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB,self.label)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB,self.label)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

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
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
