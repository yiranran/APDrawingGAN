import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class APDrawingGANModel(BaseModel):
    def name(self):
        return 'APDrawingGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')# no_lsgan=True, use_lsgan=False
        parser.set_defaults(dataset_mode='aligned')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if self.isTrain and self.opt.no_l1_loss:
            self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        if self.isTrain and self.opt.use_local and not self.opt.no_G_local_loss:
            self.loss_names.append('G_local')
        if self.isTrain and self.opt.discriminator_local:
            self.loss_names.append('D_real_local')
            self.loss_names.append('D_fake_local')
            self.loss_names.append('G_GAN_local')
        if self.isTrain:
            self.loss_names.append('G_chamfer')
            self.loss_names.append('G_chamfer2')
        self.loss_names.append('G')
        print('loss_names', self.loss_names)
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.opt.use_local:
            self.visual_names += ['fake_B0', 'fake_B1']
            self.visual_names += ['fake_B_hair', 'real_B_hair', 'real_A_hair']
            self.visual_names += ['fake_B_bg', 'real_B_bg', 'real_A_bg']
        if self.isTrain:
            self.visual_names += ['dt1', 'dt2', 'dt1gt', 'dt2gt']
        if not self.isTrain and self.opt.save2:
            self.visual_names = ['real_A', 'fake_B']
        print('visuals', self.visual_names)
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
            if self.opt.discriminator_local:
                self.model_names += ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            # auxiliary nets for loss calculation
            self.auxiliary_model_names = ['DT1', 'DT2', 'Line1', 'Line2']
        else:  # during test time, only load Gs
            self.model_names = ['G']
            self.auxiliary_model_names = []
        if self.opt.use_local:
            self.model_names += ['GLEyel','GLEyer','GLNose','GLMouth','GLHair','GLBG','GCombine']
        print('model_names', self.model_names)
        print('auxiliary_model_names', self.auxiliary_model_names)
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.nnG)
        print('netG', opt.netG)

        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            print('netD', opt.netD, opt.n_layers_D)
            if self.opt.discriminator_local:
                self.netDLEyel = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLEyer = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLNose = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLMouth = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLHair = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLBG = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                
        
        if self.opt.use_local:
            self.netGLEyel = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 3)
            self.netGLEyer = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 3)
            self.netGLNose = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 3)
            self.netGLMouth = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 3)
            self.netGLHair = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet2', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 4)
            self.netGLBG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'partunet2', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 4)
            self.netGCombine = networks.define_G(2*opt.output_nc, opt.output_nc, opt.ngf, 'combiner', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 2)
        

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            if not self.opt.use_local:
                print('G_params 1 components')
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                G_params = list(self.netG.parameters()) + list(self.netGLEyel.parameters()) + list(self.netGLEyer.parameters()) + list(self.netGLNose.parameters()) + list(self.netGLMouth.parameters()) + list(self.netGLHair.parameters()) + list(self.netGLBG.parameters()) + list(self.netGCombine.parameters()) 
                print('G_params 8 components')
                self.optimizer_G = torch.optim.Adam(G_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if not self.opt.discriminator_local:
                print('D_params 1 components')
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                D_params = list(self.netD.parameters()) + list(self.netDLEyel.parameters()) +list(self.netDLEyer.parameters()) + list(self.netDLNose.parameters()) + list(self.netDLMouth.parameters()) + list(self.netDLHair.parameters()) + list(self.netDLBG.parameters())
                print('D_params 7 components')
                self.optimizer_D = torch.optim.Adam(D_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        # ==================================auxiliary nets (loaded, parameters fixed)=============================
        if self.isTrain:
            self.nc = 1
            self.netDT1 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_dt, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDT2 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_dt, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.set_requires_grad(self.netDT1, False)
            self.set_requires_grad(self.netDT2, False)
            
            self.netLine1 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_line, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netLine2 = networks.define_G(self.nc, self.nc, opt.ngf, opt.netG_line, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.set_requires_grad(self.netLine1, False)
            self.set_requires_grad(self.netLine2, False)
        

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.use_local:
            self.real_A_eyel = input['eyel_A'].to(self.device)
            self.real_A_eyer = input['eyer_A'].to(self.device)
            self.real_A_nose = input['nose_A'].to(self.device)
            self.real_A_mouth = input['mouth_A'].to(self.device)
            self.real_B_eyel = input['eyel_B'].to(self.device)
            self.real_B_eyer = input['eyer_B'].to(self.device)
            self.real_B_nose = input['nose_B'].to(self.device)
            self.real_B_mouth = input['mouth_B'].to(self.device)
            self.center = input['center']
            self.real_A_hair = input['hair_A'].to(self.device)
            self.real_B_hair = input['hair_B'].to(self.device)
            self.real_A_bg = input['bg_A'].to(self.device)
            self.real_B_bg = input['bg_B'].to(self.device)
            self.mask = input['mask'].to(self.device) # mask for non-eyes,nose,mouth
            self.mask2 = input['mask2'].to(self.device) # mask for non-bg
        if self.isTrain:
            self.dt1gt = input['dt1gt'].to(self.device)
            self.dt2gt = input['dt2gt'].to(self.device)
        

    def forward(self):
        if not self.opt.use_local:
            self.fake_B = self.netG(self.real_A)
        else:
            self.fake_B0 = self.netG(self.real_A)
            # EYES, NOSE, MOUTH
            fake_B_eyel = self.netGLEyel(self.real_A_eyel)
            fake_B_eyer = self.netGLEyer(self.real_A_eyer)
            fake_B_nose = self.netGLNose(self.real_A_nose)
            fake_B_mouth = self.netGLMouth(self.real_A_mouth)
            self.fake_B_nose = fake_B_nose
            self.fake_B_eyel = fake_B_eyel
            self.fake_B_eyer = fake_B_eyer
            self.fake_B_mouth = fake_B_mouth
            
            # HAIR, BG AND PARTCOMBINE
            fake_B_hair = self.netGLHair(self.real_A_hair)
            fake_B_bg = self.netGLBG(self.real_A_bg)
            self.fake_B_hair = self.masked(fake_B_hair,self.mask*self.mask2)
            self.fake_B_bg = self.masked(fake_B_bg,self.inverse_mask(self.mask2))
            self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op)
            
            # FUSION NET
            self.fake_B = self.netGCombine(torch.cat([self.fake_B0,self.fake_B1],1))

    
        
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1)) # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        if self.opt.discriminator_local:
            fake_AB_parts = self.getLocalParts(fake_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_fake_local = 0
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_fake_tmp = net(fake_AB_parts[i].detach())
                addw = self.getaddw(local_names[i])
                self.loss_D_fake_local = self.loss_D_fake_local + self.criterionGAN(pred_fake_tmp, False) * addw
            self.loss_D_fake = self.loss_D_fake + self.loss_D_fake_local

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        if self.opt.discriminator_local:
            real_AB_parts = self.getLocalParts(real_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_real_local = 0
            for i in range(len(real_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_real_tmp = net(real_AB_parts[i])
                addw = self.getaddw(local_names[i])
                self.loss_D_real_local = self.loss_D_real_local + self.criterionGAN(pred_real_tmp, True) * addw
            self.loss_D_real = self.loss_D_real + self.loss_D_real_local

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        if self.opt.discriminator_local:
            fake_AB_parts = self.getLocalParts(fake_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_G_GAN_local = 0
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_fake_tmp = net(fake_AB_parts[i])
                addw = self.getaddw(local_names[i])
                self.loss_G_GAN_local = self.loss_G_GAN_local + self.criterionGAN(pred_fake_tmp, True) * addw
            if self.opt.gan_loss_strategy == 1:
                self.loss_G_GAN = (self.loss_G_GAN + self.loss_G_GAN_local) / (len(fake_AB_parts) + 1)
            elif self.opt.gan_loss_strategy == 2:
                self.loss_G_GAN_local = self.loss_G_GAN_local * 0.25
                self.loss_G_GAN = self.loss_G_GAN + self.loss_G_GAN_local

        # Second, G(A) = B
        if not self.opt.no_l1_loss:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        if self.opt.use_local and not self.opt.no_G_local_loss:
            local_names = ['eyel','eyer','nose','mouth','hair','bg']
            self.loss_G_local = 0
            for i in range(len(local_names)):
                fakeblocal = getattr(self, 'fake_B_' + local_names[i])
                realblocal = getattr(self, 'real_B_' + local_names[i])
                addw = self.getaddw(local_names[i])
                self.loss_G_local = self.loss_G_local + self.criterionL1(fakeblocal,realblocal) * self.opt.lambda_local * addw

        # Third, distance transform loss (chamfer matching)
        if self.fake_B.shape[1] == 3:
            tmp = self.fake_B[:,0,...]*0.299+self.fake_B[:,1,...]*0.587+self.fake_B[:,2,...]*0.114
            fake_B_gray = tmp.unsqueeze(1)
        else:
            fake_B_gray = self.fake_B
        if self.real_B.shape[1] == 3:
            tmp = self.real_B[:,0,...]*0.299+self.real_B[:,1,...]*0.587+self.real_B[:,2,...]*0.114
            real_B_gray = tmp.unsqueeze(1)
        else:
            real_B_gray = self.real_B
        
        # d_CM(a_i,G(p_i))
        self.dt1 = self.netDT1(fake_B_gray)
        self.dt2 = self.netDT2(fake_B_gray)
        dt1 = self.dt1/2.0+0.5#[-1,1]->[0,1]
        dt2 = self.dt2/2.0+0.5
        
        bs = real_B_gray.shape[0]
        real_B_gray_line1 = self.netLine1(real_B_gray)
        real_B_gray_line2 = self.netLine2(real_B_gray)
        self.loss_G_chamfer = (dt1[(real_B_gray<0)&(real_B_gray_line1<0)].sum() + dt2[(real_B_gray>=0)&(real_B_gray_line2>=0)].sum()) / bs * self.opt.lambda_chamfer

        # d_CM(G(p_i),a_i)
        dt1gt = self.dt1gt
        dt2gt = self.dt2gt
        self.dt1gt = (self.dt1gt-0.5)*2
        self.dt2gt = (self.dt2gt-0.5)*2

        fake_B_gray_line1 = self.netLine1(fake_B_gray)
        fake_B_gray_line2 = self.netLine2(fake_B_gray)
        self.loss_G_chamfer2 = (dt1gt[(fake_B_gray<0)&(fake_B_gray_line1<0)].sum() + dt2gt[(fake_B_gray>=0)&(fake_B_gray_line2>=0)].sum()) / bs * self.opt.lambda_chamfer2
                

        self.loss_G = self.loss_G_GAN
        if 'G_L1' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_L1
        if 'G_local' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_local
        if 'G_chamfer' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_chamfer
        if 'G_chamfer2' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_chamfer2

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True) # enable backprop for D
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, True)
            self.set_requires_grad(self.netDLEyer, True)
            self.set_requires_grad(self.netDLNose, True)
            self.set_requires_grad(self.netDLMouth, True)
            self.set_requires_grad(self.netDLHair, True)
            self.set_requires_grad(self.netDLBG, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False) # D requires no gradients when optimizing G
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, False)
            self.set_requires_grad(self.netDLEyer, False)
            self.set_requires_grad(self.netDLNose, False)
            self.set_requires_grad(self.netDLMouth, False)
            self.set_requires_grad(self.netDLHair, False)
            self.set_requires_grad(self.netDLBG, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
