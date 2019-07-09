from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
         
        parser.set_defaults(dataset_mode='single')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G']
        self.auxiliary_model_names = []
        if self.opt.use_local:
            self.model_names += ['GLEyel','GLEyer','GLNose','GLMouth','GLHair','GLBG','GCombine']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.nnG)
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
        

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        if self.opt.use_local:
            self.real_A_eyel = input['eyel_A'].to(self.device)
            self.real_A_eyer = input['eyer_A'].to(self.device)
            self.real_A_nose = input['nose_A'].to(self.device)
            self.real_A_mouth = input['mouth_A'].to(self.device)
            self.center = input['center']
            self.real_A_hair = input['hair_A'].to(self.device)
            self.real_A_bg = input['bg_A'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.mask2 = input['mask2'].to(self.device)

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
            
            # HAIR, BG AND PARTCOMBINE
            fake_B_hair = self.netGLHair(self.real_A_hair)
            fake_B_bg = self.netGLBG(self.real_A_bg)
            self.fake_B_hair = self.masked(fake_B_hair,self.mask*self.mask2)
            self.fake_B_bg = self.masked(fake_B_bg,self.inverse_mask(self.mask2))
            self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op)
            
            # FUSION NET
            self.fake_B = self.netGCombine(torch.cat([self.fake_B0,self.fake_B1],1))
