from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # ============================================loss=========================================================
        # L1 and local
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_local', type=float, default=25.0, help='weight for Local loss')
        # chamfer loss
        parser.add_argument('--lambda_chamfer', type=float, default=0.1, help='weight for chamfer loss')
        parser.add_argument('--lambda_chamfer2', type=float, default=0.1, help='weight for chamfer loss2')
        # =====================================auxilary net structure===============================================
        # dt & line net structure
        parser.add_argument('--netG_dt', type=str, default='unet_512', help='selects model to use for netG_dt, for chamfer loss')
        parser.add_argument('--netG_line', type=str, default='unet_512', help='selects model to use for netG_line, for chamfer loss')
        # multiple discriminators
        parser.add_argument('--discriminator_local', action='store_true', help='use six diffent local discriminator for 6 local regions')
        parser.add_argument('--gan_loss_strategy', type=int, default=2, help='specify how to calculate gan loss for g, 1: average global and local discriminators; 2: not change global discriminator weight, 0.25 for local')
        parser.add_argument('--addw_eye', type=float, default=1.0, help='additional weight for eye region')
        parser.add_argument('--addw_nose', type=float, default=1.0, help='additional weight for nose region')
        parser.add_argument('--addw_mouth', type=float, default=1.0, help='additional weight for mouth region')
        parser.add_argument('--addw_hair', type=float, default=1.0, help='additional weight for hair region')
        parser.add_argument('--addw_bg', type=float, default=1.0, help='additional weight for bg region')
        # ==========================================ablation========================================================
        parser.add_argument('--no_l1_loss', action='store_true', help='no l1 loss')
        parser.add_argument('--no_G_local_loss', action='store_true', help='not using local transfer loss for local generator output')

        self.isTrain = True
        return parser
