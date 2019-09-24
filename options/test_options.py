from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--test_std', type=float, default=0.2, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--which_epoch_pix2pix', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--name_pix2pix', type=str, default='label_channel_pix2pix_lsgan_1', help='Name of the experiment')
        self.parser.add_argument('--model_pix2pix', type=str, default='stochastic_label_channel_gated_pix2pix', help='Which pix2pix model')
        self.parser.add_argument('--checkpoints_dir_pix2pix', type=str, default='checkpoints_stochastic_lsgan', help='Which pix2pix checkpoints directory')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--target_label', type=int, default=7, help='Number of channels in the images')
        self.parser.add_argument('--shadow', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--no_shadow_intermediate', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--disable_browser', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.isTrain = False
