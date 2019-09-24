
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pix2pixhd':
        assert(opt.dataset_mode == 'aligned' or opt.dataset_mode=='labeled')
        from .pix2pixhd_model import Pix2PixModel
        model = Pix2PixModel(opt)

    elif opt.model == 'sparse_wgangp_pix2pix':
        assert(opt.dataset_mode == 'labeled')
        from .sparse_wgangp_pix2pix_model import SparseWGANGPPix2PixModel
        model = SparseWGANGPPix2PixModel()


    elif opt.model == 'stochastic_label_channel_gated_pix2pix':
        assert(opt.dataset_mode == 'labeled')
        from .stochastic_label_channel_gated_pix2pix_model import StochasticLabelChannelGatedPix2PixModel
        model = StochasticLabelChannelGatedPix2PixModel()


    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
