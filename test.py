import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.loadSize=256
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
if opt.dataset_mode == 'labeled':
    opt.n_classes = data_loader.get_dataset().num_classes
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

if opt.repeat_block:
    web_dir_repeat = os.path.join(opt.results_dir, opt.name, 'repeat', '%s_%s' % (opt.phase, opt.which_epoch))
    webpage_repeat = html.HTML( web_dir_repeat, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('i=%s ..  process image... %s' % (i,img_path))
    visualizer.save_images(webpage, visuals, img_path)

    if opt.repeat_block:
        model.repeat_block()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('Repeat i=%s ..  process image... %s' % (i,img_path))
        visualizer.save_images(webpage_repeat, visuals, img_path)

    if opt.interpolate:
        model.set_input(data)
        model.test()
        visuals=model.get_latent_space_visualization()
        img_path = model.get_image_paths()
        visualizer.save_images(webpage, visuals, img_path)

    if opt.interpolate_noise:
        model.set_input(data)
        model.test()
        visuals=model.get_latent_noise_visualization()
        img_path = model.get_image_paths()
        visualizer.save_images(webpage, visuals, img_path)



    if i<20 and (opt.model=="infogan_pix2pix" or opt.model=='infogan_shared_pix2pix') :
        visuals=model.get_latent_space_visualization()
        visualizer.save_images(webpage, visuals, img_path)
        print("Latent Space Visualization Success")
        #break
webpage.save()
if opt.repeat_block:
    webpage_repeat.save()
