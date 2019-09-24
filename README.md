<img src='imgs/pineapple.gif' align="right" width=330>

<br><br><br>

# iSketchNFill

We propose an interactive GAN-based sketch-to-image translation method
that helps novice users easily create images of simple objects.
The user starts with a sparse sketch and a desired object category, and the network then recommends its plausible completion(s) and shows a corresponding synthesized image. This enables a feedback loop, where the user can edit the sketch based on the network's recommendations, while the network is able to better synthesize the image that the user might have in mind.
In order to use a single model for a wide array of object classes, we introduce a gating-based approach for class conditioning, which allows us to generate distinct classes without feature mixing, from a single generator network.

#### iSketchNFill: [[Project]](https://arnabgho.github.io/iSketchNFill/) [[Paper]](http://www.robots.ox.ac.uk/~tvg/publications/2019/ICCV_ISF_camera_ready.pdf)
<img src ='docs/resources/imgs/teaser.png' width="1000px"/>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

- Install PyTorch 1.0+ and dependencies from http://pytorch.org
- Install Torchvision
- Install all requirements
```
pip install -r requirements.txt
```
- Clone this repo:
```
git clone https://github.com/arnabgho/iSketchNFill
cd iSketchNFill
```


### Interactive session with a pre-trained model (iSketchNFill)

- Download pretrained models


```
bash scripts/download_pretrained_scribble_dataset.sh
```

- Play with the interface

```
python main_gui_shadow_draw_color.py --name wgangp_sparse_label_channel_pix2pix_autocomplete_multiscale_nz_256_nc_1_nf_32_gp_0_multigpu --model sparse_wgangp_pix2pix  --checkpoints_dir checkpoints_sparse --gpu_ids 0 --nz 256 --sparseSize 4 --fineSize 128 --ngf 32 --ndf 32 --num_interpolate 8 --input_nc 1 --output_nc 1
```

- One can change the number of shadows with the parameter --num_interpolate
- One can change the quality of the completed results by varying the standard deviation --test_std, lower values indicate higher quality while higher values indicate lower quality completions.

## Citation
If you use this code for your research, please cite our paper.
```
@article{iSketchNFill2019,
title={Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation},
author={Ghosh, Arnab and Zhang, Richard and Dokania, Puneet K. and Wang, Oliver and Efros, Alexei A. and Torr, Philip H. S. and Shechtman, Eli},
journal = {arxiv},
year = {2019}
}
```
## Acknowledgement
Code is inspired by [pytorch-CycleGAN-and-pix2pix]( https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix ).
