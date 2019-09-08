<img src='imgs/pineapple.gif' align="right" width=330>

<br><br><br>

# iSketchNFill

We propose an interactive GAN-based sketch-to-image translation method
that helps novice users easily create images of simple objects.
The user starts with a sparse sketch and a desired object category, and the network then recommends its plausible completion(s) and shows a corresponding synthesized image. This enables a feedback loop, where the user can edit the sketch based on the network's recommendations, while the network is able to better synthesize the image that the user might have in mind.
In order to use a single model for a wide array of object classes, we introduce a gating-based approach for class conditioning, which allows us to generate distinct classes without feature mixing, from a single generator network.


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

- Install PyTorch and dependencies from http://pytorch.org
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
python main_shadow_draw.py --args
```


#### iSketchNFill: [[Project]](https://arnabgho.github.io/iSketchNFill/) [[Paper]](http://www.robots.ox.ac.uk/~tvg/publications/2019/ICCV_ISF_camera_ready.pdf)
<img src ='imgs/teaser_v7.png' width="1000px"/>



## Citation
If you use this code for your research, please cite our paper.
```
@article{iSketchNFill2019,
title={Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation},
author={Ghosh, Arnab and Zhang, Richard and Dokania, Puneet K and Wang, Oliver and Efros, Alexei A and Torr, Philip H S and Shechtman, Eli},
journal = {arxiv},
year = {2019}
}
```
## Acknowledgement
Code is inspired by [pytorch-CycleGAN-and-pix2pix]( https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix ).
