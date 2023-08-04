# Image-based Fashion Clothing Try-on with Style Transfer

based on: [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON) and [PyTorch Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

File Sturecture
```
.
├── test                    # Contains Inference code for Flow-Style-VTON and the majority of our added code.
│   ├── checkpoints         # would contain checkpoints for generator models and StyleGAN warp models
│   │   ├── ...
│   ├── data                # data loader for [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON)
│   │   ├── ...
│   ├── dataset             # Initial 6 triples of images and styles used to test our Clothes try-on with Style Transfer algorithms
│   │   ├── styles
│   │   ├── test_clothes
│   │   ├── test_edge
│   │   ├── test_img
│   ├── dataset2            # A second dataset from: [kaggle](https://www.kaggle.com/datasets/rkuo2000/viton-dataset) likely taken from [ACGPN paper](https://github.com/switchablenorms/DeepFashion_Try_On)
│   │   ├── styles          # not included, but added here
│   │   ├── test_clothes    # from kaggle dataset; change name from test_color to test_clothes
│   │   ├── test_edge       # from kaggle dataset
│   │   ├── test_img        # from kaggle dataset
│   │   ├── README.txt
│   ├── models              # model definitions from [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON)
│   │   ├── ...
│   ├── options             # options definitions from [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON)
│   │   ├── ...
│   ├── results             # results from running our evaluation/experiments. Not included in this repository because it is too large.
│   │   ├── ...
│   ├── util                # util function definitions from [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON)
│   │   ├── ...
│   ├── test_pairs.txt      # file that defines which pair of `person` and `clothes` to match. 
│   ├── test.py             # Inference code for [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON)
│   ├── test.sh             # Inference script for [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON)

│   ├── set_up_test_pairs.py    # set up the `test_pairs.txt` which is used to determine which pair of `person` and `clothes` to match. Randomly selects 500 people and clothes. CHANGE `dataroot` variable to change the dataset. Default is `dataset2`
│   ├── fid_scoring.sh          # bash script for evaluation using [pytorch ID score](https://github.com/mseitzer/pytorch-fid)
│   ├── ssim_scoring.py         # python script for evaluation using [pytorch ssim](https://github.com/VainF/pytorch-msssim)
│   ├── vton-no-NST.py                      # virtual try-on without style transfer (similar to test.py)
│   ├── stylized-vton-on-warp.py            # virtual try-on with Simple Segmented Neural Style Transfer(SSNST) does SSNST on warped clothes
│   ├── stylized-vton-before_warp.py        # virtual try-on with SSNST; does SSNST on clothes before warp
│   ├── stylized-vton-after-warp.py         # virtual try-on with normal Neural Style Transfer(NST); does NST on the result of Flow-Style-VTON
│   ├── stylized-vton-on-warp-no-mask.py    # virtual try-on with normal Neural Style Transfer(NST); does NST on the warped clothes (does not add background)

│   ├── requirements.txt    # python requirements

├── train                   # Contains training code. See [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON) for more details
│   ├── ...
└── README.txt
```

## Requirements

- Anaconda or Miniconda
- python 3.6.13
- torch 1.1.0 (as no third party libraries are required in this codebase, other versions should work, not yet tested)
- torchvision 0.3.0
- tensorboardX
- opencv

```
conda install -y pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
conda install -y cupy matplotlib
pip install opencv-python imageio natsort

cd test
pip install -r requirements.txt
```

## Inference (`cd` to test folder)

All inference, evaluation, and experiments are done from `Flow-Style-VTON/test`

## Test
[FID](https://github.com/mseitzer/pytorch-fid) and [SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim)

## Training ( `cd` to the train folder)

Use the data set found [here](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing) from [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON)

VITON_traindata structure:
```
.
├── train_color         # just clothes (in color)
├── train_densepose     # obtained from [Dense pose](https://github.com/facebookresearch/DensePose)
├── train_edge          # segmented edges of clothes (from train_color)
├── train_img           # image of person wearing clothes (plain background)
├── train_label         # label of training image
├── train_pose          # pose data likely obtained from [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
```

Used the checkpoint of vgg from [VGG_Model](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing) and put into the folder `train/models`. This is used for perceptual loss computation

(note: from [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON))
### Stage 1: Parser-Based Appearance Flow Style
```
sh scripts/train_PBAFN_stage1_fs.sh
```
### Stage 2: Parser-Based Generator
```
sh scripts/train_PBAFN_e2e_fs.sh
```

### Stage 3: Parser-Free Appearance Flow Style
```
sh scripts/train_PFAFN_stage1_fs.sh
```

### Stage 4: Parser-Free Generator
```
sh scripts/train_PFAFN_e2e_fs.sh
```

## References

see [Flow-Style-VTON](https://github.com/SenHe/Flow-Style-VTON) for more details on inference and training the virtual try-on
```
@inproceedings{he2022fs_vton,
  title={Style-Based Global Appearance Flow for Virtual Try-On},
  author={He, Sen and Song, Yi-Zhe and Xiang, Tao},
  booktitle={CVPR},
  year={2022}
}
```

and [PyTorch Neural Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) about the implementation of Neural Style Transfer