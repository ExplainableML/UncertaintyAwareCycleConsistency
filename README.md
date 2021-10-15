# UncertaintyAwareCycleConsistency
This repository provides the building blocks and the API for the work presented in the NeurIPS'21 paper
[Robustness via Uncertainty-aware Cycle Consistency]().
Translation methods often learn deterministic mappings without explicitly modelling the robustness to outliers or predictive uncertainty, leading to performance degradation when encountering unseen perturbations at test time. To address this, we propose a method based on Uncertainty-aware Generalized Adaptive Cycle Consistency (UGAC), which models the per-pixel residual by generalized Gaussian distribution, capable of modelling heavy-tailed distributions.

![](./arch.png)

### Requirements
```
python >= 3.6.10
pytorch >= 1.6.0
jupyter lab
torchio
scikit-image
scikit-learn
```

The structure of the repository is as follows:
```
root
 |-ckpt/ (will save all the checkpoints)
 |-data/ (save your data and related script)
 |-src/ (contains all the source code)
    |-ds.py 
    |-networks.py
    |-utils.py
    |-losses.py
```

### Preparing Datasets
To prepare your datasets to use with this repo, place the root directory of the dataset in `data/`.
The recommended way to structure your data is shown below.
```
data/
    |-Dataset_1/
        |-A/
            |-image1.png
            |-image2.png
            |-image3.png
            |-...
        |-B/
            |-image1.png
            |-image2.png
            |-image3.png
            |-...
```
Note the images need not be paired. The python script `src/ds.py` provides the PyTorch `Dataset` class to read such a dataset, used as explained below.
```python
class Images_w_nameList(data.Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, root1, root2, flist1, flist2, transform=None):
```
Here `root1` and `root2` represents the root directory for domain A and B, respectively.
`flist1` and `flist2` contain image names for domain A and domain B. Note, if `flist1` and `flist2` are aligned then dataset will load paired images. To use it as unsupervised dataset loader ensure that `flist1` and `flist2` are not aligned.



### Learning models with uncertainty
`src/networks.py` provides the generator and discriminator architectures.

`src/utils.py` provides two training APIs `train_i2i_UNet3headGAN` and `train_i2i_Cas_UNet3headGAN`. The first API is to be used to train the primary GAN, whereas the second API is to be used to train the subsequent GANs. 

An example command to use the first API is:
```python
netG_A = CasUNet_3head(1,1)
netD_A = NLayerDiscriminator(1, n_layers=4)
netG_A, netD_A = train_i2i_UNet3headGAN(
    netG_A, netD_A,
    train_loader, test_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-5,
    ckpt_path='../ckpt/i2i_0_UNet3headGAN',
)
```
This will save checkpoints in `../ckpt/` named as `i2i_0_UNet3headGAN_eph*.pth`

An example command to use the second API (here we assumed the primary GAN and first subsequent GAN are trained already):
```python
# first load the prior Generators 
netG_A1 = CasUNet_3head(1,1)
netG_A1.load_state_dict(torch.load('../ckpt/i2i_0_UNet3headGAN_eph49_G_A.pth'))
netG_A2 = UNet_3head(4,1)
netG_A2.load_state_dict(torch.load('../ckpt/i2i_1_UNet3headGAN_eph49_G_A.pth'))

#initialize the current GAN
netG_A3 = UNet_3head(4,1)
netD_A = NLayerDiscriminator(1, n_layers=4)

#train the cascaded framework
list_netG_A, list_netD_A = train_uncorr2CT_Cas_UNet3headGAN(
    [netG_A1, netG_A2, netG_A3], [netD_A],
    train_loader, test_loader,
    dtype=torch.cuda.FloatTensor,
    device='cuda',
    num_epochs=50,
    init_lr=1e-5,
    ckpt_path='../ckpt/i2i_2_UNet3headGAN',
)
```

### Bibtex
If you find the bits from this project helpful, please cite the following works:
```
@inproceedings{upadhyay2021uncerguidedi2i,
  title={Uncertainty Guided Progressive GANs for Medical Image Translation},
  author={Upadhyay, Uddeshya and Chen, Yanbei and Hebb, Tobias and Gatidis, Sergios and Akata, Zeynep},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2021},
  organization={Springer}
}
```
and
```
@article{upadhyay2021uncertainty,
  title={Uncertainty-aware Generalized Adaptive CycleGAN},
  author={Upadhyay, Uddeshya and Chen, Yanbei and Akata, Zeynep},
  journal={arXiv preprint arXiv:2102.11747},
  year={2021}
}
```