# LifelongReID
Offical implementation of our Lifelong Person Re-Identification via Adaptive Knowledge Accumulation in CVPR2021.

Code are coming soon.

## Citation
```
@InProceedings{pucvpr2021,
author = {Pu, Nan and Chen, Wei and Liu, Yu and Bakker, Erwin M. and Lew, Michael S.},
title = {Lifelong Person Re-Identification via Adaptive Knowledge Accumulation},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```
# Install
## Enviornment
conda create -n lreid python=3.7

conda activate lreid

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch

conda install opencv

pip install Cython sklearn numpy prettytable easydict tqdm visdom matplotlib

If you want to use fp16, please follow https://github.com/NVIDIA/apex to install apex, which is just a optional pakage.

The following codes work in our enviroment, but it could not work on other enviroment.

git clone https://github.com/NVIDIA/apex

cd apex

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

## lreid toolbox
Then, you could clone our project and install lreid

git clone https://github.com/TPCD/LifelongReID

cd LifelongReID

python setup.py develop



# Acknowledgement
The code is based on the PyTorch implementation of the [Torchreid](https://github.com/KaiyangZhou/deep-person-reid]) and [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch).
