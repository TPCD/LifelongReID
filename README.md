# LifelongReID
Offical implementation of our [Lifelong Person Re-Identification via Adaptive Knowledge Accumulation](https://arxiv.org/abs/2103.12462) in CVPR2021 
by [Nan Pu](https://tpcd.github.io/), Wei Chen, [Yu Liu](https://visionyuliu.github.io/), [Erwin M. Bakker](https://www.universiteitleiden.nl/en/staffmembers/erwin-bakker/publications#tab-4) and [Michael S. Lew](http://liacs.leidenuniv.nl/~lewms/).

We provide a lifelong person reid toolbox [lreid](https://github.com/TPCD/LifelongReID) in this repo. 

More details please see our paper.

![Framework](docs/aka.png)
## Citation
```
@InProceedings{pu_cvpr2021,
author = {Pu, Nan and Chen, Wei and Liu, Yu and Bakker, Erwin M. and Lew, Michael S.},
title = {Lifelong Person Re-Identification via Adaptive Knowledge Accumulation},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2021}
}
```
# Install
## Enviornment
```bash
conda create -n lreid python=3.7
conda activate lreid
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
conda install opencv
pip install Cython sklearn numpy prettytable easydict tqdm matplotlib
```
For visualization, you might need to install visdom:
```bash
pip install visdom
```

If you want to use fp16, please follow https://github.com/NVIDIA/apex to install apex, which is just a optional pakage.
The following codes work in our enviroment, but it could not work on other enviroment.
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
## lreid toolbox
Then, you could clone our project and install lreid
```bash
git clone https://github.com/TPCD/LifelongReID
cd LifelongReID
python setup.py develop
```

## Dataset prepration
Please follow [Torchreid_Dataset_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) to download datasets and unzip them to your data path (we refer to 'machine_dataset_path' in train_test.py). Alternatively, you could download some of unseen-domain datasets in [DualNorm](https://github.com/BJTUJia/person_reID_DualNorm).

## Train & Test

python train_test.py

# Acknowledgement
The code is based on the PyTorch implementation of the [Torchreid](https://github.com/KaiyangZhou/deep-person-reid]) and [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch).
