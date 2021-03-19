# LifelongReID
offical implement of our Lifelong Person Re-Identification via Adaptive Knowledge Accumulation in CVPR2021.





# Install

conda create -n lreid python=3.7
conda activate lreid
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
conda install opencv
pip install Cython sklearn numpy prettytable easydict tqdm visdom matplotlib
If you want to use fp16, please follow https://github.com/NVIDIA/apex to install apex, which is just a optional pakage.
Then, you could clone our project and install lreid
clone 
python setup.py develop
