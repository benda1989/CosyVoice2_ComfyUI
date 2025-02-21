# CosyVoice2 for ComfyUI
A plugin of ComfyUI for CosyVoice2, one component for text to Sonic Video 
## install plugin
```sh
git clone https://github.com/benda1989/CosyVoice2_ComfyUI.git
```
## Install dependency packages
```sh
cd CosyVoice2_ComfyUI
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel 
```
## copy models
By default project will download CosyVoice2-0.5B and CosyVoice-ttsfrd into pretrained_models,
you can copy it if you downloaded before