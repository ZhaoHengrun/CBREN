# CBREN
CBREN: Convolutional Neural Networks for Constant Bit Rate Video Quality Enhancement, IEEE TCSVT.    <br/>
Pytorch implementation of CBREN. <br/>

Note: different from the paper, this code adds residual blocks to the pixel-domain branch of DRM module, but it has little impact on the effect of the network.     <br/>

Because the DCN compilation in Windows environment may cause problems, this code may only run in Linux environment.     <br/>
## Requirements
Python 3.8<br/>
PyTorch 1.6.0<br/>
Numpy 1.19.2<br/>
Pillow 7.2.0<br/>
OpenCV 4.4.0.44<br/>
## Prepare
### Build
Deformable convolution is used in this code from:
https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch <br/>
Run `sh make.sh` to compile the deformable convolution. 
If there are compilation errors, delete the 'build/' directory before recompiling.  <br/>
### Datasets
The directories used in the project need to be created manually. <br/>
Download HEVC standard test sequence: https://pan.baidu.com/s/1m0jZfkhX_cjaoFrHlMp0Xg Extraction code: 88n9 <br/>
Hm16.0 is used to compress the standard test sequence.   <br/>
The original video and compressed video in YUV format are converted to MP4 format by ffmpeg.  <br/>
We provided the BasketballPass video in MP4 format as a demonstration in this project.  <br/>
Run `tools/video_get_frames.py` to obtain the image sequence in PNG format from the video.   <br/>
### Models
We provide models that have been trained: https://pan.baidu.com/s/1sszHgZ1tYVEu8toyUkFaUw Extraction code: i0zs<br/>

## Run
Run`python run.py`, and the generated images are saved in `results/`.	<br/>
If the size of GPU memory is not large enough to run the sequences A and B, please run `run_group_A&B.py`.	<br/>
