Assuming you are using a single GPU 0, for multi-level projector resnet101, run the command ``CUDA_VISIBLE_DEVICES=0 python3 train.py -m mt_prj_resnet101``. For single-level projector resnet101, run the command ``CUDA_VISIBLE_DEVICES=0 python3 train.py -m sg_prj_resnet101``. For baseline resnet101, run the command ``CUDA_VISIBLE_DEVICES=0 python3 train.py -m resnet101``. Similarly for other ResNet variants. Check the file **train.py** for more information.  
Please install **Python 3.8+**.  
[Link to download the dataset](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo). You just need the **images** folder only.  
Replace ``name = k[:]`` by ``name = k[7:]`` in **utils/auto_load_resume** if you resumes training a model which is trained on multiple GPUs.  
***Note***: in the model names, sg: single_level, mt: multi-level.

# Ipsilateral Mammogram Assessment under Multi-View Network

## Install
Run the command
``python install -r requirements.txt``
The code is built on Pytorch version 1.9.1.

## Training

To train with Resnet18, Early Fusion model type and Avarage aggregation, let run the command
``python train.py --model resnet18 --modeltype EF --aggregation avg``