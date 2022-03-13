# A Brief revisit to Medical Image segmentation
A simple revisit to segmentation with some tissue images
##### Table of Contents  
[Installations](#headers)

[Dataset preparation](#headers) 

[Training config](#Headers)  

[Inference](#Headers)

[Results](#Headers)

[Discussion and TODOs](#Headers)

...snip...   
<a name="Installations"/>
## Installing packages
Apart from the usual 'Python3.8+Pytorch1.5+cu11.0' we need two more packages. One for patchifying, please install `pip install patchify`. Additionally install albumination package by `pip install albuminations`. Both of these packages will be used to create our dataset. 
<a name="Dataset preparation"/>
## Data preparation
The original data consists of 27 images with resolutions 512x512. Since it is almost impossible to train a simple U-net as it would underfit such an arrangement, we will break the images into patches with the help of patchify package. We will dump the 128x128 resolution images in the `data` folder. Also, unzip the original dataset inside this folder. The rest should be taken care of. 
1. Please run `python3 preprocess.py`.
Once patchify has dumped the corresponding images, we will apply some basic transformation with the help of albumination package. See `ùtils.transform` and you will get an idea of the standard transformations we will use. With stride value of 20 pixels, we will use the following config:
We use horizontal, vertical and brightness contrast features from the transformation package.

Train Images|Validation Images|
--- |--- |
7410 |3510 |

The standard protocol used here is that we will use all the images and patchify them except the images marked with labels `*03.tif` (9 images out of 27+ images).
<a name="Training config"/>
## Training information
We have selected a very simple U-net. It is slightly different than the original U-net paper, please have a look at `model.py`. 

2. Please run the training file `python3 train.py --gpu_id <int> --lr 2e-5 --dimensions 64`

With 50 epochs, you probably have to wait for 3 hours with one-gpu (RTX3090). 

3. Optional: You can look at the statistics by running `tensorboard --logdir=.`

We are training the network with `BCEwithLogits` loss as it is numerically stable. Additionally, we will look at accuracy and dice score. We will pick up a best model from the best dice scores. Some visualisations from the tensorboard are provided. 

Train Loss|Validation Dice Loss| Validation mask |
--- |--- |--- |
![alt text](https://github.com/aribryan/segmentation_revisit/blob/main/images/training_segmentation.png) |![alt text](https://github.com/aribryan/segmentation_revisit/blob/main/images/val_dice.png)|![alt text](https://github.com/aribryan/segmentation_revisit/blob/main/images/tensorboard_crop.png)|

## Run inference to see masks
After you are done with training, we will move towards visualisation of the full resolution images. Since we already don't have enough dataset, we will take the full resolution version of the validation images. Additionally, we will try to count the contours as well. We will use OpenCV's `findContour` functionality.

IMPORTANT: download the pretrained checkpoint from here, and put it inside the directory `log/ckpt/`.

4. After the pre-trained checkpoints have been placed, along with the validation images, please run the inference code by running `python3 inference.py --gpu_id <int> --ckpt <'./log/ckpt/parametersxxxx'> --image_folder ./results/`

## Let's see the results
The data has been prepared with Patchify package. 

## Discussions
On an initial training iteration, the network has managed to do fairly well. 
