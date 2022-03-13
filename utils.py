
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pdb


def accuracy(pred, gt,threshold=0.5):

    predictions = (torch.sigmoid(pred)).float()>threshold
    num_correct = (predictions == gt).sum()
    num_pixels = torch.numel(pred)
    accuracy = (num_correct/num_pixels)
    # black pixel outputs 80 percent, so get a better metric like IoU

    return accuracy

def dice(pred,gt, threshold=0.5):

    predictions = (torch.sigmoid(pred)).float() > threshold
    dice = ((2*predictions.float()*gt).sum())/((predictions.float()+gt).sum() + 1e-6)

    return dice

def image_logger(logdir,image, prediction, gt, count,threshold=0.5):

    writer = SummaryWriter(logdir)
    # now extract two inputs and upsample it

    rgb_image = image
    predictions = ((torch.sigmoid(prediction)).float() > threshold).float()
    gt_mask = gt

    img_grid = torchvision.utils.make_grid(torch.cat([rgb_image, predictions.repeat(1,3,1,1), gt_mask.repeat(1,3,1,1)], dim=3), nrow=1)
    # write to tensorboard
    writer.add_image('val/Ref_Prediction_GroundTruth', img_grid,global_step=count)

def scalar_logger(logdir,mode,loss=None,acc=None,dice=None,count=0):

    writer = SummaryWriter(logdir)
    if mode == 'train':
        writer.add_scalar("train/loss", loss, count)
    else:
        writer.add_scalar("val/accuracy", acc, count)
        writer.add_scalar("val/dice_score", dice, count)

def transform(mode='train'):

    if mode=='train' :
        transform = A.Compose([A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.5),
                                 A.RandomBrightnessContrast(brightness_limit=0.3,contrast_limit=0.3,p=0.5),
                                 A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),ToTensorV2(),])
    else:
        transform = A.Compose([A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),ToTensorV2(),])

    return transform

def ckpt_loader():
    pass