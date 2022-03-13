import torch
import numpy as np
import argparse

from model import simpleUNet
from dataloader import BinSegData

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from tqdm import tqdm
import logging
from datetime import datetime
import os

from utils import accuracy,dice, image_logger, scalar_logger, transform
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dimensions',default=32, type=int,help='channel expansion in Unet')  # odd
parser.add_argument('--gpu_id', type=int, default=0, help='gpu number in a cluster')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int,help='training batch size')
parser.add_argument('--log', default="./log",help='log dump')
parser.add_argument('--epoch', default=50, type=int, help='max epoch')
parser.add_argument('--root', default="./data/", type=str,help='data root directory')

args = parser.parse_args()

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s' % (args.log, s)
if not os.path.exists(result_root): os.mkdir(result_root)

logging.basicConfig(filename='%s/train.log' % args.log, format='%(asctime)s %(message)s', level=logging.INFO)
device = torch.cuda.set_device(args.gpu_id)


model = simpleUNet(args).cuda().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.2)
model.train()

# no need to transform as tensor is made already
#data_transform = transforms.Compose([transforms.ToTensor()])

train_data = BinSegData(root=args.root, mode='train', transform=transform('train'))
val_data = BinSegData(root=args.root, mode='val', transform=transform('val'))

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)



@torch.no_grad()
def validate(model):

    model.eval()

    image, gt_mask, pred = None, None, None

    acc_score = np.zeros(len(val_dataloader))
    dice_score = np.zeros(len(val_dataloader))

    t = tqdm(iter(val_dataloader), leave=True, total=len(val_dataloader))
    for idx, data in enumerate(t):

        image, gt_mask = data['image'].cuda().to(device), data['mask'].cuda().to(device)
        pred = model(image)

        acc_score[idx] = accuracy(pred,gt_mask)*100
        dice_score[idx] = dice(pred,gt_mask)



        #del pred

        t.set_description('[validate] accuracy: %f' % acc_score[:idx + 1].mean())
        t.set_description('[validate] dice score: %f' % dice_score[:idx + 1].mean())

        t.refresh()

    return acc_score,dice_score,image,gt_mask,pred


max_epoch = args.epoch
for epoch in range(max_epoch):

    model.train()
    running_loss = 0.0
    log_loss = 0.0

    t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        #scheduler.step()

        image, gt_mask = data['image'].cuda().to(device), data['mask'].cuda().to(device)
        pred = model(image)

        loss = criterion(pred, gt_mask)


        del pred

        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        log_loss += loss.data.item()

        if idx % 50 == 0:
            running_loss /= 50
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch + 1, running_loss))
            t.refresh()

    scalar_logger(logdir=result_root, mode='train', loss=log_loss, count=epoch + 1)

    accuracy_, dice_,img_val,gt_val,pred_val = validate(model)
    image_logger(logdir=result_root,image=img_val, prediction=pred_val, gt=gt_val, count=epoch+1)
    scalar_logger(logdir=result_root, mode='val', acc=accuracy_.mean(),dice=dice_.mean(), count=epoch+1)
    logging.info('epoch:%d accuracy:%f' % (epoch + 1, accuracy_.mean()))
    logging.info('epoch:%d dice:%f' % (epoch + 1, dice_.mean()))


    # save checkpoints

    EPOCH = epoch+1
    LOSS = dice_.mean()
    PATH = "%s/parameter%d"%(result_root, epoch+1)
    torch.save(model.state_dict(), "%s/parameter%d_dice%d" % (result_root, epoch + 1,LOSS))
