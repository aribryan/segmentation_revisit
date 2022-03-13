# Inference for files marked with only _03.png
import numpy as np
import torch
import argparse
import os,glob
import cv2 as cv
import time
from PIL import Image
import pdb

from model import simpleUNet
from utils import transform

def run_inference(args,model,rgb_file, mask_file,uid,device,transform=None,debug=False):


    with torch.no_grad():

        rgb = np.array(Image.open(rgb_file))

        image = torch.from_numpy(np.asarray(rgb) / 255).float()
        data_input = image.permute(2, 1, 0).unsqueeze(0).cuda().to(device)  # CHW
        if transform!=None:
            transformed = transform(image=np.asarray(rgb))
            data_input = transformed['image'].unsqueeze(0).cuda().to(device)
        prediction = model(data_input)
        binary_mask = ((torch.sigmoid(prediction)).float() > 0.5).float()

    # save results in a nice folder

    pred_cv = binary_mask.squeeze(0).permute(1,2,0).cpu().numpy()
    gt_mask = np.array(Image.open(mask_file))

    cv.imwrite(args.image_folder + 'prediction_'+ str(uid)+'.png',pred_cv*255)
    cv.imwrite(args.image_folder + 'gt_' + str(uid) + '.png', gt_mask)
    cv.imwrite(args.image_folder + 'gb_' + str(uid) + '.png', rgb)

    ##### count contours ###########
    contours, hierarchy = cv.findContours(pred_cv.astype(np.uint8).squeeze(2), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_gt, hierarchy_gt = cv.findContours(gt_mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #cv.drawContours(pred_cv.astype(np.uint8)*255, contours, -1, (0, 255, 0), 3)
    #cv.imwrite(args.image_folder + 'prediction_cont_' + str(uid) + '.png', pred_cv * 255)
    print("predicted number of contours ",len(contours)," for uid ",uid)
    #cv.drawContours(np.expand_dims(gt_mask.astype(np.uint8)*255,axis=2), contours_gt, -1, (0, 0, 255), 3)
    #cv.imwrite(args.image_folder + 'gt_cont_' + str(uid) + '.png', pred_cv * 255)
    print("GT number of contours ", len(contours_gt), " for uid ", uid)



def model_loader(args):
    # program to run all inference
    device = torch.cuda.set_device(args.gpu_id)

    model = simpleUNet(args).cuda().to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    transformed = transform(mode='val')

    print('KEYS matched, model loaded -- ')

    # read directory and dump image by image
    rgb_file_list_val = glob.glob(args.root + "/tissue_images/*03.tif")
    uid = 0.
    # test data dump
    for rgb_filename in rgb_file_list_val:
        mask_file = rgb_filename.replace('tissue_images','mask binary').replace('.tif','.png')
        run_inference(args,model, rgb_file=rgb_filename, mask_file=mask_file,uid=uid,device=device,transform=transformed)
        uid += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        help="dataset root folder",
        default="./data/",
    )

    parser.add_argument(
        "--ckpt",
        help="address of the folder to dump checkpoint",
        type=str,
        default='./log/20220313200909/parameter5_dice0'
    )

    parser.add_argument(
        "--image_folder",
        help="where the result image should be dumped",
        type=str,
        default='./results/'
    )
    parser.add_argument('--dimensions', default=64, type=int, help='channel expansion in Unet')
    parser.add_argument('--gpu_id', default=0, type=int, help='specify gpu id in cluster')

    args = parser.parse_args()

    model_loader(args)

    print('done')