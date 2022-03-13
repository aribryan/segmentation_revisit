# Function for dumping image patches for segmentation training
# We will use patchify sampling and make sure the masks

import numpy as np
import argparse
import os,glob
import cv2 as cv
import time
from patchify import patchify
from PIL import Image
import pdb


def dump_image_patches(args,mode='train',debug=False,rgb_file=None,mask_file=None):

    # read files, directory , sort
    if debug:
        # this is just a random example
        image_rgb = np.array(Image.open(args.root + 'tissue_images/Human_AdrenalGland_02.tif'))
        image_mask = np.array(Image.open(args.root  +'mask binary/Human_AdrenalGland_02.png'))
    else:
        image_rgb = np.array(Image.open(rgb_file))  # np.random.rand(512,512,3)
        image_mask = np.array(Image.open(mask_file))

    rgb_patches = patchify(image_rgb,(args.patch_dim,args.patch_dim,3),step=args.patch_stride)
    mask_patches = patchify(image_mask,(args.patch_dim,args.patch_dim),step=args.patch_stride)

    directory_rgb = args.root + '/patches_rgb_' + mode
    directory_mask = args.root + '/patches_mask_' + mode
    #uid = float(time.time())

    #pdb.set_trace()

    uid = rgb_file.replace(args.root + '/tissue_images/','').replace('.tif','_patch_')

    if not os.path.exists(directory_rgb):
        os.makedirs(directory_rgb)
        os.makedirs(directory_mask)

    for i in range(rgb_patches.shape[0]):
        for j in range(rgb_patches.shape[0]):


            rgb_patch = rgb_patches[i,j,0,:,:,:]
            mask_patch = mask_patches[i,j,:,:]
            # save patches
            cv.imwrite(directory_rgb + '/' + uid + str(i) + str(j) + '.png',rgb_patch)
            cv.imwrite(directory_mask + '/' + uid + str(i) + str(j) + '.png', mask_patch)


def loader(args):
    # read directory and dump image by image
    # rest are training

    rgb_file_list_all = glob.glob(args.root + "/tissue_images/*.tif")
    rgb_file_list_val = glob.glob(args.root + "/tissue_images/*03.tif")
    rgb_file_list_train = [x for x in rgb_file_list_all if x not in rgb_file_list_val]
    # train data dump
    for rgb_filename in rgb_file_list_train:
        mask_file = rgb_filename.replace('tissue_images','mask binary').replace('.tif','.png')
        dump_image_patches(args,  mode='train', rgb_file=rgb_filename, mask_file=mask_file)
    # test data dump
    for rgb_filename in rgb_file_list_val:
        mask_file = rgb_filename.replace('tissue_images','mask binary').replace('.tif','.png')
        dump_image_patches(args,  mode='val', rgb_file=rgb_filename, mask_file=mask_file)

def testing(args,debug=True):
    # make sure the mask and the images are the same
    # after sampling
    dump_image_patches(args,debug=debug)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        help="dataset root folder",
        default="./data/",
    )

    parser.add_argument(
        "--patch_dim",
        help="dimension of the patches to be sampled in int",
        type=int,
        default=128
    )

    parser.add_argument(
        "--patch_stride",
        help="strides for the patch creation",
        type=int,
        default=20
    )

    args = parser.parse_args()

    loader(args)

    print('done')