import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import glob
import argparse
from utils import transform

import pdb

class BinSegData(Dataset):

    def __init__(self, root,mode='train',transform=None):

        # Get image list
        self.image_list = glob.glob(root + '/patches_rgb_' + str(mode) + '/*.png')
        self.data_len = len(self.image_list)
        self.transform = transform

    def __getitem__(self, index):

        data = dict.fromkeys(['image','mask'])

        im_patch = Image.open(self.image_list[index])
        im_mask = Image.open((self.image_list[index]).replace('rgb','mask'))

        image = torch.from_numpy(np.asarray(im_patch)/255).float()
        mask = torch.from_numpy(np.asarray(im_mask)/255).float()

        data['image'] = image.permute(2,1,0) # CHW
        data['mask'] = mask.unsqueeze(2).permute(2,1,0) # CHW

        if self.transform is not None:

            transformed = self.transform(image=np.asarray(im_patch),mask=np.asarray(im_mask))
            data['image'] = transformed['image']
            data['mask'] = (transformed['mask'].float()/255).unsqueeze(0)


        return data

    def __len__(self):
        return self.data_len


def test(args):

    dataset = BinSegData(args.root,args.mode, transform=transform('train'))
    sample = dataset.__getitem__(10)

    #pdb.set_trace()

    # batch it
    dp = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=None, drop_last=False)
    # iterate and access one element to match size or display

    it = iter(dp)
    element = next(it) # access a batch element


    assert sample['image'].shape[1] == sample['mask'].shape[1], 'dimension match required'
    print('dataflow test passed')

    assert element['image'].shape[0] == element['mask'].shape[0], 'dimension match required'
    print('dataflow batch test passed')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        help="dataset root folder",
        default="./data/",
    )

    parser.add_argument(
        "--mode",
        help="train or val",
        type=str,
        default='val'
    )

    args = parser.parse_args()

    test(args)

    print('done')