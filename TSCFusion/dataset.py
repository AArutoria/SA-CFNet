import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import kornia.utils as KU
import glob
import math
from utils.utils import affine
import torch.nn.functional as F


def add_padding(image_tensor, padding_size=50, padding_value=0):
    """
    在图像周围添加padding

    参数:
    - image_tensor: 形状为 [B, 1, H, W] 的图像 tensor
    - padding_size: 每边添加的 padding 大小
    - padding_value: 填充值，默认为0 (黑色)

    返回:
    - 带有 padding 的图像 tensor
    """
    return F.pad(image_tensor, (padding_size, padding_size, padding_size, padding_size), mode='constant',
                 value=padding_value)
class TrainData(Dataset):                                      

    def __init__(self, opt):
        super(TrainData, self).__init__()

        self.files_rgb = sorted(glob.glob("%s/rgb/*" % opt.train_data_path))
        self.files_sar = sorted(glob.glob("%s/sar/*" % opt.train_data_path))
       
        self.opt = opt
        self.affine = affine

    def __getitem__(self, index):

        item_rgb = imread(self.files_rgb[index % len(self.files_rgb)]) #[1,1,256,256]
        item_sar = imread(self.files_sar[index % len(self.files_sar)])

        random_numbers = torch.rand(8).numpy() * 2 - 1
        # rgb_warp, gt_tp, flow = self.affine(item_rgb)
        #item_rgb_warp[1,256,256] gt_tp[2,3] flow[1,256,256,2]
        item_rgb_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_rgb],padding_modes=['zeros'], opt=self.opt)
        item_sar_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_sar],padding_modes=['zeros'], opt=self.opt)

        item_sar = item_sar.squeeze(0) #[1,256,256]
        item_rgb = item_rgb.squeeze(0)

        item_sar = add_padding(item_sar, padding_size=60)
        item_rgb = add_padding(item_rgb, padding_size=60)
        item_sar_warp = add_padding(item_sar_warp, padding_size=60)
        item_rgb_warp = add_padding(item_rgb_warp, padding_size=60)




        return item_rgb, item_sar, item_rgb_warp, item_sar_warp, gt_tp, flow

    def __len__(self):
        return len(self.files_rgb)


class TestData(Dataset):
    def __init__(self, opt):
    
        self.files_rgb = sorted(glob.glob("%s/rgb/*" % opt.test_data_path))
        self.files_sar = sorted(glob.glob("%s/sar/*" % opt.test_data_path))

        self.names_rgb = [os.path.basename(file) for file in self.files_rgb]
        self.names_sar = [os.path.basename(file) for file in self.files_sar]

        self.opt = opt
        self.affine = affine

    def __getitem__(self, index):
  
        item_rgb = imread(self.files_rgb[index % len(self.files_rgb)]) #[b,1,256,256]
        item_sar = imread(self.files_sar[index % len(self.files_sar)])#[B,1,256,256]0

        random_numbers = torch.rand(8).numpy() * 2 - 1 #8
        #item_rgb_warp [1,256,256] gt_tp[2,3] flow[B,256,256,2]
        item_rgb_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_rgb], padding_modes=['zeros'], opt=self.opt)
        item_sar_warp, gt_tp, flow = self.affine(random_numbers=random_numbers, imgs=[item_sar], padding_modes=['zeros'], opt=self.opt)

        item_sar = item_sar.squeeze(0)
        item_rgb = item_rgb.squeeze(0)


        item_sar = add_padding(item_sar, padding_size=60)
        item_rgb = add_padding(item_rgb, padding_size=60)
        item_sar_warp = add_padding(item_sar_warp, padding_size=60)
        item_rgb_warp = add_padding(item_rgb_warp, padding_size=60)
           
        return item_rgb, item_sar, item_rgb_warp, item_sar_warp, gt_tp, flow

    def __len__(self):
        return len(self.files_rgb)



def imread(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im_ts = (KU.image_to_tensor(img)/255.).float()
    im_ts = im_ts.unsqueeze(0)
    return im_ts


def img_save(img, filename):
    img = img.squeeze().cpu()
    img = KU.tensor_to_image(img*255)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))