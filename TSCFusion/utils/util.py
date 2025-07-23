#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
import kornia
import kornia.utils as KU
from PIL import Image


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def rotate_translate_image(image, rotation_matrices, translation_vectors):
    """
    对图像进行旋转和平移操作，支持批量操作。

    :param image: 输入图像，形状为 (B, C, H, W)。
    :param rotation_matrices: 形状为 (B, 2, 2) 的批量旋转矩阵。
    :param translation_vectors: 形状为 (B, 2, 1) 的批量平移向量。
    :return: 变换后的图像。
    """
    B, C, H, W = image.size()

    # 确保 rotation_matrices 和 translation_vectors 是 torch 张量
    rotation_matrices = torch.tensor(rotation_matrices, dtype=torch.float32)
    translation_vectors = torch.tensor(translation_vectors, dtype=torch.float32)

    # 创建 2x3 的仿射变换矩阵
    # print(translation_vectors.shape)
    # print(rotation_matrices.shape)
    affine_matrices = torch.cat([rotation_matrices, translation_vectors], dim=2)

    # 创建仿射变换网格
    grid = F.affine_grid(affine_matrices, image.size(), align_corners=False)

    # 应用变换
    transformed_image = F.grid_sample(image, grid, align_corners=False)

    return transformed_image


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def affine_to_flow(tp, b):
    tp = tp.reshape(-1, 2, 3)
    a = torch.Tensor([[[0, 0, 1]]]).cuda().repeat(b, 1, 1)
    tp = torch.cat((tp, a), dim=1)
    grid = KU.create_meshgrid(256, 256).cuda().repeat(b, 1, 1, 1)
    flow = kornia.geometry.linalg.transform_points(tp, grid)
    return flow, flow-grid

def STN(img, pre_tps): #IMG[8,1,256,256] pre_tps[8,6]
    aff_mat = pre_tps.reshape(-1, 2, 3) #[8,2,3]
    img_grid = F.affine_grid(aff_mat, img.size()).float() #[8,256,256,2]

    img_reg = F.grid_sample(img, img_grid, align_corners=True) #[8,1,256,256]
    return img_reg

class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).cuda()
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=True), flow


def save_batch_images(batch, output_dir, prefix='image', normalize=True):
    """
    Save images from a batch tensor to a specified directory.

    Parameters:
        batch (torch.Tensor): A batch of images with shape (batch_size, channels, height, width).
        output_dir (str): Directory where images will be saved.
        prefix (str): Prefix for the filenames.
        normalize (bool): Whether to normalize image values to [0, 255].
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensor to numpy array
    batch_np = batch.detach().cpu().numpy()  # Use detach() to remove gradients and convert to numpy array

    # Process each image in the batch
    for i in range(batch_np.shape[0]):
        image = batch_np[i]  # Get the i-th image

        # Handle grayscale and RGB images
        if image.shape[0] == 1:  # Grayscale image
            image = image.squeeze(0)  # Remove the channel dimension
        else:  # RGB image
            image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

        # Normalize the image to [0, 255]
        if normalize:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Convert to PIL Image and save
        image_pil = Image.fromarray(image)
        filename = os.path.join(output_dir, f'{prefix}_{i}.png')
        image_pil.save(filename)
        print(f'Saved {filename}')


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')