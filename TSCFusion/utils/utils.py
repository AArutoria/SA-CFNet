import time
import datetime
import sys
import numpy as np
import torch
import torch.nn.functional as F
import math
from utils.elastic_transform import ElasticTransform
import random
import cv2
import torch.nn as nn
from visdom import Visdom
from torch.autograd import Variable
from torch import Tensor
import kornia
import pathlib

import torch


def remove_padding(image_tensor, original_size=(256, 256)):
    """
    从 [B, C, 356, 356] 的图像中删除全0填充，恢复成 [B, C, 256, 256] 的图像。

    参数:
    - image_tensor: 形状为 [B, C, 356, 356] 的图像张量
    - original_size: 恢复后的图像大小 (H, W)

    返回:
    - 恢复后的图像张量
    """
    _, _, padded_height, padded_width = image_tensor.shape
    target_height, target_width = original_size

    # 计算裁剪起点
    start_h = (padded_height - target_height) // 2
    start_w = (padded_width - target_width) // 2

    # 裁剪图像
    cropped_image_tensor = image_tensor[:, :, start_h:start_h + target_height, start_w:start_w + target_width]

    return cropped_image_tensor


# 示例图像张量
batch_images = torch.randn(4, 3, 356, 356)  # 假设图像大小为 356x356

# 删除填充恢复到原始大小
cropped_images = remove_padding(batch_images, original_size=(256, 256))



def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(p), im_cv)

def affine(random_numbers, imgs, padding_modes, opt):
    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]
    if not isinstance(padding_modes, list) and not isinstance(padding_modes, tuple):
        padding_modes = [padding_modes]

    if opt.dim == 3:
        tmp = np.ones(3)
        tmp[0:3] = random_numbers[0:3]
        scaling = tmp * opt.scaling + 1
        tmp[0:3] = random_numbers[3:6]
        rotation = tmp * opt.rotation
        tmp[0:2] = random_numbers[6:8]
        tmp[2] = 0
        translation = tmp * opt.translation
    else:
        scaling = random_numbers[0:2] * opt.scaling + 1
        rotation = random_numbers[2] * opt.rotation
        translation = random_numbers[3] * opt.translation

    theta = create_affine_transformation_matrix(
        n_dims=opt.dim, scaling=scaling, rotation=rotation, shearing=None, translation=translation)

    # 计算扭曲后的四角点
    # four_corners = np.array([[0, 0], [0, 255], [255, 255], [255, 0]], dtype=np.float32).reshape(-1, 1, 2)
    # T = np.array([[2 / 256, 0, -1],
    #               [0, 2 / 256, -1],
    #               [0, 0, 1]])
    # matrix_warp = np.linalg.inv(T) @ np.linalg.inv(theta) @ T
    # new_four_point = cv2.transform(four_corners, matrix_warp[:-1,:])
    # new_four_point = torch.from_numpy(new_four_point).to(torch.float32)

    # 计算配准gt_tp
    matrix = np.linalg.inv(theta)
    gt_tp = matrix[:-1, :]
    gt_tp1 = torch.from_numpy(gt_tp).to(torch.float32)
    size = imgs[0].size()
    gt_flow = F.affine_grid(gt_tp1.unsqueeze(0), size, align_corners=True)


    theta = theta[:-1, :]
    theta = torch.from_numpy(theta).to(torch.float32)
    size = imgs[0].size()
    grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True)

    res_img = []
    for img, mode in zip(imgs, padding_modes):
        res_img.append(F.grid_sample(img, grid, align_corners=True, padding_mode=mode).squeeze(0))

    return res_img[0] if len(res_img) == 1 else res_img, gt_tp, gt_flow

# def affine(img):
#     elastic = ElasticTransform(kernel_size=101, sigma=32)
#     ir, elastic_disp = elastic(img)
#
#
#     scale = round(random.uniform(0.95, 1.15), 1)
#     translate_x = round(random.uniform(-0.05, 0.05), 3)
#     translate_y = round(random.uniform(-0.05, 0.05), 3)
#     rotate_angle = round(random.uniform(-math.pi / 6, math.pi / 6), 2)
#
#     cos_angle = math.cos(rotate_angle)
#     sin_angle = math.sin(rotate_angle)
#
#     theta = torch.tensor([[
#         [scale * cos_angle, -scale * sin_angle, translate_x],
#         [scale * sin_angle, scale * cos_angle, translate_y]
#     ]], dtype=torch.float)
#     rotata = torch.tensor([[
#         [scale * cos_angle, -scale * sin_angle],
#         [scale * sin_angle, scale * cos_angle]
#     ]], dtype=torch.float)
#     translate = torch.tensor([[
#         [translate_x],
#         [translate_y]
#     ]], dtype=torch.float)
#
#     a = torch.tensor([[[0, 0, 1]]]).float()
#     mat = torch.cat((theta, a), dim=1)
#     mat_inv = torch.inverse(mat) #[1,3,3]
#     inv_aff = mat_inv[:, :-1, :] #[1,2,3]
#
#     affine_grid = F.affine_grid(theta, size=[1, 1,256, 256], align_corners=False)
#     transformed_image = F.grid_sample(ir, affine_grid, align_corners=False)
#
#     grid_inv = F.affine_grid(inv_aff, size=[1,1, 256, 256], align_corners=False) #[1,256,256,2]
#     reg_image = F.grid_sample(transformed_image, grid_inv, align_corners=False)
#     transformed_image = transformed_image.squeeze(0)
#     img = img.squeeze(0)
#     rotata = rotata.squeeze(0)
#     translate = translate.squeeze(0)
#
#     disp = affine_grid + elastic_disp
#
#     return transformed_image, img,rotata,translate,disp


def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """
        create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    trans_scaling = np.eye(n_dims + 1)
    trans_shearing = np.eye(n_dims + 1)
    trans_translation = np.eye(n_dims + 1)

    if scaling is not None:
        trans_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        trans_shearing[shearing_index] = shearing

    if translation is not None:
        trans_translation[np.arange(n_dims), n_dims *
                          np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot = np.eye(n_dims + 1)
        trans_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation), np.sin(rotation),
                                                                     np.sin(rotation) * -1, np.cos(rotation)]
        return trans_translation @ trans_rot @ trans_shearing @ trans_scaling

    else:
        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        trans_rot1 = np.eye(n_dims + 1)
        trans_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]),
                                                                      np.sin(
                                                                          rotation[0]) * -1,
                                                                      np.cos(rotation[0])]
        trans_rot2 = np.eye(n_dims + 1)
        trans_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]),
                                                                      np.sin(
                                                                          rotation[1]) * -1,
                                                                      np.sin(
                                                                          rotation[1]),
                                                                      np.cos(rotation[1])]
        trans_rot3 = np.eye(n_dims + 1)
        trans_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]),
                                                                      np.sin(
                                                                          rotation[2]) * -1,
                                                                      np.cos(rotation[2])]
        return trans_translation @ trans_rot3 @ trans_rot2 @ trans_rot1 @ trans_shearing @ trans_scaling


def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(p), im_cv)