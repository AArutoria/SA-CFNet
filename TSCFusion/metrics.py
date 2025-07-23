
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import cv2
import kornia
from sklearn.metrics.cluster import mutual_info_score

import math

import torch

import torch
import torch.nn as nn


# class MutualInformation(nn.Module):
#     def __init__(self, bins=256):
#         super(MutualInformation, self).__init__()
#         self.bins = bins  # 直方图的bin数目
#
#     def compute_joint_histogram(self, x, y):
#         """计算两个图像的联合直方图"""
#         batch_size = x.shape[0]
#         # 初始化联合直方图
#         joint_hist = torch.zeros((batch_size, self.bins, self.bins), dtype=torch.float32).cuda()
#
#         for b in range(batch_size):
#             x_flat = (x[b] * (self.bins - 1)).long().flatten()
#             y_flat = (y[b] * (self.bins - 1)).long().flatten()
#
#             for i in range(x_flat.size(0)):
#                 joint_hist[b, x_flat[i], y_flat[i]] += 1
#
#         # 归一化直方图
#         joint_hist = joint_hist / torch.sum(joint_hist, dim=(1, 2), keepdim=True)
#         return joint_hist
#
#     def compute_marginal_histogram(self, joint_hist):
#         """从联合直方图中计算边缘直方图"""
#         p_x = torch.sum(joint_hist, dim=2)  # p(x)
#         p_y = torch.sum(joint_hist, dim=1)  # p(y)
#         return p_x, p_y
#
#     def compute_mi(self, joint_hist, p_x, p_y):
#         """计算互信息"""
#         p_x = p_x.unsqueeze(2)  # shape: (batch_size, bins, 1)
#         p_y = p_y.unsqueeze(1)  # shape: (batch_size, 1, bins)
#
#         # 避免log(0)
#         joint_hist = torch.clamp(joint_hist, min=1e-10)
#         p_x = torch.clamp(p_x, min=1e-10)
#         p_y = torch.clamp(p_y, min=1e-10)
#
#         mi = joint_hist * (torch.log(joint_hist) - torch.log(p_x) - torch.log(p_y))
#         mi = torch.sum(mi, dim=(1, 2))  # 对所有bins求和
#         return mi
#
#     def forward(self, x, y):
#         """计算输入的两个图像的互信息"""
#         # 假设输入x和y都是[1, 1, 256, 256]的格式
#         x = x.view(x.shape[0], -1)  # 展平
#         y = y.view(y.shape[0], -1)
#
#         # 计算联合直方图
#         joint_hist = self.compute_joint_histogram(x, y)
#
#         # 计算边缘直方图
#         p_x, p_y = self.compute_marginal_histogram(joint_hist)
#
#         # 计算互信息
#         mi = self.compute_mi(joint_hist, p_x, p_y)
#         return mi


import torch
import torch.nn as nn


class MutualInformation(nn.Module):
    def __init__(self, bins=256):
        super(MutualInformation, self).__init__()
        self.bins = bins  # 设置直方图的bin数

    def compute_joint_histogram(self, x, y):
        """计算两个图像的联合直方图"""
        batch_size = x.shape[0]
        x_flat = (x * (self.bins - 1)).long().view(batch_size, -1)
        y_flat = (y * (self.bins - 1)).long().view(batch_size, -1)

        # 计算联合直方图
        joint_hist = torch.zeros(batch_size, self.bins, self.bins, dtype=torch.float32, device=x.device)
        for b in range(batch_size):
            indices = x_flat[b] * self.bins + y_flat[b]
            hist = torch.bincount(indices, minlength=self.bins * self.bins).float()
            joint_hist[b] = hist.view(self.bins, self.bins)

        # 归一化直方图
        joint_hist = joint_hist / joint_hist.sum(dim=(1, 2), keepdim=True)
        return joint_hist

    def compute_marginal_histogram(self, joint_hist):
        """从联合直方图中计算边缘直方图"""
        p_x = joint_hist.sum(dim=2)  # p(x)
        p_y = joint_hist.sum(dim=1)  # p(y)
        return p_x, p_y

    def compute_mi(self, joint_hist, p_x, p_y):
        """计算互信息"""
        p_x = p_x.unsqueeze(2)  # shape: (batch_size, bins, 1)
        p_y = p_y.unsqueeze(1)  # shape: (batch_size, 1, bins)

        # 避免log(0)
        joint_hist = torch.clamp(joint_hist, min=1e-10)
        p_x = torch.clamp(p_x, min=1e-10)
        p_y = torch.clamp(p_y, min=1e-10)

        mi = joint_hist * (torch.log(joint_hist) - torch.log(p_x) - torch.log(p_y))
        mi = mi.sum(dim=(1, 2))  # 对所有bins求和
        return mi

    def forward(self, x, y):
        """计算输入的两个图像的互信息"""
        # 假设输入x和y都是[1, 1, 256, 256]的格式
        x = x.view(x.shape[0], -1)  # 展平
        y = y.view(y.shape[0], -1)

        # 计算联合直方图
        joint_hist = self.compute_joint_histogram(x, y)

        # 计算边缘直方图
        p_x, p_y = self.compute_marginal_histogram(joint_hist)

        # 计算互信息
        mi = self.compute_mi(joint_hist, p_x, p_y)
        return mi


# # 使用示例
# x = torch.rand((1, 1, 256, 256)).cuda()  # 随机生成一个tensor图像
# y = torch.rand((1, 1, 256, 256)).cuda()  # 随机生成一个tensor图像
#
# mi_calculator = MutualInformation(bins=256).cuda()
# mi = mi_calculator(x, y)
# print(f"Mutual Information: {mi.item()}")


class LNCC(nn.Module):
    """
        Local (over window) normalized cross correlation.
    """
    def __init__(self):
        super(LNCC, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def forward(self, I, J, win=[17]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims

        sum_filt = torch.ones([1, 1, *win]).cuda()

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)

class NCC(nn.Module):
    """
        Normalized cross correlation.
    """
    def __init__(self):
        super(NCC, self).__init__()

    def similarity_loss(self, tgt, warped_img):
        sizes = np.prod(list(tgt.shape)[1:])
        flatten1 = torch.reshape(tgt, (-1, sizes))
        flatten2 = torch.reshape(warped_img, (-1, sizes))

        mean1 = torch.reshape(torch.mean(flatten1, dim=-1), (-1, 1))
        mean2 = torch.reshape(torch.mean(flatten2, dim=-1), (-1, 1))
        var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
        var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
        cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
        pearson_r = cov12 / torch.sqrt((var1 + 1e-5) * (var2 + 1e-6))
        # ncc_value = torch.sum(1 - pearson_r)
        ncc_value = torch.sum(pearson_r)
        return ncc_value

    def forward(self, y_true, y_pred):
        return self.similarity_loss(y_true, y_pred)



class MSE(nn.Module):
    """
    Mean squared error loss.
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class MAE(nn.Module):
    """
    Mean Absolute Error (MAE) loss.
    """
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class RMSE(nn.Module):
    """
    Root Mean Squared Error (RMSE) loss.
    """
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

class RMI(nn.Module):
    """
    PyTorch Module which calculates the Region Mutual Information loss (https://arxiv.org/abs/1910.12037).
    """

    def __init__(self,
                 with_logits=True,
                 radius=3,
                 bce_weight=0.5,
                 downsampling_method='avg',
                 stride=3,
                 use_log_trace=True,
                 use_double_precision=True,
                 epsilon=1e-5):
        """
        :param with_logits:
            If True, apply the sigmoid function to the prediction before calculating loss.
        :param radius:
            RMI radius.
        :param bce_weight:
            Weight of the binary cross entropy. Must be between 0 and 1.
        :param downsampling_method:
            Downsampling method used before calculating RMI. Must be one of ['avg', 'max', 'region-extraction'].
            If 'region-extraction', then downscaling is done during the region extraction phase. Meaning that the stride is the spacing between consecutive regions.
        :param stride:
            Stride used for downsampling.
        :param use_log_trace:
            Whether to calculate the log of the trace, instead of the log of the determinant. See equation (15).
        :param use_double_precision:
            Calculate the RMI using doubles in order to fix potential numerical issues.
        :param epsilon:
            Magnitude of the entries added to the diagonal of M in order to fix potential numerical issues.
        """
        super(RMI, self).__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def rmi_loss(self, input, target):
        """
        Calculates the RMI loss between the prediction and target.

        :return:
            RMI loss
        """
        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        # Get region vectors
        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        # Convert to doubles for better precision
        if self.use_double_precision:
            y = y.double()
            p = p.double()

        # Small diagonal matrix to fix numerical issues
        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        # Subtract mean
        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        # Covariances
        y_cov = y @ transpose(y)
        p_cov = p @ transpose(p)
        y_p_cov = y @ transpose(p)

        # Approximated posterior covariance matrix of Y given P
        m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)

        # Lower bound of RMI
        if self.use_log_trace:
            rmi = 0.5 * log_trace(m + eps)
        else:
            rmi = 0.5 * log_det(m + eps)

        # Normalize
        rmi = rmi / float(vector_size)

        # Sum over classes, mean over samples.
        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):
        """
        Downsamples and extracts square regions from x.
        Returns the flattened vectors of length radius*radius.
        """
        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == 'region-extraction' else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius ** 2, -1))
        return x_regions

    def downsample(self, x):
        # Skip if stride is 1
        if self.stride == 1:
            return x

        # Skip if we pool during region extraction.
        if self.downsampling_method == 'region-extraction':
            return x

        padding = self.stride // 2
        if self.downsampling_method == 'max':
            return F.max_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        if self.downsampling_method == 'avg':
            return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        raise ValueError(self.downsampling_method)

    def forward(self, target, input):
        # Calculate BCE if needed
        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        # Apply sigmoid to get probabilities. See final paragraph of section 4.
        if self.with_logits:
            input = torch.sigmoid(input)

        # Calculate RMI loss
        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

def transpose(x):
    return x.transpose(-2, -1)

def inverse(x):
    return torch.inverse(x)

def log_trace(x):
    x = torch.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)

def log_det(x):
    return torch.logdet(x)

def imread(path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False):
    im_cv = cv2.imread(str(path), flags)
    assert im_cv is not None, f"Image {str(path)} is invalid."
    im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    return im_ts.unsqueeze(0) if unsqueeze else im_ts


# def calc_img_metrics(mse_metric, ncc_metric, lncc_metric, rmi_metric, mi_metric,root_in, root_gt):
def calc_img_metrics(mse_metric, mae_metric, rmse_metric,root_in, root_gt):

    MSE_list = []
    MAE_list = []
    RMSE_list = []
    in_img_list = sorted(os.listdir(root_in))
    gt_img_list = sorted(os.listdir(root_gt))
    log_file_path = os.path.join(os.path.dirname(root_in), "metrics_log.txt")  # 保存日志文件到 root_in 的上一级目录

    for in_img, gt_img in zip(in_img_list, gt_img_list):
        in_img_path = os.path.join(root_in, in_img)
        gt_img_path = os.path.join(root_gt, gt_img)

        img_in = imread(in_img_path, unsqueeze=True).cuda()
        img_gt = imread(gt_img_path, unsqueeze=True).cuda()

        mse_value  = mse_metric(img_gt, img_in)
        mae_value  = mae_metric(img_gt, img_in)
        rmse_value = rmse_metric(img_gt, img_in)
        MSE_list.append(mse_value)
        MAE_list.append(mae_value)
        RMSE_list.append(rmse_value)

        # print("{} MSE = {:.5}, NCC = {:.5}, LNCC = {:.5}, RMI = {:.5}, MI = {:.5}".format(in_img, mse_value, ncc_value, lncc_value,rmi_value, mi_value.item()))
        # print("{} MSE = {:.5}, NCC = {:.5}, LNCC = {:.5}, RMI = {:.5}".format(in_img, mse_value, ncc_value, lncc_value,rmi_value))
    # log = 'Average MSE={:.5}, NCC={:.5}, LNCC={:.5}, RMI = {:.5}, MI = {:.5}'.format(sum(MSE_list)/len(MSE_list), sum(NCC_list)/len(NCC_list), sum(LNCC_list)/len(LNCC_list), sum(RMI_list)/len(RMI_list), (sum(MI_list)/len(MI_list)).item())
    log = 'Average MSE={:.5}, MAE={:.5}, RMSE={:.5}'.format(sum(MSE_list)/len(MSE_list), sum(MAE_list)/len(MAE_list), sum(RMSE_list)/len(RMSE_list))
    print(log)

    with open(log_file_path, "a") as log_file:
        log_file.write(log + "\n")

    return MSE_list, MAE_list, RMSE_list, log


if __name__ == '__main__':

    root_in = r'F:\one_layers\ir_reg'
    root_gt = r'H:\data\RoadScene\Test\vi'

    mse_metric  = MSE().cuda()
    rmse_metric = RMSE().cuda()
    mae_metric  = MAE().cuda()
    # rmi_metric  = RMI().cuda()
    # mi_metric = MutualInformation(bins=256).cuda()


    # TODO: Calculate Mse metric
    # MSE_list, NCC_list, LNCC_list, log = calc_img_metrics(mse_metric, ncc_metric, lncc_metric, rmi_metric,root_in, root_gt)
    MSE_list, NCC_list, LNCC_list, log = calc_img_metrics(mse_metric, mae_metric, rmse_metric,root_in,root_gt)
    # TODO: Calculate NCC metric
    # print(in_img_list)
    # print(gt_img_list)
