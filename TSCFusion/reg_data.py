import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform
from PIL import Image

# affine = AffineTransform(degrees=30, translate=0.1)
# elastic = ElasticTransform(kernel_size=101, sigma=16)
# class RegData(torch.utils.data.Dataset):
#     """
#     Load dataset with infrared folder path and visible folder path
#     """
#
#     # TODO: remove ground truth reference
#     def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, crop=lambda x: x):
#     # def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path):
#         super(RegData, self).__init__()
#         self.crop = crop
#         # gain infrared and visible images list
#         self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
#         self.it_list = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
#
#
#
#     def __getitem__(self, index):
#         # gain image path
#         ir_path = self.ir_list[index]
#         it_path = self.it_list[index]
#
#         assert ir_path.name == it_path.name, f"Mismatch ir:{ir_path.name} vi:{it_path.name}."
#
#         # read image as type Tensor
#         ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
#         it = self.imread(path=it_path, flags=cv2.IMREAD_GRAYSCALE)
#
#
#         # crop same patch
#         patch = torch.cat([ir, it], dim=0)
#         patch = torchvision.transforms.functional.to_pil_image(patch)
#         patch = self.crop(patch)
#         patch = torchvision.transforms.functional.to_tensor(patch)
#         ir, vi = torch.chunk(patch, 2, dim=0)
#
#         ir,vi,ir_warp, vi_warp,affine_param,disp = affine(ir,vi)
#         # warped1, warped2, -disp, affine_param
#         ir_elastic, ir_elastic_disp,vi_elastic, vi_elastic_disp = elastic(ir_affine)
#         ir_disp = ir_affine_disp + ir_elastic_disp
#         vi_disp = vi_affine_disp + vi_elastic_disp
#         ir_warp = ir_elastic
#         vi_warp = ir_elastic
#
#         ir_warp.detach_()
#         ir_disp.detach_()
#
#         # vi_affine, vi_affine_disp = affine(vi)
#         # vi_elastic, vi_elastic_disp = elastic(vi_affine)
#         # vi_disp = vi_affine_disp + vi_elastic_disp
#         # vi_warp = ir_elastic
#
#         vi_warp.detach_()
#         vi_disp.detach_()
#
#         # return ((ir, vi), (str(ir_path), str(it_path)),ir_warp,vi_warp
#         return (ir, vi), (str(ir_path), str(it_path))
#
#     # gt_tp[B,2,3]  gt_disp[B,1,1256,256,2]
#
#     def __len__(self):
#         return len(self.ir_list)

import torch.nn.functional as F


def add_padding(image_tensor, padding_size=60, padding_value=0):
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

class RegData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, crop=lambda x: x):
        super(RegData, self).__init__()
        self.crop = crop
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]


    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]

        assert ir_path.name == it_path.name, f"Mismatch ir:{ir_path.name} vi:{it_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        it = self.imread(path=it_path, flags=cv2.IMREAD_GRAYSCALE)


        # # crop same patch
        # patch = torch.cat([ir, it], dim=0)
        # patch = torchvision.transforms.functional.to_pil_image(patch)
        # patch = self.crop(patch)
        # patch = torchvision.transforms.functional.to_tensor(patch)
        # ir, vi = torch.chunk(patch, 2, dim=0)
        ir = add_padding(ir, padding_size=50)
        it = add_padding(it, padding_size=50)

        ir_path = str(ir_path)
        it_path = str(it_path)

        return ir, it, ir_path, it_path

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class RegTestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    # def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path, disp_folder: pathlib.Path):
    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path):
        super(RegTestData, self).__init__()

        # gain images list
        self.ir_list   = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list   = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        # self.disp_list = [x for x in sorted(disp_folder.glob('*')) if x.suffix in ['.npy']]


    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]
        # disp_path = self.disp_list[index]

        assert ir_path.name == it_path.name, f"Mismatch ir:{ir_path.name} vi:{it_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False)
        it = self.imread(path=it_path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False)
        # disp = torch.from_numpy(np.load(disp_path))


        # return (ir, it, disp), (str(ir_path), str(it_path), str(disp_path))
        ir_path = str(ir_path)
        it_path = str(it_path)

        ir = add_padding(ir, padding_size=50)
        it = add_padding(it, padding_size=50)

        return ir, it, ir_path, it_path

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts.unsqueeze(0) if unsqueeze else im_ts




