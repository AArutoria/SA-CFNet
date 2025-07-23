import torch
from torch.utils.data import DataLoader
from options import TrainOptions
from dataset import TestData, img_save
# from model import HR
from model import ADRNet
from utils.saver import Saver
from time import time
from tqdm import tqdm
from modules.losses import *
from build_table import *
import torch.nn as nn
# from thop import profile
from reg_data import RegTestData
from utils.utils import imsave,remove_padding

import os
from options import TrainOptions
from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform
from train import Train_and_test

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(opts):
    model = ADRNet()
    model.cuda()
    resweight = torch.load(r'F:\Code\TSCFusion\ADRNet-main\result\First_experiments\Absolt\One_layer\RoadScene\RoadSceneing_140.pth')['RES']
    # unweight = torch.load('D:\lxh\ADRNet-main\no_transformer\model_save\RoadScene\RoadSceneing_138.pth')['UN']
    model.RES.load_state_dict(resweight)
    # model.UN.load_state_dict(unweight)

    testdataset = RegTestData(opts.test_ir, opts.test_it)
    # testdataset = TestData(config)
    test_loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=opts.nThreads)
    # model.eval()
    start = time()
    total_loss = 0
    l1loss = nn.L1Loss()
    # for idx, [opt, sar, opt_warp, sar_warp,gt_tp, gt_disp,name_rgb,name_sar] in p_bar:
    p_test_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for it, [sar, rgb, sar_path, rgb_path] in p_test_bar:
        name, ext = os.path.splitext(os.path.basename(sar_path[0]))
        file_name = name + ext
        # affine = AffineTransform(degrees=30, translate=0.05)
        # elastic = ElasticTransform(kernel_size=101, sigma=16)
        # sar_affine, affine_disp, affine_theta = affine(sar)  # ir_affine [B,1,256,256] affine_theta[B,2,3] affine_disp[B,256,256,2]
        # sar_elastic, elastic_disp = elastic(sar_affine)
        # gt_disp = affine_disp + elastic_disp  # [B,256,256,2]
        sar_warp = sar
        # sar_warp = torch.cat((sar_warp,sar_warp,sar_warp),dim=1)
        # sar = torch.cat((sar, sar, sar), dim=1)
        # # sar = torch.cat((sar, sar, sar), dim=1)
        # rgb = torch.cat((rgb, rgb, rgb), dim=1)



        image_rgb_ = rgb.cuda()
        image_sar_ = sar.cuda()
        image_sar_warp_ = sar_warp.cuda()
        # gt_tp_ = affine_theta.cuda()
        # gt_disp_ = gt_disp.cuda()


        b,c,h,w = image_rgb_.shape
    
        with torch.no_grad():
            # image_rgb_, image_sar_, image_rgb_warp_, image_sar_warp_, gt_tp_, gt_disp_

            # image_sar_reg, cor_rmse = model.test_forward(image_rgb_, image_sar_, image_sar_warp_,gt_tp_, gt_disp_)
            image_sar_reg = model.test_forward(image_rgb_, image_sar_, image_sar_warp_)

            # gt = F.grid_sample(ir_warp_tensor, gt_disp, mode='bilinear', padding_mode='zeros', align_corners=True)
            # loss = torch.sum(abs(flow-gt_disp).pow(2))/(h*w)
            # # loss = l1loss(sar_reg_aff*255, gt*255)
            # total_loss = total_loss + loss
        image_rgb_ = remove_padding(image_rgb_, original_size=(256, 256))
        image_sar_reg = remove_padding(image_sar_reg, original_size=(256, 256))
        imsave(image_sar_reg, opts.dst / 'ir_reg', file_name)
        imsave(image_sar_warp_, opts.dst / 'after_reg', file_name)
        imsave(image_rgb_, opts.dst / 'reference_ir', file_name)
        # img_save(image_sar_reg, os.path.join("D:\\lxh\ADRNet-main\\no_transforme\\result\\reg", name_sar[0]+'.jpg'))
            # img_save(rgb_reg_aff, os.path.join("D:\\lxh\ADRNet-main\\no_transforme\\result\\reg"))
    # print((total_loss/335).sqrt())
   



if __name__ == '__main__':
    parser = TrainOptions()   # 加载选项信息
    opt = parser.parse()
    print('\n--- options load success ---')
    test(opts=opt)
