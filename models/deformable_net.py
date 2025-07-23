import torch
import torch.nn as nn
from torch.onnx.symbolic_opset9 import conv1d

from models.layers import SpatialTransformer, ResizeTransform, conv_block, predict_flow, conv2D, MatchCost
import numpy as np
from CoFiNet.imagenet import ImageEncoder,ImageUpSample
import torch.nn.functional as F
from einops import rearrange
from CoFiNet.transformer.attention import*
from CoFiNet.transformer.transformer import*
from CoFiNet.transformer.position_encoding import*
from models.pixel_wise_mapping import remap_using_flow_fields
from utils_2d.warp import Warper2d, warp2D
image_warp = Warper2d()

shape = (256, 256)
num_iter = 2

class DeformableNet(nn.Module):
    def __init__(self):
        super(DeformableNet, self).__init__()
        int_steps = 7   #
        self.inshape = shape
        self.downshape = (128,128)
        #
        down_shape2 = [int(d / 4) for d in self.inshape] # [64, 64]
        down_shape1 = [int(d / 2) for d in self.inshape] # [128, 128]
        self.spatial_transform_f = SpatialTransformer(volsize=self.downshape)
        self.spatial_transform_m = SpatialTransformer(volsize=self.inshape)
        # self.spatial_transform_s = SpatialTransformer(volsize=self.inshape)
        # self.spatial_transform_m = SpatialTransformer().cuda()
        self.num_iter = num_iter

        # FeatureLearning/Encoder functions
        dim = 2
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 1, 16, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 16, 32, 1))  # 1
        self.enc.append(conv_block(dim, 32, 64, 1))  # 2
        self.enc.append(conv_block(dim, 64, 64, 2))  # 3
        self.enc.append(conv_block(dim, 64, 128, 1))  # 4
        self.enc.append(conv_block(dim, 128, 128, 1))  # 5
        # self.enc = nn.ModuleList()
        # self.enc.append(conv_block(dim, 1, 16, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        # self.enc.append(conv_block(dim, 16, 16, 1))  # 1
        # self.enc.append(conv_block(dim, 16, 16, 1))  # 2
        # self.enc.append(conv_block(dim, 16, 32, 2))  # 3
        # self.enc.append(conv_block(dim, 32, 32, 1))  # 4
        # self.enc.append(conv_block(dim, 32, 64, 1))  # 5


        # Dncoder functions
        od = 32 + 1 ######UMF+Transformer
        od = 34   ####### UMF+Transformer+inner
        # od = 64 + 1
        # od =130 ########UMF+inner
        self.conv2_0 = conv_block(dim, od, 48, 1) # [48, 32, 16]
        self.enc.append(self.conv2_0)
        self.conv2_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv2_1)
        self.conv2_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv2_2)
        # self.predict_flow2a = predict_flow(1, 2)
        self.predict_flow2a = predict_flow(16, 2)
        # self.predict_flow2a = predict_flow(34, 2)
        self.enc.append(self.predict_flow2a)

        self.dc_conv2_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1) # [48, 48, 32]
        self.enc.append(self.dc_conv2_0)
        self.dc_conv2_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv2_1)
        # self.dc_conv2_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv2_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv2_2)
        self.predict_flow2b = predict_flow(32, 2)
        # self.predict_flow2b = predict_flow(64, 2)
        self.enc.append(self.predict_flow2b)

        # od = 1 + 16 + 16 + 2
        # od = 69 ######UMF+Transformer
        od = 70 ########UMF+Transformer+inner
        # od = 84 #############UMF+inner
        self.conv1_0 = conv_block(dim, od, 48, 1)
        self.enc.append(self.conv1_0)
        self.conv1_1 = conv_block(dim, 48, 32, 1)
        self.enc.append(self.conv1_1)
        self.conv1_2 = conv_block(dim, 32, 16, 1)
        self.enc.append(self.conv1_2)
        self.predict_flow1a = predict_flow(16, 2)
        self.enc.append(self.predict_flow1a)

        self.dc_conv1_0 = conv2D(2, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.enc.append(self.dc_conv1_0)
        self.dc_conv1_1 = conv2D(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc.append(self.dc_conv1_1)
        self.dc_conv1_2 = conv2D(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.enc.append(self.dc_conv1_2)
        self.predict_flow1b = predict_flow(32, 2)
        self.enc.append(self.predict_flow1b)

        self.resize = ResizeTransform(1 / 2, dim)

        self.img_encoder = ImageEncoder()
        self.pe_H = int(256 / 8)
        self.pe_W = int(256 / 8)
        # self.img_pos_encoding = PositionEmbeddingCoordsSine(2, 128)
        self.transformer = LocalFeatureTransformer(D_MODEL=128, NHEAD=4, LAYER_NAMES=['self', 'cross'] * 1,
                                                   ATTENTION='full')
        self.transformer = LocalFeatureTransformer(D_MODEL=128, NHEAD=4, LAYER_NAMES=['cross'] * 1,
                                                   ATTENTION='full')
        self.img_score_layer = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.InstanceNorm2d(128), nn.ReLU(),
                                             nn.Conv2d(128, 64, 1, bias=False), nn.InstanceNorm2d(64), nn.ReLU(),
                                             nn.Conv2d(64, 1, 1, bias=False), nn.Sigmoid())
        self.img_upsample_1 = ImageUpSample(128+64, 128)
        self.img_upsample_2 = ImageUpSample(128 + 64, 64)
        self.conv1 = nn.Conv2d(4,2,1)



    def load_state_dict(self, state_dict, strict = False):
        state_dict.pop('spatial_transform_m.grid')
        state_dict.pop('spatial_transform_f.grid')
        super().load_state_dict(state_dict, strict)

    def forward(self, tgt, src, shape=None):

        # c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 16, 128, 128])
        # c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 16, 128, 128])
        # c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 32, 64, 64])
        # c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 32, 64, 64])
        #
        # c12 = rearrange(c12, 'b c h w -> b (h w) c')
        # c22 = rearrange(c22, 'b c h w -> b (h w) c')
        # c11, c21 = self.transformer(c12, c22) #[B,1024,128]
        #
        # c11 = rearrange(c11, 'b (h w) c -> b c h w', h=c12.shape[2])
        # c21 = rearrange(c21, 'b (h w) c -> b c h w', h=c22.shape[2]) #[B,128,32,32]
        #
        # c11 = c11.permute(0,3,1,2)
        # c21 = c21.permute(0,3,1,2)
        #
        # ##################### Estimation at scale-2 #######################
        # similar = MatchCost(c11, c21) #[B,1,32,32]   # torch.Size([16,  1, 64, 64])
        # flow2 = self.predict_flow2a(similar) # torch.Size([16, 2, 64, 64]) flow2: flow field
        # # c11, disp_pre = self.spatial_transform_f(c11, flow2)  # torch.Size([16, 16, 128, 128])
        # # corr2 = MatchCost(c11, c21)  # [B,1,32,32]   # torch.Size([16,  1, 64, 64])
        # # flow2 = self.predict_flow2a(corr2)  # torch.Size([16, 2, 64, 64]) flow2: flow field
        #
        # # upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])
        #
        # x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])
        #
        # refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        # int_flow2 = refine_flow2
        # # int_flow2 = self.integrate2(refine_flow2)
        # up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128]) #[2,2,64,64]
        # # flow = up_int_flow2.permute(0,2,3,1)
        # flow = up_int_flow2
        # src, disp_pre = self.spatial_transform_m(src, flow) # torch.Size([16, 16, 128, 128])
        # f_warp, _ = self.spatial_transform_m(tgt, (-flow))
        #
        #
        # # #
        # c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 16, 128, 128])
        # c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 16, 128, 128])
        # c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 32, 64, 64])
        # c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 32, 64, 64])
        #
        # c12 = rearrange(c12, 'b c h w -> b (h w) c')
        # c22 = rearrange(c22, 'b c h w -> b (h w) c')
        # c11, c21 = self.transformer(c12, c22) #[B,1024,128]
        #
        # c11 = rearrange(c11, 'b (h w) c -> b c h w', h=c12.shape[2])
        # c21 = rearrange(c21, 'b (h w) c -> b c h w', h=c22.shape[2]) #[B,128,32,32]
        #
        # c11 = c11.permute(0,3,1,2)
        # c21 = c21.permute(0,3,1,2)
        #
        # ##################### Estimation at scale-2 #######################
        # similar = MatchCost(c11, c21) #[B,1,32,32]   # torch.Size([16,  1, 64, 64])
        # flow2 = self.predict_flow2a(similar) # torch.Size([16, 2, 64, 64]) flow2: flow field
        # # c11, disp_pre = self.spatial_transform_f(c11, flow2)  # torch.Size([16, 16, 128, 128])
        # # corr2 = MatchCost(c11, c21)  # [B,1,32,32]   # torch.Size([16,  1, 64, 64])
        # # flow2 = self.predict_flow2a(corr2)  # torch.Size([16, 2, 64, 64]) flow2: flow field
        #
        # # upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])
        #
        # x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])
        #
        # refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        # int_flow2 = refine_flow2
        # # int_flow2 = self.integrate2(refine_flow2)
        # up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128]) #[2,2,64,64]
        # # flow = up_int_flow2.permute(0,2,3,1)
        # flow = up_int_flow2
        # src, disp_pre = self.spatial_transform_m(src, flow) # torch.Size([16, 16, 128, 128])
        # f_warp, _ = self.spatial_transform_m(tgt, (-flow))
        #
        # #
        # c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 16, 128, 128])
        # c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 16, 128, 128])
        # c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 32, 64, 64])
        # c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 32, 64, 64])
        #
        # c12 = rearrange(c12, 'b c h w -> b (h w) c')
        # c22 = rearrange(c22, 'b c h w -> b (h w) c')
        # c11, c21 = self.transformer(c12, c22) #[B,1024,128]
        #
        # c11 = rearrange(c11, 'b (h w) c -> b c h w', h=c12.shape[2])
        # c21 = rearrange(c21, 'b (h w) c -> b c h w', h=c22.shape[2]) #[B,128,32,32]
        #
        # c11 = c11.permute(0,3,1,2)
        # c21 = c21.permute(0,3,1,2)
        #
        # ##################### Estimation at scale-2 #######################
        # similar = MatchCost(c11, c21) #[B,1,32,32]   # torch.Size([16,  1, 64, 64])
        # flow2 = self.predict_flow2a(similar) # torch.Size([16, 2, 64, 64]) flow2: flow field
        # # c11, disp_pre = self.spatial_transform_f(c11, flow2)  # torch.Size([16, 16, 128, 128])
        # # corr2 = MatchCost(c11, c21)  # [B,1,32,32]   # torch.Size([16,  1, 64, 64])
        # # flow2 = self.predict_flow2a(corr2)  # torch.Size([16, 2, 64, 64]) flow2: flow field
        #
        # # upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])
        #
        # x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])
        #
        # refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        # int_flow2 = refine_flow2
        # # int_flow2 = self.integrate2(refine_flow2)
        # up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128]) #[2,2,64,64]
        # # flow = up_int_flow2.permute(0,2,3,1)
        # flow = up_int_flow2
        # m_warp, disp_pre = self.spatial_transform_m(src, flow) # torch.Size([16, 16, 128, 128])
        # f_warp, _ = self.spatial_transform_m(tgt, (-flow))
        # #


        # #################UMF+transformer##############################
        # if shape is not None:
        #     down_shape1 = [int(d / 2) for d in shape]
        #     self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        #     self.spatial_transform_m = SpatialTransformer(volsize=shape)
        # ##################### Feature extraction #########################
        # c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 16, 128, 128])
        # c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 16, 128, 128])
        # c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 32, 64, 64])
        # c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 32, 64, 64])
        #
        # c12 = rearrange(c12, 'b c h w -> b (h w) c')
        # c22 = rearrange(c22, 'b c h w -> b (h w) c')
        # c12, c22 = self.transformer(c12, c22) #[B,1024,128]
        #
        # c12 = rearrange(c12, 'b (h w) c -> b c h w', h=c12.shape[2])
        # c22 = rearrange(c22, 'b (h w) c -> b c h w', h=c22.shape[2]) #[B,128,32,32]
        #
        # c12= c12.permute(0,3,1,2)
        # c22 = c22.permute(0,3,1,2)
        #
        # ##################### Estimation at scale-2 #######################
        # corr2 = MatchCost(c22, c12)    # torch.Size([16,  1, 64, 64])
        # x = torch.cat((corr2, c22), 1) # torch.Size([16, 33, 64, 64])
        # x = self.conv2_0(x) # torch.Size([16, 48, 64, 64])
        # x = self.conv2_1(x) # torch.Size([16, 32, 64, 64])
        # x = self.conv2_2(x) # torch.Size([16, 16, 64, 64])
        # flow2 = self.predict_flow2a(x) # torch.Size([16, 2, 64, 64]) flow2: flow field
        # # upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])
        #
        # x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])
        #
        # refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        # int_flow2 = refine_flow2
        # # int_flow2 = self.integrate2(refine_flow2)
        # up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128])
        # # features_s_warped, _ = self.spatial_transform_f(c11, up_int_flow2) # torch.Size([16, 16, 128, 128])
        # features_s_warped, _ = self.spatial_transform_f(c11, int_flow2)  # torch.Size([16, 16, 128, 128])
        #
        #
        # ##################### Estimation at scale-1 #######################
        # corr1 = MatchCost(c21, features_s_warped) # torch.Size([16, 1, 128, 128])
        # # x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1) # torch.Size([16, 35, 112, 112])
        # x = torch.cat((corr1, c21, int_flow2, flow2), 1)  # torch.Size([16, 35, 112, 112])
        # x = self.conv1_0(x) # torch.Size([16, 48, 128, 128])
        # x = self.conv1_1(x) # torch.Size([16, 32, 128, 128])
        # x = self.conv1_2(x) # torch.Size([16, 16, 128, 128])
        # # flow1 = self.predict_flow1a(x) + up_int_flow2 # torch.Size([16, 2, 128, 128])
        # flow1 = self.predict_flow1a(x) + int_flow2  # torch.Size([16, 2, 128, 128])
        #
        # x = self.dc_conv1_0(flow1) # torch.Size([16, 48, 128, 128])
        # x = self.dc_conv1_1(x) # torch.Size([16, 48, 128, 128])
        # x = self.dc_conv1_2(x) # torch.Size([16, 32, 128, 128])
        # refine_flow1 = self.predict_flow1b(x) + flow1 # torch.Size([16, 2, 128, 128])
        # int_flow1 = refine_flow1
        # # int_flow1 = self.integrate1(refine_flow1)
        #
        # ##################### Upsample to scale-0 #######################
        # flow = self.resize(int_flow1) # torch.Size([16, 2, 256, 256])
        # m_warp, disp_pre = self.spatial_transform_m(src, flow) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])
        # # wd+
        # f_warp, _ = self.spatial_transform_m(tgt, (-flow)) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])

        #################UMF+transformer+feature_inner##############################
        if shape is not None:
            down_shape1 = [int(d / 2) for d in shape]
            self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
            self.spatial_transform_m = SpatialTransformer(volsize=shape)
        ##################### Feature extraction #########################
        c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 64, 128, 128])
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 64, 128, 128])
        c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 128, 64, 64])
        c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 128, 64, 64])

        c12 = rearrange(c12, 'b c h w -> b (h w) c')#[B,4096,128]
        c22 = rearrange(c22, 'b c h w -> b (h w) c')#[B,4096,128]
        c12, c22 = self.transformer(c12, c22) #[B,4096,128]

        c12 = rearrange(c12, 'b (h w) c -> b c h w', h=c12.shape[2]) #[B,128,128,32]
        c22 = rearrange(c22, 'b (h w) c -> b c h w', h=c22.shape[2]) #[B,128,128,32]

        c12= c12.permute(0,3,1,2) #[B,32,128,128]
        c22 = c22.permute(0,3,1,2)

        ##################### Estimation at scale-2 #######################
        corr2 = MatchCost(c22, c12)    # torch.Size([B,  1, 128, 128])
        inner = torch.sum(c22*c12,dim=1).unsqueeze(1) #B,  1, 128, 128]
        corr2 = torch.concat([corr2,inner],dim=1) #[B,2,128,128]
        x = torch.cat((corr2, c22), 1) # torch.Size([B, 34, 128, 128])
        x = self.conv2_0(x) # torch.Size([B, 48, 128, 128])
        x = self.conv2_1(x) # torch.Size([B, 32, 128, 128])
        x = self.conv2_2(x) # torch.Size([B, 16, 128, 128])
        flow2 = self.predict_flow2a(x) # torch.Size([B, 2, 128, 128]) flow2: flow field
        # upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])

        x = self.dc_conv2_0(flow2) # torch.Size([B, 48, 128, 128])
        x = self.dc_conv2_1(x) # torch.Size([B, 48, 128, 128])
        x = self.dc_conv2_2(x) # torch.Size([B, 32, 128, 128])

        refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([B, 2, 128, 128])
        # refine_flow2 = flow2
        int_flow2 = refine_flow2
        # int_flow2 = self.integrate2(refine_flow2)
        # up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128])
        # features_s_warped, _ = self.spatial_transform_f(c11, up_int_flow2) # torch.Size([16, 16, 128, 128])
        features_s_warped, _ = self.spatial_transform_f(c11, int_flow2)  # torch.Size([B, 64, 128, 128])


        ##################### Estimation at scale-1 #######################
        corr1 = MatchCost(c21, features_s_warped) # torch.Size([B, 1, 128, 128])
        inner = torch.sum(c21*features_s_warped,dim=1).unsqueeze(1) #[B,1,128,128]
        corr1 = torch.concat([corr1,inner],dim=1) #[B,2,128,128]
        # x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1) # torch.Size([16, 35, 112, 112])
        x = torch.cat((corr1, c21, int_flow2, flow2), 1)  # torch.Size([B, 70, 128, 128])
        x = self.conv1_0(x) # torch.Size([16, 48, 128, 128]) #[B,48,128,128]
        x = self.conv1_1(x) # torch.Size([16, 32, 128, 128]) #[B,32,128,128]
        x = self.conv1_2(x) # torch.Size([16, 16, 128, 128])  #[B,16,128,128]
        # flow1 = self.predict_flow1a(x) + up_int_flow2 # torch.Size([16, 2, 128, 128])
        flow1 = self.predict_flow1a(x) + int_flow2  # torch.Size([16, 2, 128, 128]) #[B,2,128,128]

        x = self.dc_conv1_0(flow1) # torch.Size([B, 48, 128, 128])
        x = self.dc_conv1_1(x) # torch.Size([B, 48, 128, 128])
        x = self.dc_conv1_2(x) # torch.Size([B, 32, 128, 128])
        refine_flow1 = self.predict_flow1b(x) + flow1 # torch.Size([B, 2, 128, 128])
        int_flow1 = refine_flow1 #[B,2,128,128]
        # int_flow1 = self.integrate1(refine_flow1)

        ##################### Upsample to scale-0 #######################
        flow = self.resize(int_flow1) # torch.Size([16, 2, 256, 256])
        m_warp, disp_pre = self.spatial_transform_m(src, flow) # torch.Size([B, 1, 256, 256]) torch.Size([16, 256, 256, 2])
        # disp_pre = flow
        # m_warp = image_warp(flow, src)
        # wd+
        f_warp, _ = self.spatial_transform_m(tgt, (-flow)) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])
        # f_warp = image_warp((-flow), tgt)
        #
        # #################UMF+one_cross_transformer+feature_inner##############################
        # if shape is not None:
        #     down_shape1 = [int(d / 2) for d in shape]
        #     self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        #     self.spatial_transform_m = SpatialTransformer(volsize=shape)
        # ##################### Feature extraction #########################
        # c11 = self.enc[2](self.enc[1](self.enc[0](src))) # torch.Size([16, 16, 128, 128])
        # c21 = self.enc[2](self.enc[1](self.enc[0](tgt))) # torch.Size([16, 16, 128, 128])
        # c12 = self.enc[5](self.enc[4](self.enc[3](c11))) # torch.Size([16, 32, 64, 64])
        # c22 = self.enc[5](self.enc[4](self.enc[3](c21))) # torch.Size([16, 32, 64, 64])
        #
        # # c12 = rearrange(c12, 'b c h w -> b (h w) c')
        # # c22 = rearrange(c22, 'b c h w -> b (h w) c')
        # # c12, c22 = self.transformer(c12, c22) #[B,1024,128]
        # #
        # # c12 = rearrange(c12, 'b (h w) c -> b c h w', h=c12.shape[2])
        # # c22 = rearrange(c22, 'b (h w) c -> b c h w', h=c22.shape[2]) #[B,128,32,32]
        # #
        # # c12= c12.permute(0,3,1,2)
        # # c22 = c22.permute(0,3,1,2)
        #
        # ##################### Estimation at scale-2 #######################
        # corr2 = MatchCost(c22, c12)    # torch.Size([16,  1, 64, 64])
        # inner = torch.sum(c22*c12,dim=1).unsqueeze(1) #[B,356,356]
        # corr2 = torch.concat([corr2,inner],dim=1)
        # x = torch.cat((corr2, c22), 1) # torch.Size([16, 33, 64, 64])
        # x = self.conv2_0(x) # torch.Size([16, 48, 64, 64])
        # x = self.conv2_1(x) # torch.Size([16, 32, 64, 64])
        # x = self.conv2_2(x) # torch.Size([16, 16, 64, 64])
        # flow2 = self.predict_flow2a(x) # torch.Size([16, 2, 64, 64]) flow2: flow field
        # upfeat2 = self.resize(x) # torch.Size([16, 16, 128, 128])
        #
        # x = self.dc_conv2_0(flow2) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_1(x) # torch.Size([16, 48, 64, 64])
        # x = self.dc_conv2_2(x) # torch.Size([16, 32, 64, 64])
        #
        # refine_flow2 = self.predict_flow2b(x) + flow2 # torch.Size([16, 2, 64, 64])
        # int_flow2 = refine_flow2
        # # int_flow2 = self.integrate2(refine_flow2)
        # up_int_flow2 = self.resize(int_flow2) # torch.Size([16, 2, 128, 128])
        # features_s_warped, _ = self.spatial_transform_f(c11, up_int_flow2) # torch.Size([16, 16, 128, 128])
        # # features_s_warped, _ = self.spatial_transform_f(c11, int_flow2)  # torch.Size([16, 16, 128, 128])
        #
        #
        # ##################### Estimation at scale-1 #######################
        # corr1 = MatchCost(c21, features_s_warped) # torch.Size([16, 1, 128, 128])
        # inner = torch.sum(c21*features_s_warped,dim=1).unsqueeze(1) #[B,356,356]
        # corr1 = torch.concat([corr1,inner],dim=1)
        # # x = torch.cat((corr1, c21, int_flow2, corr2), 1) # torch.Size([16, 35, 112, 112])
        # x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1)  # torch.Size([16, 35, 112, 112])
        # x = self.conv1_0(x) # torch.Size([16, 48, 128, 128])
        # x = self.conv1_1(x) # torch.Size([16, 32, 128, 128])
        # x = self.conv1_2(x) # torch.Size([16, 16, 128, 128])
        # flow1 = self.predict_flow1a(x) + up_int_flow2 # torch.Size([16, 2, 128, 128])
        # # flow1 = self.predict_flow1a(x) + int_flow2  # torch.Size([16, 2, 128, 128])
        #
        # x = self.dc_conv1_0(flow1) # torch.Size([16, 48, 128, 128])
        # x = self.dc_conv1_1(x) # torch.Size([16, 48, 128, 128])
        # x = self.dc_conv1_2(x) # torch.Size([16, 32, 128, 128])
        # refine_flow1 = self.predict_flow1b(x) + flow1 # torch.Size([16, 2, 128, 128])
        # int_flow1 = refine_flow1
        # # int_flow1 = self.integrate1(refine_flow1)
        #
        # ##################### Upsample to scale-0 #######################
        # flow = self.resize(int_flow1) # torch.Size([16, 2, 256, 256])
        # m_warp, disp_pre = self.spatial_transform_m(src, flow) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])
        # # wd+
        # f_warp, _ = self.spatial_transform_m(tgt, (-flow)) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])


        return m_warp, f_warp, flow, int_flow2, disp_pre


def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()


class SSpatialTransformer(nn.Module):
    def __init__(self):
        super(SSpatialTransformer, self).__init__()
        # int_steps = 7   #
        self.inshape = shape
        self.downshape = (128,128)

        down_shape2 = [int(d / 4) for d in self.inshape] # [64, 64]
        down_shape1 = [int(d / 2) for d in self.inshape] # [128, 128]
        self.spatial_transform_f = SpatialTransformer(volsize=self.downshape)
        self.spatial_transform_m = SpatialTransformer(volsize=self.inshape)
        # self.spatial_transform_s = SpatialTransformer(volsize=self.inshape)
        # self.spatial_transform_m = SpatialTransformer().cuda()


    def load_state_dict(self, state_dict, strict = False):
        state_dict.pop('spatial_transform_m.grid')
        state_dict.pop('spatial_transform_f.grid')
        super().load_state_dict(state_dict, strict)

    def forward(self, src, flow, shape=None):

        src, disp_pre = self.spatial_transform_m(src, flow) # torch.Size([16, 16, 128, 128])

        return src




if __name__ == '__main__':
    #
    model = DeformableNet().cuda()
    a = torch.randn(8, 1, 256, 256).cuda()
    b = torch.randn(8, 1, 256, 256).cuda()
    m_warp, f_warp, flow, int_flow1, int_flow2, disp_pre = model(a,b)
    print(m_warp.shape, f_warp.shape, flow.shape)
    print(int_flow1.shape, int_flow2.shape)


