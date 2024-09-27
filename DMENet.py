from swintransformer import swin_s
import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from math import log
def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool
class ConvBNR(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,dilation=1):
        super(ConvBNR, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ConvBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CAM_Multimodal_Module(Module):
    def __init__(self):
        super(CAM_Multimodal_Module, self).__init__()
    def forward(self, rgb, depth):
        m_batchsize, C, height, width = rgb.size()
        proj_query = rgb.view(m_batchsize, C, -1)
        proj_key = depth.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        cl_index = torch.argmax(energy, dim=2)
        rgbd = rgb + depth[torch.arange(m_batchsize).unsqueeze(1), cl_index, :]
        return rgbd


class PixelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        attention_map = self.sigmoid(x)
        return attention_map

class DepthAttentionModel(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.rgb_conv = nn.Conv2d(channel, 32, kernel_size=1)
        self.depth_conv = nn.Conv2d(channel, 32, kernel_size=1)
        self.attention_module = PixelAttentionModule(64)  # 假设特征合并后的通道数
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_conv(rgb)
        depth_feat = self.depth_conv(depth)
        combined_feat = torch.cat((rgb_feat, depth_feat), dim=1)
        attention_map = self.attention_module(combined_feat)
        weighted_depth_feat = depth * attention_map
        return weighted_depth_feat

class DAFF(nn.Module):
    def __init__(self, in_channels, out_channels,last_channels):
        super(DAFF, self).__init__()
        self.cam = CAM_Multimodal_Module()
        self.ssa = SSA()
        self.layer_ful2 = nn.Sequential(
            nn.Conv2d(in_channels + last_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), )
        self.depa=DepthAttentionModel(in_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    def forward(self, rgb, depth, rgbd_last):
        w_depth=self.depa(rgb,depth)
        fused = self.cam(rgb, w_depth)
        weighted = self.ssa(rgb, depth, fused)*fused+fused
        f_affo = weighted+rgb
        rgbd = torch.cat([f_affo, rgbd_last], dim=1)
        rgbd = self.layer_ful2(rgbd)
        return rgbd

class DAFF0(nn.Module):
    def __init__(self, in_channels):
        super(DAFF0, self).__init__()
        self.cam = CAM_Multimodal_Module()
        self.ssa = SSA()
        self.depa = DepthAttentionModel(in_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    def forward(self, rgb, depth):
        w_depth = self.depa(rgb, depth)
        fused = self.cam(rgb, w_depth)
        weighted = self.ssa(rgb, depth, fused)*fused+fused
        f_affo =weighted+rgb
        return f_affo

class SSA(nn.Module):
    def __init__(self):
        super(SSA, self).__init__()
        self.spatial_attention1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), )
        self.spatial_attention2 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), )
        self.spatial_attention3 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), )
        self.sigmoid = nn.Sigmoid()
    def forward(self, rgb, depth, rgbd):
        max_pool_rgb = torch.max(rgb, dim=1, keepdim=True)[0]
        avg_pool_rgb = torch.mean(rgb, dim=1, keepdim=True)
        x_r = torch.cat([max_pool_rgb, avg_pool_rgb], dim=1)
        max_pool_depth = torch.max(depth, dim=1, keepdim=True)[0]
        avg_pool_depth = torch.mean(depth, dim=1, keepdim=True)
        x_d = torch.cat([max_pool_depth, avg_pool_depth], dim=1)
        max_pool_rgbd = torch.max(rgbd, dim=1, keepdim=True)[0]
        avg_pool_rgbd = torch.mean(rgbd, dim=1, keepdim=True)
        x_rgbd = torch.cat([max_pool_rgbd, avg_pool_rgbd], dim=1)
        SA1 = self.spatial_attention1(x_r)
        SA2 = self.spatial_attention2(x_d)
        SA3 = self.spatial_attention3(x_rgbd)
        SA_fused = SA1 * SA3 + SA2 * SA3  + SA3
        SA_fused = self.sigmoid(SA_fused)
        return SA_fused

class FAM(nn.Module):
    def __init__(self,inchannel1,inchannel2,channel):
        super(FAM, self).__init__()
        self.auxconv=ConvBNR(inchannel1+inchannel2,(inchannel1+inchannel2)//2,kernel_size=1,padding=0)
        self.aux=GCM((inchannel1+inchannel2)//2,channel)
        self.ssa=SSA()
    def forward(self, rgb, depth, rgbd, rgbde1,rgbde2):
        w= rgbde2.size(2)
        rgbde2_o=rgbde2
        if rgbde1.size(2) != w:
            rgbde1 = F.interpolate(rgbde1, size=(int(w*0.75), int(w*0.75)),mode='bilinear', align_corners=False)
            rgbde2 = F.interpolate(rgbde2, size=(int(w*0.75), int(w*0.75)),mode='bilinear', align_corners=False)
            rgbd = F.interpolate(rgbd, size=(int(w*0.75), int(w*0.75)),mode='bilinear', align_corners=False)
        aux = torch.cat([rgbde1, rgbde2], dim=1)
        aux = self.auxconv(aux)
        aux= self.aux(aux)
        agg=torch.cat([aux,rgbd],dim=1)
        if agg.size(2) != w:
            agg = F.interpolate(agg, size=(w, w), mode='bilinear', align_corners=False)
        agg=self.ssa(rgb,depth,agg)*agg+agg
        agg=torch.cat([agg,rgbde2_o],dim=1)
        return agg

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel=32):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
            ConvBN(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            ConvBN(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            ConvBN(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
            ConvBN(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            ConvBN(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            ConvBN(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
            ConvBN(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            ConvBN(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            ConvBN(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = ConvBN(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = ConvBN(in_channel, out_channel, 1)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class GCM_egfe(nn.Module):
    def __init__(self, in_channel, out_channel=32):
        super(GCM_egfe, self).__init__()
        self.egfe = EGFE(in_channel)
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
            ConvBN(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            ConvBN(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            ConvBN(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
            ConvBN(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            ConvBN(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            ConvBN(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            ConvBN(in_channel, out_channel, 1),
            ConvBN(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            ConvBN(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            ConvBN(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = ConvBN(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = ConvBN(in_channel, out_channel, 1)
    def forward(self, x,edge,f_edge):
        x=self.egfe(x,edge,f_edge)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.q_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)
    def forward(self, rgbd, x):
        batch_size = rgbd.size(0)
        q = self.q_conv(x).view(batch_size, self.inter_channels, -1)
        q = q.permute(0, 2, 1)
        k = self.k_conv(rgbd).view(batch_size, self.inter_channels, -1)
        v = self.v_conv(rgbd).view(batch_size, self.inter_channels, -1)
        v = v.permute(0, 2, 1)
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(attention, v)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *rgbd.size()[2:])
        out = self.out_conv(out)
        return rgbd + out

class CAED(nn.Module):
    def __init__(self,in_channels1,in_channels2,dep_channel):
        super(CAED, self).__init__()
        self.reduce1_rgbd = ConvBNR(in_channels1, 64, kernel_size=1,padding=0)
        self.reduce1_rgb = ConvBNR(in_channels1, 64, kernel_size=1,padding=0)
        self.conv2d_1 = ConvBNR(64, 64, 3)
        self.reduce4_rgbd = ConvBNR(in_channels2, 256, kernel_size=1,padding=0)
        self.reduce4_dep = ConvBNR(dep_channel, 256, kernel_size=1,padding=0)
        self.conv2d_2 = ConvBNR(256, 256, 3)
        self.crossatt1=CrossAttention(64)
        self.crossatt2=CrossAttention(256)
        self.block1 = nn.Sequential(ConvBNR(256 + 64, 256, 3))
        self.block2 = nn.Sequential(ConvBNR(256, 256, 3),
                                   nn.Conv2d(256, 1, 1))
        self.sigmoid=nn.Sigmoid()
    def forward(self, x1, x4,r1,d4):
        size = x1.size()[2:]
        x1 = self.reduce1_rgbd(x1)
        r1 = self.reduce1_rgb(r1)
        x1= self.crossatt1(x1,r1)
        x1= self.conv2d_1(x1)
        x4 = self.reduce4_rgbd(x4)
        d4 = self.reduce4_dep(d4)
        x4 = self.crossatt2(x4,d4)
        x4 = self.conv2d_2(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        f_edge = self.block1(out)
        out=self.block2(f_edge)
        out=self.sigmoid(out)
        return out,f_edge

class EGFE(nn.Module):
    def __init__(self, channel):
        super(EGFE, self).__init__()
        self.conv0= ConvBNR(256,128,kernel_size=1,padding=0)
        self.conv1 = ConvBNR(128, 128, 3, padding=1)
        self.conv2 = ConvBNR(128,channel,kernel_size=1,padding=0)
        self.reconv = ConvBNR(channel, channel, 3, padding=1)
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d_1 = ConvBNR(channel, channel, 3, padding=1)
        self.conv2d_3_11 = ConvBNR(channel, channel, 1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d_1 = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv1d_2 = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, c, att,f_edge):
        if f_edge.size() != att.size():
            f_edge = F.interpolate(f_edge, c.size()[2:], mode='bilinear', align_corners=False)
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        w= self.conv0(f_edge)
        w= self.conv1(w)
        w= self.conv2(w)
        w = w*c+c
        w= self.reconv(w)
        x = w * att + w
        x = self.conv2d_1(x)
        wei = self.avg_pool(x)
        wei = self.conv1d_2(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei
        return x

class DMENet(nn.Module):
    def __init__(self, channel=32):
        super(DMENet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.layer_rgb = swin_s(pretrained=True)
        self.layer_dep = swin_s(pretrained= True)
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)

        self.egdt = CAED(96, 768,768)

        self.fu_0 = DAFF0(96)  #
        self.pool_fu_0 = maxpool()
        self.fu_1 = DAFF(192, 96, 96)  # MixedFusion_Block_IMfusion
        self.pool_fu_1 = maxpool()
        self.fu_2 = DAFF(384, 192, 96)
        self.pool_fu_2 = maxpool()
        self.fu_3 = DAFF(768, 384, 192)
        self.pool_fu_3 = maxpool()
        self.fu_4 = DAFF(768, 768, 384)

        self.ful_gcm_4 = GCM_egfe(768, channel)
        self.fam4=FAM(768,384,channel)
        self.ful_gcm_3 = GCM_egfe(384+2*channel, channel)
        self.fam3 = FAM(384, 192, channel)
        self.ful_gcm_2 = GCM_egfe(192 + 2 * channel, channel)
        self.fam2 = FAM(192, 96, channel)
        self.ful_gcm_1 = GCM_egfe(96 + 2 * channel, channel)
        self.fam1 = FAM(96, 96, channel)
        self.ful_gcm_0 = GCM_egfe(96 + 2 * channel, channel)
        self.ful_conv_out = nn.Conv2d(channel, 1, 1)

        self.rgb_gcm_4 = GCM(768, channel)
        self.rgb_gcm_3 = GCM(768+ channel, channel)
        self.rgb_gcm_2 = GCM(384 + channel, channel)
        self.rgb_gcm_1 = GCM(192 + channel, channel)
        self.rgb_gcm_0 = GCM(96 + channel, channel)
        self.rgb_conv_out = nn.Conv2d(channel, 1, 1)

        self.d_gcm_4 = GCM(768, channel)
        self.d_gcm_3 = GCM(768+ channel, channel)
        self.d_gcm_2 = GCM(384 + channel, channel)
        self.d_gcm_1 = GCM(192 + channel, channel)
        self.d_gcm_0 = GCM(96 + channel, channel)
        self.d_conv_out = nn.Conv2d(channel, 1, 1)


        self.ful_conv_out1 = nn.Conv2d(channel, 1, 1)
        self.aux1_out = nn.Conv2d(channel, 1, 1)
        self.ful_conv_out2 = nn.Conv2d(channel, 1, 1)
        self.aux2_out = nn.Conv2d(channel, 1, 1)
        self.ful_conv_out3 = nn.Conv2d(channel, 1, 1)
        self.aux3_out = nn.Conv2d(channel, 1, 1)
        self.ful_conv_out4 = nn.Conv2d(channel, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs, depths):

        img_0, img_1, img_2, img_3,img_4 = self.layer_rgb(imgs)
        dep_0, dep_1, dep_2, dep_3,dep_4 = self.layer_dep(self.layer_dep0(depths))

        ful_0 = self.fu_0(img_0, dep_0)
        ful_1 = self.fu_1(img_1, dep_1, self.pool_fu_0(ful_0))
        ful_2 = self.fu_2(img_2, dep_2, self.pool_fu_1(ful_1))
        ful_3 = self.fu_3(img_3, dep_3, self.pool_fu_2(ful_2))
        ful_4 = self.fu_4(img_4, dep_4, ful_3)

        edge,f_edge = self.egdt(ful_0, ful_4,img_0,dep_4)

        x_rgb_42 = self.rgb_gcm_4(img_4)
        x_d_42 = self.d_gcm_4(dep_4)
        x_ful_42 = self.ful_gcm_4(ful_4, edge,f_edge)

        x_rgb_3_cat = torch.cat([img_3, x_rgb_42], dim=1)
        x_rgb_32 = self.rgb_gcm_3(x_rgb_3_cat)
        x_d_3_cat = torch.cat([dep_3, x_d_42], dim=1)
        x_d_32 = self.d_gcm_3(x_d_3_cat)
        agg4=self.fam4(x_rgb_32,x_d_32,x_ful_42,ful_4,ful_3)
        x_ful_32 = self.ful_gcm_3(agg4, edge,f_edge)

        x_rgb_2_cat = torch.cat([img_2, self.upsample_2(x_rgb_32)], dim=1)
        x_rgb_22 = self.rgb_gcm_2(x_rgb_2_cat)
        x_d_2_cat = torch.cat([dep_2, self.upsample_2(x_d_32)], dim=1)
        x_d_22 = self.d_gcm_2(x_d_2_cat)
        agg3 = self.fam3(x_rgb_22, x_d_22, x_ful_32, ful_3, ful_2)
        x_ful_22 = self.ful_gcm_2(agg3, edge,f_edge)

        x_rgb_1_cat = torch.cat([img_1, self.upsample_2(x_rgb_22)], dim=1)
        x_rgb_12 = self.rgb_gcm_1(x_rgb_1_cat)
        x_d_1_cat = torch.cat([dep_1, self.upsample_2(x_d_22)], dim=1)
        x_d_12 = self.d_gcm_1(x_d_1_cat)
        agg2 = self.fam2(x_rgb_12, x_d_12, x_ful_22, ful_2, ful_1)
        x_ful_12 = self.ful_gcm_1(agg2, edge,f_edge)

        x_rgb_0_cat = torch.cat([img_0, self.upsample_2(x_rgb_12)], dim=1)
        x_rgb_02 = self.rgb_gcm_0(x_rgb_0_cat)
        rgb_out = self.upsample_4(self.rgb_conv_out(x_rgb_02))
        x_d_0_cat = torch.cat([dep_0, self.upsample_2(x_d_12)], dim=1)
        x_d_02 = self.d_gcm_0(x_d_0_cat)
        d_out = self.upsample_4(self.d_conv_out(x_d_02))
        agg1 = self.fam1(x_rgb_02, x_d_02, x_ful_12, ful_1, ful_0)
        x_ful_02 = self.ful_gcm_0(agg1, edge,f_edge)

        ful_out = self.upsample_4(self.ful_conv_out(x_ful_02))
        edge_out = self.upsample_4(edge)
        ful_out2 = self.upsample_8(self.ful_conv_out1(x_ful_12))
        ful_out3 = self.upsample_16(self.ful_conv_out2(x_ful_22))
        ful_out4 = self.upsample_32(self.ful_conv_out3(x_ful_32))
        ful_out5 = self.upsample_32(self.ful_conv_out3(x_ful_42))

        ful_out, d_out, rgb_out,  ful_out2, ful_out3, ful_out4,ful_out5=self.sigmoid(ful_out),self.sigmoid(d_out),self.sigmoid(rgb_out),self.sigmoid(ful_out2),self.sigmoid(ful_out3),self.sigmoid(ful_out4),self.sigmoid(ful_out5)
        return ful_out, d_out, rgb_out,edge_out,ful_out2,ful_out3,ful_out4,ful_out5

# Test code

if __name__ == '__main__':
    rgb = torch.rand((2, 3, 224, 224)).cuda()
    depth = torch.rand((2, 1, 224, 224)).cuda()
    model = DMENet(32).cuda()
    l = model(rgb, depth)

    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))
    print(l[0].size(), l[1].size(), l[2].size(),l[3].size())