from tkinter import E
import torch
import torch.nn as nn
import torch.nn.init

from modules.layers import Attn

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8


def normalization(planes, norm='in'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError("Does not support this kind of norm.")
    return m


class EnBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', act_type='lrelu', relufactor=0.2):
        super(EnBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', act_type='lrelu',
                 relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class TumorPredict(nn.Module):
    def __init__(self, in_channels=64, norm='in', num_cls=4):
        super(TumorPredict, self).__init__()

        self.tumor_mask = nn.Sequential(
            general_conv3d(in_channels, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1))

    def forward(self, x):
        pre_mask = self.tumor_mask(x)
        return pre_mask


class TumorFusion(nn.Module):
    def __init__(self, in_channels=64, norm='in', num_cls=4):
        super(TumorFusion, self).__init__()

        self.tumor_out = nn.Sequential(
            general_conv3d(in_channels * 3, int(in_channels // 4), k_size=1, padding=0, stride=1),
            general_conv3d(int(in_channels // 4), int(in_channels // 4), k_size=3, padding=1, stride=1),
            general_conv3d(int(in_channels // 4), in_channels, k_size=1, padding=0, stride=1))

    def forward(self, x):
        fusion_out = self.tumor_out(x)
        return fusion_out


class Expert(nn.Module):
    def __init__(self, num_cls=4):
        super(Expert, self).__init__()
        self.num_cls = num_cls
        self.x1_mask = TumorPredict(in_channels=8, num_cls=4)
        self.x2_mask = TumorPredict(in_channels=16, num_cls=4)
        self.x3_mask = TumorPredict(in_channels=32, num_cls=4)
        self.x4_mask = TumorPredict(in_channels=64, num_cls=4)

        self.x1_merge = TumorFusion(in_channels=8)
        self.x2_merge = TumorFusion(in_channels=16)
        self.x3_merge = TumorFusion(in_channels=32)
        self.x4_merge = TumorFusion(in_channels=64)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4):

        _, c1, _, _, _ = x1.size()
        _, c2, _, _, _ = x2.size()
        _, c3, _, _, _ = x3.size()
        _, c4, _, _, _ = x4.size()

        x1_mask = self.x1_mask(x1)  # [1,4,128,128,128]
        x2_mask = self.x2_mask(x2)  # [1,4,64,64,64]
        x3_mask = self.x3_mask(x3)  # [1,4,32,32,32]
        x4_mask = self.x4_mask(x4)  # [1,4,16,16,16]

        x1_masks = torch.unsqueeze(x1_mask.detach(), 2).repeat(1, 1, c1, 1, 1, 1)
        x2_masks = torch.unsqueeze(x2_mask.detach(), 2).repeat(1, 1, c2, 1, 1, 1)
        x3_masks = torch.unsqueeze(x3_mask.detach(), 2).repeat(1, 1, c3, 1, 1, 1)
        x4_masks = torch.unsqueeze(x4_mask.detach(), 2).repeat(1, 1, c4, 1, 1, 1)

        x1_know = x1 * x1_masks
        x2_know = x2 * x2_masks
        x3_know = x3 * x3_masks
        x4_know = x4 * x4_masks

        x1_ex123 = [x1_know[:, i + 1, ...] for i in range(3)]
        x2_ex123 = [x2_know[:, i + 1, ...] for i in range(3)]
        x3_ex123 = [x3_know[:, i + 1, ...] for i in range(3)]
        x4_ex123 = [x4_know[:, i + 1, ...] for i in range(3)]

        x1_out = self.x1_merge(torch.cat([x1_ex123[i] for i in range(3)], dim=1))  # [1,8,128,128,128]
        x2_out = self.x2_merge(torch.cat([x2_ex123[i] for i in range(3)], dim=1))
        x3_out = self.x3_merge(torch.cat([x3_ex123[i] for i in range(3)], dim=1))
        x4_out = self.x4_merge(torch.cat([x4_ex123[i] for i in range(3)], dim=1))

        expert_know = (x1_out, x2_out, x3_out, x4_out)
        expert_masks = (x1_mask, self.up2(x2_mask), self.up4(x3_mask), self.up8(x4_mask))

        return expert_know, expert_masks


class FusionModal(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(FusionModal, self).__init__()
        self.fusion_layer = nn.Sequential(
            EnBlock(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            EnBlock(in_channel, in_channel, k_size=3, padding=1, stride=1),
            EnBlock(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)


class FusionExpert(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(FusionExpert, self).__init__()
        self.fusion_layer = nn.Sequential(
            EnBlock(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            EnBlock(in_channel, in_channel, k_size=3, padding=1, stride=1),
            EnBlock(in_channel, in_channel, k_size=1, padding=0, stride=1))
        self.conv = nn.Conv3d(in_channels=in_channel * 2, out_channels=in_channel, kernel_size=3, stride=1, padding=1,
                              bias=True)

    def forward(self, x_stack, x_expert):
        full_modal = self.fusion_layer(x_stack)
        out = self.conv(torch.cat((full_modal, x_expert), dim=1))
        return out



class TumorGuide(nn.Module):
    def __init__(self):
        super(TumorGuide, self).__init__()

        self.glo_fc1 = nn.Sequential(nn.Linear(512, 256),
                                     nn.ReLU()
                                     )
        self.glo_fc2 = nn.Sequential(nn.Linear(512, 256),
                                     nn.ReLU()
                                     )
        self.glo_fc3 = nn.Sequential(nn.Linear(512, 256),
                                     nn.ReLU()
                                     )
        self.corr_atte1 = nn.Sequential(
            nn.Conv3d(512 + 256, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(256),
            nn.Conv3d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 1, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(1)
        )
        self.corr_atte2 = nn.Sequential(
            nn.Conv3d(512 + 256, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(256),
            nn.Conv3d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 1, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(1)
        )
        self.corr_atte3 = nn.Sequential(
            nn.Conv3d(512 + 256, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(256),
            nn.Conv3d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 1, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(1)
        )
        self.tumor_fusion = nn.Sequential(
            general_conv3d(512 * 3, int(512 // 4), k_size=1, padding=0, stride=1),
            general_conv3d(int(512 // 4), int(512 // 4), k_size=3, padding=1, stride=1),
            general_conv3d(int(512 // 4), 512, k_size=1, padding=0, stride=1))

    def forward(self, x, tumor1, tumor2, tumor3):
        B, C, H, W, D = x.size()
        gap_1 = tumor1.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        gap_2 = tumor2.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        gap_3 = tumor3.mean(dim=-1).mean(dim=-1).mean(dim=-1)

        t1_expand = self.glo_fc1(gap_1).view(B, 256, 1, 1, 1).contiguous().expand(B, 256, H, W, D)
        t2_expand = self.glo_fc2(gap_2).view(B, 256, 1, 1, 1).contiguous().expand(B, 256, H, W, D)
        t3_expand = self.glo_fc3(gap_3).view(B, 256, 1, 1, 1).contiguous().expand(B, 256, H, W, D)

        t1_guide = torch.cat((x, t1_expand), dim=1)
        t2_guide = torch.cat((x, t2_expand), dim=1)
        t3_guide = torch.cat((x, t3_expand), dim=1)

        t1_corr_map = torch.sigmoid(self.corr_atte1(t1_guide))
        t2_corr_map = torch.sigmoid(self.corr_atte2(t2_guide))
        t3_corr_map = torch.sigmoid(self.corr_atte3(t3_guide))

        t1_corr = x * t1_corr_map
        t2_corr = x * t2_corr_map
        t3_corr = x * t3_corr_map

        out = self.tumor_fusion(torch.cat((t1_corr, t2_corr, t3_corr), dim=1))
        return out


class Fusion(nn.Module):
    def __init__(self, in_channel=512, num_cls=4):
        super(Fusion, self).__init__()
        self.num_cls = num_cls

        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.prm_flair = TumorPredict(in_channels=in_channel)
        self.prm_t1ce = TumorPredict(in_channels=in_channel)
        self.prm_t1 = TumorPredict(in_channels=in_channel)
        self.prm_t2 = TumorPredict(in_channels=in_channel)

        self.modal1 = TumorGuide()
        self.modal2 = TumorGuide()
        self.modal3 = TumorGuide()
        self.modal4 = TumorGuide()
        self.attention = Attn(embedding_dim=128)
        self.flair_decode_conv = nn.Conv3d(512, 8*16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(512, 8*16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(512, 8*16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(512, 8*16, kernel_size=1, stride=1, padding=0)
        self.multimodal_decode_conv = nn.Conv3d(512, 8*16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, 8**3, 128))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, 8**3, 128))
        self.t1_pos = nn.Parameter(torch.zeros(1, 8**3, 128))
        self.t2_pos = nn.Parameter(torch.zeros(1, 8**3, 128))

    def forward(self, x_stack):
        B, K, C, H, W, D = x_stack.size()
        flair = x_stack[:, 0:1, ...].view(B, -1, H, W, D)
        t1ce = x_stack[:, 1:2, ...].view(B, -1, H, W, D)
        t1 = x_stack[:, 2:3, ...].view(B, -1, H, W, D)
        t2 = x_stack[:, 3:4, ...].view(B, -1, H, W, D)

        flair_mask = self.prm_flair(flair)
        t1ce_mask = self.prm_t1ce(t1ce)
        t1_mask = self.prm_t1(t1)
        t2_mask = self.prm_t2(t2)

        flair_mask_expand = torch.unsqueeze(flair_mask.detach(), 2).repeat(1, 1, C, 1, 1, 1)
        t1ce_mask_expand = torch.unsqueeze(t1ce_mask.detach(), 2).repeat(1, 1, C, 1, 1, 1)
        t1_mask_expand = torch.unsqueeze(t1_mask.detach(), 2).repeat(1, 1, C, 1, 1, 1)
        t2_mask_expand = torch.unsqueeze(t2_mask.detach(), 2).repeat(1, 1, C, 1, 1, 1)

        flair_tumor = flair * flair_mask_expand
        t1ce_tumor = t1ce * t1ce_mask_expand
        t1_tumor = t1 * t1_mask_expand
        t2_tumor = t2 * t2_mask_expand

        flair_corr = self.modal1(flair, flair_tumor[:, 1, ...], flair_tumor[:, 2, ...], flair_tumor[:, 3, ...])
        t1ce_corr = self.modal2(t1ce, t1ce_tumor[:, 1, ...], t1ce_tumor[:, 2, ...], t1ce_tumor[:, 3, ...])
        t1_corr = self.modal3(t1, t1_tumor[:, 1, ...], t1_tumor[:, 2, ...], t1_tumor[:, 3, ...])
        t2_corr = self.modal4(t2, t2_tumor[:, 1, ...], t2_tumor[:, 2, ...], t2_tumor[:, 3, ...])

        flair_out = self.flair_decode_conv(flair_corr)
        t1ce_out = self.t1ce_decode_conv(t1ce_corr)
        t1_out = self.t1_decode_conv(t1_corr)
        t2_out = self.t2_decode_conv(t2_corr)

        multimodal_token = torch.cat(
            (flair_out.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, 128),
             t1ce_out.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, 128),
             t1_out.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, 128),
             t2_out.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, 128),
             ), dim=1)


        multimodal_token = self.attention(multimodal_token)
        multimodal_feat = multimodal_token.view(multimodal_token.size(0), 8, 8, 8, 512).permute(0, 4, 1, 2, 3).contiguous()

        return multimodal_feat, (flair_out, t1ce_out, t1_out, t2_out), (self.up16(flair_mask), self.up16(t1ce_mask), self.up16(t1_mask), self.up16(t2_mask))



