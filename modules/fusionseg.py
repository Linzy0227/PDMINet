import torch.nn as nn
import sys
sys.path.append('./')
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math

from modules.net import Fusion, FusionExpert, TumorPredict, EnBlock, FusionModal, Expert
from modules.layers import coTransformer, Attn


class Encoder(nn.Module):
    def __init__(self, in_channels=1, basic_dims=8):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=in_channels, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, bias=True)
        self.e1_c2 = EnBlock(basic_dims, basic_dims)
        self.e1_c3 = EnBlock(basic_dims, basic_dims)

        self.e2_c1 = EnBlock(basic_dims, basic_dims * 2, stride=2)
        self.e2_c2 = EnBlock(basic_dims * 2, basic_dims * 2)
        self.e2_c3 = EnBlock(basic_dims * 2, basic_dims * 2)

        self.e3_c1 = EnBlock(basic_dims * 2, basic_dims * 4, stride=2)
        self.e3_c2 = EnBlock(basic_dims * 4, basic_dims * 4)
        self.e3_c3 = EnBlock(basic_dims * 4, basic_dims * 4)

        self.e4_c1 = EnBlock(basic_dims * 4, basic_dims * 8, stride=2)
        self.e4_c2 = EnBlock(basic_dims * 8, basic_dims * 8)
        self.e4_c3 = EnBlock(basic_dims * 8, basic_dims * 8)

        self.e5_c1 = EnBlock(basic_dims * 8, basic_dims * 16, stride=2)
        self.e5_c2 = EnBlock(basic_dims * 16, basic_dims * 16)
        self.e5_c3 = EnBlock(basic_dims * 16, basic_dims * 16)

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5

class Decoder_shared(nn.Module):
    def __init__(self, num_cls=4, basic_dims=8):
        super(Decoder_shared, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = EnBlock(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = EnBlock(basic_dims * 16, basic_dims * 8)
        self.d4_out = EnBlock(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = EnBlock(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = EnBlock(basic_dims * 8, basic_dims * 4)
        self.d3_out = EnBlock(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = EnBlock(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = EnBlock(basic_dims * 4, basic_dims * 2)
        self.d2_out = EnBlock(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = EnBlock(basic_dims * 2, basic_dims)
        self.d1_c2 = EnBlock(basic_dims * 2, basic_dims)
        self.d1_out = EnBlock(basic_dims, basic_dims, k_size=1, padding=0)

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred


class DecoderExpert(nn.Module):
    def __init__(self, num_cls=4, basic_dims=8):
        super(DecoderExpert, self).__init__()

        self.d4_c1 = EnBlock(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = EnBlock(basic_dims * 16, basic_dims * 8)
        self.d4_out = EnBlock(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3_c1 = EnBlock(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = EnBlock(basic_dims * 8, basic_dims * 4)
        self.d3_out = EnBlock(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2_c1 = EnBlock(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = EnBlock(basic_dims * 4, basic_dims * 2)
        self.d2_out = EnBlock(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1_c1 = EnBlock(basic_dims * 2, basic_dims)
        self.d1_c2 = EnBlock(basic_dims * 2, basic_dims)
        self.d1_out = EnBlock(basic_dims, basic_dims, k_size=1, padding=0)

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.fusion5 = FusionModal(in_channel=basic_dims * 16, num_cls=num_cls)
        self.fusionExp4 = FusionExpert(in_channel=basic_dims*8, num_cls=num_cls)
        self.fusionExp3 = FusionExpert(in_channel=basic_dims*4, num_cls=num_cls)
        self.fusionExp2 = FusionExpert(in_channel=basic_dims*2, num_cls=num_cls)
        self.fusionExp1 = FusionExpert(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, x5, expert_know):
        de_x5 = self.fusion5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.fusionExp4(x4, expert_know[3])
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.fusionExp3(x3, expert_know[2])
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.fusionExp2(x2, expert_know[1])
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.fusionExp1(x1, expert_know[0])
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class ConModal(nn.Module):
    def __init__(self):
        super(ConModal, self).__init__()
    
    def forward(self, x):
        B, K, C, H, W, Z = x.size()
        x = x.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=4, transformer_basic_dims=512, basic_dims=8, depth=4, num_heads=8, mlp_dim=4096, patch_size=8):
        super(Model, self).__init__()

        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.expert_encoder = Encoder(in_channels=4)
        self.expert = Expert()

        self.flair_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.co_transformer = coTransformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)

        self.fusion = Fusion()

        self.masker = ConModal()
        self.decoder_fuse_expert = DecoderExpert(num_cls=num_cls)
        self.decoder_shared = Decoder_shared(num_cls=num_cls)
        self.decoder_expert = Decoder_shared(num_cls=num_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        exp_x1, exp_x2, exp_x3, exp_x4, exp_x5 = self.expert_encoder(x)
        expert_out = self.decoder_shared(exp_x1, exp_x2, exp_x3, exp_x4, exp_x5)
        expert_know, expert_masks = self.expert(exp_x1, exp_x2, exp_x3, exp_x4)

        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, 512)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, 512)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, 512)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, 512)

        flair_intra_token_x5, t1ce_intra_token_x5, t1_intra_token_x5, t2_intra_token_x5 = self.co_transformer(flair_token_x5, t1ce_token_x5, t1_token_x5, t2_token_x5, self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos)

        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), 8, 8, 8, 512).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), 8, 8, 8, 512).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), 8, 8, 8, 512).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), 8, 8, 8, 512).permute(0, 4, 1, 2, 3).contiguous()

        feat_total_x5 = torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1)

        fusion_out, feat_corrs, tumor_masks = self.fusion(feat_total_x5)

        flair_pred = self.decoder_shared(flair_x1, flair_x2, flair_x3, flair_x4, feat_corrs[0])
        t1ce_pred = self.decoder_shared(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, feat_corrs[1])
        t1_pred = self.decoder_shared(t1_x1, t1_x2, t1_x3, t1_x4, feat_corrs[2])
        t2_pred = self.decoder_shared(t2_x1, t2_x2, t2_x3, t2_x4, feat_corrs[3])

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1))
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1))
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1))
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1))

        final_out, deep_preds = self.decoder_fuse_expert(x1, x2, x3, x4, fusion_out, expert_know)

        share_decoder_out = (flair_pred, t1ce_pred, t1_pred, t2_pred)

        return final_out, expert_out, share_decoder_out, deep_preds, expert_masks, tumor_masks



