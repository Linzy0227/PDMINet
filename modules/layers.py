import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        ) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x) 
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads 
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_m1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_m2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_m3 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_m4 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj3 = nn.Linear(dim, dim)
        self.proj4 = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, m1, m2, m3, m4):
        B, N, C = m1.shape
        qkv_m1 = (
            self.qkv_m1(m1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        qkv_m2 = (
            self.qkv_m2(m2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        qkv_m3 = (
            self.qkv_m3(m3).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        qkv_m4 = (
            self.qkv_m4(m4).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        q_m1, k_m1, v_m1 = (
            qkv_m1[0],
            qkv_m1[1],
            qkv_m1[2],
        )
        q_m2, k_m2, v_m2 = (
            qkv_m2[0],
            qkv_m2[1],
            qkv_m2[2],
        )
        q_m3, k_m3, v_m3 = (
            qkv_m3[0],
            qkv_m3[1],
            qkv_m3[2],
        )
        q_m4, k_m4, v_m4 = (
            qkv_m4[0],
            qkv_m4[1],
            qkv_m4[2],
        )

        attn_m2_m1 = (q_m1 @ k_m2.transpose(-2, -1)) * self.scale
        attn_m2_m1 = attn_m2_m1.softmax(dim=-1)
        attn_m2_m1 = self.attn_drop(attn_m2_m1)

        x_m2_m1 = (attn_m2_m1 @ v_m2).transpose(1, 2).reshape(B, N, C)
        x_m2_m1 = self.proj1(x_m2_m1)
        x_m2_m1 = self.proj_drop(x_m2_m1)

        attn_m3_m1 = (q_m1 @ k_m3.transpose(-2, -1)) * self.scale
        attn_m3_m1 = attn_m3_m1.softmax(dim=-1)
        attn_m3_m1 = self.attn_drop(attn_m3_m1)

        x_m3_m1 = (attn_m3_m1 @ v_m3).transpose(1, 2).reshape(B, N, C)
        x_m3_m1 = self.proj1(x_m3_m1)
        x_m3_m1 = self.proj_drop(x_m3_m1)

        attn_m4_m1 = (q_m1 @ k_m4.transpose(-2, -1)) * self.scale
        attn_m4_m1 = attn_m4_m1.softmax(dim=-1)
        attn_m4_m1 = self.attn_drop(attn_m4_m1)

        x_m4_m1 = (attn_m4_m1 @ v_m4).transpose(1, 2).reshape(B, N, C)
        x_m4_m1 = self.proj1(x_m4_m1)
        x_m4_m1 = self.proj_drop(x_m4_m1)

        attn_m1_m2 = (q_m2 @ k_m1.transpose(-2, -1)) * self.scale
        attn_m1_m2 = attn_m1_m2.softmax(dim=-1)
        attn_m1_m2 = self.attn_drop(attn_m1_m2)

        x_m1_m2 = (attn_m1_m2 @ v_m1).transpose(1, 2).reshape(B, N, C)
        x_m1_m2 = self.proj2(x_m1_m2)
        x_m1_m2 = self.proj_drop(x_m1_m2)

        attn_m3_m2 = (q_m2 @ k_m3.transpose(-2, -1)) * self.scale
        attn_m3_m2 = attn_m3_m2.softmax(dim=-1)
        attn_m3_m2 = self.attn_drop(attn_m3_m2)

        x_m3_m2 = (attn_m3_m2 @ v_m3).transpose(1, 2).reshape(B, N, C)
        x_m3_m2 = self.proj2(x_m3_m2)
        x_m3_m2 = self.proj_drop(x_m3_m2)

        attn_m4_m2 = (q_m2 @ k_m4.transpose(-2, -1)) * self.scale
        attn_m4_m2 = attn_m4_m2.softmax(dim=-1)
        attn_m4_m2 = self.attn_drop(attn_m4_m2)

        x_m4_m2 = (attn_m4_m2 @ v_m4).transpose(1, 2).reshape(B, N, C)
        x_m4_m2 = self.proj2(x_m4_m2)
        x_m4_m2 = self.proj_drop(x_m4_m2)

        attn_m1_m3 = (q_m3 @ k_m1.transpose(-2, -1)) * self.scale
        attn_m1_m3 = attn_m1_m3.softmax(dim=-1)
        attn_m1_m3 = self.attn_drop(attn_m1_m3)

        x_m1_m3 = (attn_m1_m3 @ v_m1).transpose(1, 2).reshape(B, N, C)
        x_m1_m3 = self.proj3(x_m1_m3)
        x_m1_m3 = self.proj_drop(x_m1_m3)

        attn_m2_m3 = (q_m3 @ k_m2.transpose(-2, -1)) * self.scale
        attn_m2_m3 = attn_m2_m3.softmax(dim=-1)
        attn_m2_m3 = self.attn_drop(attn_m2_m3)

        x_m2_m3 = (attn_m2_m3 @ v_m2).transpose(1, 2).reshape(B, N, C)
        x_m2_m3 = self.proj3(x_m2_m3)
        x_m2_m3 = self.proj_drop(x_m2_m3)

        attn_m4_m3 = (q_m3 @ k_m4.transpose(-2, -1)) * self.scale
        attn_m4_m3 = attn_m4_m3.softmax(dim=-1)
        attn_m4_m3 = self.attn_drop(attn_m4_m3)

        x_m4_m3 = (attn_m4_m3 @ v_m4).transpose(1, 2).reshape(B, N, C)
        x_m4_m3 = self.proj3(x_m4_m3)
        x_m4_m3 = self.proj_drop(x_m4_m3)

        attn_m1_m4 = (q_m4 @ k_m1.transpose(-2, -1)) * self.scale
        attn_m1_m4 = attn_m1_m4.softmax(dim=-1)
        attn_m1_m4 = self.attn_drop(attn_m1_m4)

        x_m1_m4 = (attn_m1_m4 @ v_m1).transpose(1, 2).reshape(B, N, C)
        x_m1_m4 = self.proj4(x_m1_m4)
        x_m1_m4 = self.proj_drop(x_m1_m4)

        attn_m2_m4 = (q_m4 @ k_m2.transpose(-2, -1)) * self.scale
        attn_m2_m4 = attn_m2_m4.softmax(dim=-1)
        attn_m2_m4 = self.attn_drop(attn_m2_m4)

        x_m2_m4 = (attn_m2_m4 @ v_m2).transpose(1, 2).reshape(B, N, C)
        x_m2_m4 = self.proj4(x_m2_m4)
        x_m2_m4 = self.proj_drop(x_m2_m4)

        attn_m3_m4 = (q_m4 @ k_m3.transpose(-2, -1)) * self.scale
        attn_m3_m4 = attn_m3_m4.softmax(dim=-1)
        attn_m3_m4 = self.attn_drop(attn_m3_m4)

        x_m3_m4 = (attn_m3_m4 @ v_m3).transpose(1, 2).reshape(B, N, C)
        x_m3_m4 = self.proj4(x_m3_m4)
        x_m3_m4 = self.proj_drop(x_m3_m4)

        return x_m2_m1 + x_m3_m1 + x_m4_m1, x_m1_m2 + x_m3_m2 + x_m4_m2, x_m1_m3 + x_m2_m3 + x_m4_m3, x_m1_m4 + x_m2_m4 + x_m3_m4


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class CrossResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, m1, m2, m3, m4):
        m1_attn, m2_attn, m3_attn, m4_attn = self.fn(m1, m2, m3, m4)
        return m1_attn + m1, m2_attn + m2, m3_attn + m3, m4_attn + m4

class CrossPreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, m1, m2, m3, m4):
        m1 = self.norm1(m1)
        m2 = self.norm2(m2)
        m3 = self.norm3(m3)
        m4 = self.norm4(m4)
        m1_hybrid, m2_hybrid, m3_hybrid, m4_hybrid = self.fn(m1, m2, m3, m4)
        return self.dropout(m1_hybrid), self.dropout(m2_hybrid), self.dropout(m3_hybrid), self.dropout(m4_hybrid)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Attn(nn.Module):
    def __init__(self, embedding_dim=512, depth=1, heads=8, mlp_dim=4096, dropout_rate=0.1):
        super(Attn, self).__init__()
        self.cross_attention_list = nn.ModuleList()
        self.cross_ffn_list = nn.ModuleList()
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )

    def forward(self, x):
        for attn in self.cross_attention_list:
            x = attn(x)
        return x


class coTransformer(nn.Module):
    def __init__(self, embedding_dim=512, depth=1, heads=8, mlp_dim=4096, dropout_rate=0.1):
        super(coTransformer, self).__init__()
        self.self_attention_list1 = []
        self.self_attention_list2 = []
        self.self_attention_list3 = []
        self.self_attention_list4 = []
        self.cross_attention_list = []
        self.ffn_list1 = []
        self.ffn_list2 = []
        self.ffn_list3 = []
        self.ffn_list4 = []
        self.depth = depth
        for j in range(self.depth):
            self.self_attention_list1.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.self_attention_list2.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.self_attention_list3.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.self_attention_list4.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )

            self.cross_attention_list.append(
                CrossResidual(
                    CrossPreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        CrossAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.ffn_list1.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )
            self.ffn_list2.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )
            self.ffn_list3.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )
            self.ffn_list4.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.self_attention_list1 = nn.ModuleList(self.self_attention_list1)
        self.self_attention_list2 = nn.ModuleList(self.self_attention_list2)
        self.self_attention_list3 = nn.ModuleList(self.self_attention_list3)
        self.self_attention_list4 = nn.ModuleList(self.self_attention_list4)
        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.ffn_list1 = nn.ModuleList(self.ffn_list1)
        self.ffn_list2 = nn.ModuleList(self.ffn_list2)
        self.ffn_list3 = nn.ModuleList(self.ffn_list3)
        self.ffn_list4 = nn.ModuleList(self.ffn_list4)


    def forward(self, m1, m2, m3, m4, pos1, pos2, pos3, pos4):
        for j in range(self.depth):
            m1 = m1 + pos1
            m2 = m2 + pos2
            m3 = m3 + pos3
            m4 = m4 + pos4

            m1 = self.self_attention_list1[j](m1)
            m2 = self.self_attention_list2[j](m2)
            m3 = self.self_attention_list3[j](m3)
            m4 = self.self_attention_list4[j](m4)

            m_2m1, m_2m2, m_2m3, m_2m4 = self.cross_attention_list[j](m1, m2, m3, m4)

            m1_spe = self.ffn_list1[j](m_2m1)
            m2_spe = self.ffn_list2[j](m_2m2)
            m3_spe = self.ffn_list3[j](m_2m3)
            m4_spe = self.ffn_list4[j](m_2m4)
        return m1_spe, m2_spe, m3_spe, m4_spe


