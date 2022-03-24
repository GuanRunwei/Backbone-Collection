import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial
from collections import OrderedDict
import math


device = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------------------------------#
#   GELU Activation
# --------------------------------------#
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ----------------------------------------#
# Patch嵌入
# ----------------------------------------#
class PatchEmbed(nn.Module):
    def __init__(self, input_shape=[224, 224], patch_size=16, in_chans=3, num_features=768, norm_layer=None,
                 flatten=True):
        super(PatchEmbed, self).__init__()
        # 14 * 14
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        # batch, 3, 224, 224 -> batch, 768, 14, 14
        x = self.proj(x)

        # batch, 768, 14, 14 -> batch, 768, 196 -> batch, 196, 768
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)

        return x


# ----------------------------------------#
# 时隔一整年 终于开始写自己的Attention了
# ----------------------------------------#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        # 768 -> 768*3
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # batch, 196, 768
        B, N, C = x.shape

        # batch, 196, 768 -> batch, 196, 768*3 -> batch, 196, 3, 8, 768/8=96 -> 3, batch, 8, 196, 96
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 3 * 1, batch, 8, 196, 96  -> q, k, v = batch: 16, head: 8, patch: 196, each_head_attention_channels: 96
        q, k, v = qkv[0], qkv[1], qkv[2]

        # batch, 8, 196, 96 @ batch, 8, 96, 196 -> batch, 8, 196, 196
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # batch, 8, 196, 196 @ batch, 8, 196, 96 -> batch, 8, 196, 96 -> batch, 196, 8, 96 -> batch, 196, 768
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # batch, 196, 768 -> batch, 196, 768
        x = self.proj(x)

        # Dropout(batch, 196, 768)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        # batch, 196, 768 -> batch, 196, 768
        x = self.fc1(x)

        # batch, 196, 768 -> batch, 196, 768
        x = self.act(x)

        # batch, 196, 768 -> batch, 196, 768
        x = self.drop1(x)

        # batch, 196, 768 -> batch, 768, 1000
        x = self.fc2(x)

        # batch, 196, 768 -> batch, 768, 1000
        x = self.drop2(x)

        # batch, 196, 768 -> batch, 768, 1000
        return x


# -----------------------------------------------#
#   一个Transformer Encoder Block
# -----------------------------------------------#
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # res1 = batch, 196, 768 + Dropout(Mix_Attention(LayerNorm(batch, 196, 768))) -> batch, 196, 768
        x += self.drop_path(self.attn(self.norm1(x)))

        # res2 = res1 + Dropout(MLP(LayerNorm(res1))) -> batch, 196, 768
        x += self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self, input_shape=[224, 224], patch_size=16, in_chans=3, num_classes=1000, num_features=768,
            depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU
    ):
        super(VisionTransformer, self).__init__()
        # -----------------------------------------------#
        #   batch, 224, 224, 3 -> batch, 196, 768
        # -----------------------------------------------#
        self.patch_embed = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans,
                                      num_features=num_features)
        num_patches = (224 // patch_size) * (224 // patch_size)
        self.num_features = num_features
        self.new_feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]
        self.old_feature_shape = [int(224 // patch_size), int(224 // patch_size)]

        # --------------------------------------------------------------------------------------------------------------------#
        #   classtoken部分是transformer的分类特征。用于堆叠到序列化后的图片特征中，作为一个单位的序列特征进行特征提取。
        #
        #   在利用步长为16x16的卷积将输入图片划分成14x14的部分后，将14x14部分的特征平铺，一幅图片会存在序列长度为196的特征。
        #   此时生成一个classtoken，将classtoken堆叠到序列长度为196的特征上，获得一个序列长度为197的特征。
        #   在特征提取的过程中，classtoken会与图片特征进行特征的交互。最终分类时，我们取出classtoken的特征，利用全连接分类。
        # --------------------------------------------------------------------------------------------------------------------#
        #   1, 1, 768
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))
        # --------------------------------------------------------------------------------------------------------------------#
        #   为网络提取到的特征添加上位置信息。
        #   以输入图片为224, 224, 3为例，我们获得的序列化后的图片特征为196, 768。加上classtoken后就是197, 768
        #   此时生成的pos_Embedding的shape也为197, 768，代表每一个特征的位置信息。
        # --------------------------------------------------------------------------------------------------------------------#
        #   1, 197, 768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, num_features))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # -----------------------------------------------#
        #   12次 batch, 197, 768 -> batch, 197, 768
        # -----------------------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )
        # 第一个norm写在了PatchEmbed里
        self.norm = norm_layer(num_features)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # batch, 3, 224, 224 -> batch, 196, 768
        x = self.patch_embed(x)

        # 1, 1, 768 -> batch, 1, 768
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # batch, 1, 768 + batch, 196, 768 -> batch, 197, 768
        x = torch.cat((cls_token, x), dim=1)

        # 1, 1, 768
        cls_token_pe = self.pos_embed[:, 0:1, :]

        # 1, 196, 768
        img_token_pe = self.pos_embed[:, 1:, :]

        # 1, 196, 768 -> 1, 14, 14, 768 -> 1, 768, 14, 14
        img_token_pe = img_token_pe.view(1, *self.old_feature_shape, -1).permute(0, 3, 1, 2)

        # 1, 768, 14, 14 -> 以防输入图片大小不是224*224 做插值 -> 1, 768, 14, 14
        img_token_pe = F.interpolate(img_token_pe, size=self.new_feature_shape, mode='bicubic', align_corners=False)

        # 1， 768， 14， 14 -> 1, 14, 14, 768 -> 1, 196, 768
        img_token_pe = img_token_pe.permute(0, 2, 3, 1).flatten(1, 2)

        # 1, 1, 768 + 1, 196, 768 -> 1, 197, 768
        pos_embed = torch.cat([cls_token_pe, img_token_pe], dim=1)

        # batch, 197, 768 + 1, 197, 768 -> batch, 197, 768
        x = self.pos_drop(x + pos_embed)

        # 12次 batch, 197, 768
        x = self.blocks(x)

        # batch, 197, 768
        x = self.norm(x)

        # batch, 768
        return x[:, 0]

    def forward(self, x):
        # 整个Transformer Encoder = batch, 768
        x = self.forward_features(x)
        # 最后的MLP Header = batch, 768 -> 768 -> 1000 -> batch, 1000
        x = self.head(x)
        return x

    def freeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = False
            except:
                module.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = True
            except:
                module.requires_grad = True


def vit(input_shape=[224, 224], pretrained=False, num_classes=1000):
    model = VisionTransformer(input_shape).to(device)
    if pretrained:
        model.load_state_dict(torch.load("model_data/vit-patch_16.pth"))

    if num_classes != 1000:
        model.head = nn.Linear(model.num_features, num_classes)
    return model


if __name__ == '__main__':
    vit_model = vit(pretrained=True)
    print(vit_model)



