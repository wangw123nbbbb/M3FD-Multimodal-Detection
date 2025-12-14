import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite, DropPath, to_2tuple
from ..modules.attention import *
from ..modules.block import *
from ..modules.conv import *


# ============================================= C3k2_RVB_EMA begin =============================================
# 定义一个组合了卷积层和批归一化层的序列模块
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        # 添加卷积层，a是输入通道数，b是输出通道数
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # 添加批归一化层
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        # 初始化批归一化层的权重
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        # 初始化批归一化层的偏置为0
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()  # 禁用梯度计算
    def fuse_self(self):
        # 获取卷积层和批归一化层
        c, bn = self._modules.values()
        # 计算融合后的权重
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        # 计算融合后的偏置
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        # 创建新的卷积层
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        # 复制融合后的权重和偏置
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# 定义一个残差结构的包装类
class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn  # 保存传入的子模块

    def forward(self, x):
        # 前向传播时，输出为fn(x) + x，实现残差连接
        return self.fn(x) + x


# 定义RepViT的基本块
class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, use_se=True):
        super(RepViTBlock, self).__init__()

        self.identity = inp == oup  # 判断输入输出通道是否一致
        hidden_dim = 2 * inp  # 隐藏层通道数为输入通道的2倍

        # token_mixer用于混合空间信息，包含RepVGGDW和可选的SE注意力
        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),  # 深度可分离卷积
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),  # SE注意力或恒等映射
        )
        # channel_mixer用于混合通道信息，包含1x1卷积、激活、再1x1卷积
        self.channel_mixer = Residual(nn.Sequential(
            # pw，1x1卷积，升维
            Conv2d_BN(inp, hidden_dim, 1, 1, 0),
            nn.GELU(),  # GELU激活函数
            # pw-linear，1x1卷积，降维，BN权重初始化为0
            Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
        ))

    def forward(self, x):
        # 先通过token_mixer，再通过channel_mixer
        return self.channel_mixer(self.token_mixer(x))


# 带EMA注意力的RepViTBlock
class RepViTBlock_EMA(RepViTBlock):
    def __init__(self, inp, oup, use_se=True):
        super().__init__(inp, oup, use_se)

        # 用EMA注意力替换SE
        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            EMA(inp) if use_se else nn.Identity(),  # EMA注意力或恒等映射
        )


# C3k结构的变体，使用RepViTBlock
class C3k_RVB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # 计算隐藏通道数
        # 用n个RepViTBlock堆叠
        self.m = nn.Sequential(*(RepViTBlock(c_, c_, False) for _ in range(n)))


# C3k2结构的变体，使用C3k_RVB或RepViTBlock
class C3k2_RVB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # 如果c3k为True，用C3k_RVB，否则用RepViTBlock
        self.m = nn.ModuleList(
            C3k_RVB(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock(self.c, self.c, False) for _ in range(n))


# C3k结构的变体，使用带SE的RepViTBlock
class C3k_RVB_SE(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # 计算隐藏通道数
        # 用n个带SE的RepViTBlock堆叠
        self.m = nn.Sequential(*(RepViTBlock(c_, c_) for _ in range(n)))


# C3k2结构的变体，使用C3k_RVB_SE或RepViTBlock
class C3k2_RVB_SE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # 如果c3k为True，用C3k_RVB_SE，否则用RepViTBlock
        self.m = nn.ModuleList(
            C3k_RVB_SE(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock(self.c, self.c) for _ in range(n))


# C3k结构的变体，使用带EMA的RepViTBlock
class C3k_RVB_EMA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # 计算隐藏通道数
        # 用n个带EMA的RepViTBlock堆叠
        self.m = nn.Sequential(*(RepViTBlock_EMA(c_, c_) for _ in range(n)))


# C3k2结构的变体，使用C3k_RVB_EMA或RepViTBlock_EMA
class C3k2_RVB_EMA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # 如果c3k为True，用C3k_RVB_EMA，否则用RepViTBlock_EMA
        self.m = nn.ModuleList(
            C3k_RVB_EMA(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock_EMA(self.c, self.c) for _ in range(n))


# ============================================= C3k2_RVB_EMA end =============================================

# ============================================= C3k2_AdditiveBlock_CGLU begin =============================================
class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class Mlp_CASVIT(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class LocalIntegration(nn.Module):
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """

    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


class AdditiveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_CASVIT(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AdditiveBlock_CGLU(AdditiveBlock):
    def __init__(self, dim, mlp_ratio=4, attn_bias=False, drop=0, drop_path=0, act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d):
        super().__init__(dim, mlp_ratio, attn_bias, drop, drop_path, act_layer, norm_layer)
        self.mlp = ConvolutionalGLU(dim)


class C3k_AdditiveBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdditiveBlock(c_) for _ in range(n)))


class C3k2_AdditiveBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_AdditiveBlock(self.c, self.c, 2, shortcut, g) if c3k else AdditiveBlock(self.c) for _ in range(n))


class C3k_AdditiveBlock_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdditiveBlock_CGLU(c_) for _ in range(n)))


class C3k2_AdditiveBlock_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_AdditiveBlock_CGLU(self.c, self.c, 2, shortcut, g) if c3k else AdditiveBlock_CGLU(self.c) for _ in
            range(n))


# ============================================= C3k2_AdditiveBlock_CGLU end =============================================

# ============================================= C3k2_DWR_DRB begin =============================================
def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (
            conv_bias - bn.running_mean) * bn.weight / std


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        # This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class DWR_DRB(nn.Module):
    def __init__(self, dim, act=True) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3, act=act)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1, act=act)
        self.conv_3x3_d3 = DilatedReparamBlock(dim // 2, 5)
        self.conv_3x3_d5 = DilatedReparamBlock(dim // 2, 7)

        self.conv_1x1 = Conv(dim * 2, dim, k=1, act=act)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class C3k_DWR_DRB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWR_DRB(c_) for _ in range(n)))


class C3k2_DWR_DRB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DWR_DRB(self.c, self.c, 2, shortcut, g) if c3k else DWR_DRB(self.c) for _ in range(n))

# ============================================= C3k2_DWR_DRB end =============================================
