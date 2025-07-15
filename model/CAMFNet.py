import torch.nn as nn
import torch.nn.functional as F
from simplecv.interface import CVModule
from simplecv.module import SEBlock
from simplecv import registry
import torch
import math
from model.Vim import SS2D
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )

def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,dilation):

        super(depthwise_separable_conv, self).__init__()


        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, dilation=dilation,padding=dilation, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class HSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer=nn.BatchNorm2d,
        attn_drop_rate: float = 0,
        d_state: int = 16,
		size: int = 1,
		scan_type='scan',
		num_direction=4,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

def repeat_block(block_channel, r, n):
    layers = [
        nn.Sequential(
            SEBlock(block_channel, r),
            conv3x3_gn_relu(block_channel, block_channel, r)
        )
        for _ in range(n)]
    return nn.Sequential(*layers)

class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()
        self.dim=dim
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.MSC = ASPPModule(dim, atrous_rates=[6,12,18])
        self.att=HSSBlock(hidden_dim=dim, attn_drop_rate=0., d_state=16)
        self.SP=SpatialAttention()

    def forward(self, x, res):
        x=self.MSC(x)
        x=self.att(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.post_conv(x)
        return x+res

@registry.MODEL.register('CAMFNet')
class CAMFNet(nn.Module):
    def __init__(self, config):
        super(CAMFNet, self).__init__()
        self.config=config
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r

        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.top_down=nn.ModuleList([
            Fusion(inner_dim),
            Fusion(inner_dim),
            Fusion(inner_dim),
            Fusion(inner_dim),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.Th=nn.Sigmoid()

    def forward(self, x, y=None, w=None, **kwargs):
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        # Reverse the list
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down[i](out_feat_list[i], inner_feat_list[i + 1])
            out = self.fuse_3x3convs[i](inner)
            out_feat_list.append(out)
        final_feat = out_feat_list[-1]
        logit = self.cls_pred_conv(final_feat)
        return torch.softmax(logit, dim=1)

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', depthwise_separable_conv(nin=inter_channels,nout=out_channels,dilation=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features

class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = atrous_rates

        self.b0 = _DenseASPPConv(in_channels,out_channels//2,out_channels,rate1,0.1)
        self.b1 = _DenseASPPConv(in_channels+out_channels*1,out_channels//2,out_channels,rate2,0.1)
        self.b2 = _DenseASPPConv(in_channels+out_channels*2,out_channels//2,out_channels,rate3,0.1)

        self.project = nn.Sequential(nn.Conv2d(in_channels+3 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        aspp1=self.b0(x)
        x=torch.cat([aspp1,x],dim=1)

        aspp2=self.b1(x)
        x=torch.cat([aspp2,x],dim=1)

        aspp3=self.b2(x)
        x=torch.cat([aspp3,x],dim=1)

        return self.project(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x.clone()
        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out)

        out2 = self.relu1(out1)

        out = self.sigmoid(out2)

        y = x * out.view(out.size(0), 1, out.size(-2), out.size(-1))
        return residual+y

if __name__ == '__main__':
    from thop import profile
    x = torch.randn(2, 6, 32, 32).cuda()
    import argparse
    parser = argparse.ArgumentParser(description='train FreeNet')
    parser.add_argument('--in_channels', type=int, default=6,)
    parser.add_argument('--num_classes', type=int, default=2,)
    parser.add_argument('--block_channels', type=tuple, default=(96, 128, 192, 256),)
    parser.add_argument('--reduction_ratio', type=float, default=1.0,)
    parser.add_argument('--inner_dim', type=int, default=128,)
    parser.add_argument('--num_blocks', type=tuple, default=(1, 1, 1, 1),)
    args = parser.parse_args()
    net = CAMFNet(args).cuda()
    out = net(x)
    print(out.shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)