import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def conv_block(in_dim,out_dim,act_fn,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
    )
    return model
    
class Conv_residual_conv(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class Fusionnet(nn.Module):

    def __init__(self, args):
        super(Fusionnet, self).__init__()

        self.in_dim = args.input_dim
        self.out_dim = args.num_feature
        self.final_out_dim = args.output_dim
        self.out_clamp = args.out_clamp

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ELU(inplace=True)

        # encoder
        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = conv_block(self.out_dim, self.out_dim, act_fn, 2)
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = conv_block(self.out_dim * 2, self.out_dim * 2, act_fn, 2)
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = conv_block(self.out_dim * 4, self.out_dim * 4, act_fn, 2)
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = conv_block(self.out_dim * 8, self.out_dim * 8, act_fn, 2)

        # bridge
        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn)

        # decoder
        self.deconv_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.deconv_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.deconv_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 1, act_fn_2)
        self.deconv_4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output
        self.out1 = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out2 = nn.Sigmoid()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)
                
        print("------KMAR-Net Init Done------")

    def forward(self, input):        
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        out = self.out1(up_4)
        out = self.out2(out)

        if self.out_clamp is not None:
            out = torch.clamp(out, min=self.out_clamp[0], max=self.out_clamp[1])

        return out
    

if __name__ == "__main__":
    input_ = torch.randn(4, 1, 512, 512)

    Arg = namedtuple('Arg', ['input_dim', 'num_feature', 'output_dim', 'out_clamp'])
    args = Arg(1, 16, 1, None)

    m  = Fusionnet(args)
    output = m(input_)
    print("output shape : ", output.shape)
