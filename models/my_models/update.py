import torch
import torch.nn as nn
import torch.nn.functional as F

from models import mhs_channel


def initialize_msra(modules):
    print("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

    

# from models.my_models.fd_decoder import TransBlocks, BasicLayer, SelfTransformerBlcok
class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convsf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convsf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64+128+192 + mhs_channel , 128-6, 3, padding=1) #  + mhs_channel

        # self.gate_conv = nn.Conv2d(in_channels=2 * mhs_channel, out_channels=mhs_channel, kernel_size=3, padding=1)

        self.convmhs1 = nn.Conv2d(mhs_channel*2, 128, 7, padding=3)
        self.convmhs2 = nn.Conv2d(128, mhs_channel, 3, padding=1)

    def forward(self, flow, disp, sf, corr, mhs_reverse, motion_hidden_state):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        d = F.relu(self.convd1(disp))
        d = F.relu(self.convd2(d))
        scene_f = F.relu(self.convsf1(sf))
        scene_f = F.relu(self.convsf2(scene_f))

        # # gate
        # fusion_input = torch.cat([mhs_reverse, motion_hidden_state], dim=1)
        # gate = torch.sigmoid(self.gate_conv(fusion_input))  # 输出在 [0, 1] 之间
        # # fusion
        # fused = gate * mhs_reverse + (1 - gate) * motion_hidden_state
        # motion_hidden_state = F.relu(self.convmhs1(fused))

        # # # Add directly
        # motion_hidden_state = F.relu(self.convmhs1(mhs_reverse+motion_hidden_state))
        # motion_hidden_state = F.relu(self.convmhs2(motion_hidden_state))

        motion_hidden_state = F.relu(self.convmhs1(torch.cat([mhs_reverse, motion_hidden_state], dim=1)))
        motion_hidden_state = F.relu(self.convmhs2(motion_hidden_state))

        cor_flo = torch.cat([cor, flo, d, scene_f, motion_hidden_state], dim=1)
        out = F.relu(self.conv(cor_flo))

        # out, motion_hidden_state = torch.split(out, [128-6, mhs_channel], dim=1)    # feature extraction strategies, MotionEncoder split
        return torch.cat([out, flow, disp, sf], dim=1), 0

from models.my_models.gma import Aggregate    
class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+128)

        self.conv_sf= nn.Sequential(
            conv(hidden_dim, 64),
            conv(64, 32),
            conv(32, 3, isReLU=False)
            )
        self.conv_d1 = nn.Sequential(
            conv(hidden_dim, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
            )
        self.conv_mhs = nn.Sequential(
            conv(hidden_dim, 128),
            conv(128, 128),
            conv(128, mhs_channel, isReLU=False)
            )
        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, disp, sf, mhs_reverse, motion_hidden_state, attention):
        motion_features, _ = self.encoder(flow, disp, sf, corr, mhs_reverse, motion_hidden_state)
        motion_features_global = self.aggregator(attention, motion_features)
        # inp = torch.cat([inp, motion_features], dim=1)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        net = self.gru(net, inp_cat)
        disp = self.conv_d1(net)
        sf = self.conv_sf(net)
        motion_hidden_state = self.conv_mhs(net)
        # scale mask to balence gradients
        mask = self.mask(net.detach())

        return net, mask, disp, sf, motion_hidden_state

from models import softsplat
from utils.interpolation import interpolate2d_as
class FeatureProp_Softsplat(nn.Module):
    """
    Forward-warp a input tensor "x" using input optical "flow"
    A depth order between colliding pixels is determined by the input "disp"
    Also, return only valid feature: it's valid only if the dot product between the feat1 and the corresponding feat2 is above a threshold, 0.5.
    """
    def __init__(self, padding_mode="zeros"):
        super(FeatureProp_Softsplat, self).__init__()
        
        # self.warping_layer_flow = WarpingLayer_Flow()
        self.conv1x1 = conv(1, 1, kernel_size=1)     #    , padding_mode=padding_mode

    def forward(self, x, flow, disp, feat1, feat2):

        # init
        flow = interpolate2d_as(flow, x, mode="bilinear")
        disp = interpolate2d_as(disp, x, mode="bilinear")

        b, _, h, w, = flow.size()
        mask = torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device).requires_grad_(False)
        disocc = softsplat.FunctionSoftsplat(tenInput=mask, tenFlow=flow, tenMetric=None, strType='summation')
        disocc_map = (disocc > 0.5).to(dtype=flow.dtype)

        if disocc_map.sum() < (b * h * w / 2):
            return torch.zeros_like(x)
        else:
            x_warped = softsplat.FunctionSoftsplat(tenInput=x, tenFlow=flow, tenMetric=-20.0 * (0.4-disp), strType='softmax')
            feat1_warped = softsplat.FunctionSoftsplat(tenInput=feat1, tenFlow=flow, tenMetric=-20.0 * (0.4-disp), strType='softmax')

            valid_mask = (self.conv1x1((feat1_warped * feat2).sum(dim=1, keepdims=True)) > 0.5).to(dtype=x_warped.dtype)
            # soft_mask = torch.sigmoid(self.conv1x1((feat1_warped * feat2).sum(dim=1, keepdims=True))).to(dtype=x_warped.dtype)
            # mask_hard = (soft_mask  > 0.5).float()
            # valid_mask = mask_hard + (soft_mask - soft_mask.detach())

            x_warped = x_warped * valid_mask
            return x_warped.contiguous()

class BasicMotionEncoder_video(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_video, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2*2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convsf1 = nn.Conv2d(3*2, 128, 7, padding=3)
        self.convsf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64*3+192 + mhs_channel * 3, 128-2*2-1-2*3 + mhs_channel, 3, padding=1)

    def forward(self, corr_f, corr_b, sf_f, sf_b, disp, flow_f, flow_b, 
                    motion_hidden_state, forward_motion_hidden_state, backward_motion_hidden_state):
        corr_f = F.relu(self.convc1(corr_f))
        corr_b = F.relu(self.convc1(corr_b))
        cor = torch.cat([corr_f, corr_b], dim=1)
        cor = F.relu(self.convc2(cor))
        flow = torch.cat([flow_f, flow_b], dim=1)
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        d = F.relu(self.convd1(disp))
        d = F.relu(self.convd2(d))
        sf = torch.cat([sf_f, sf_b], dim=1)
        scene_f = F.relu(self.convsf1(sf))
        scene_f = F.relu(self.convsf2(scene_f))

        cor_flo = torch.cat([cor, flo, d, scene_f, forward_motion_hidden_state, backward_motion_hidden_state, motion_hidden_state], dim=1)
        out = F.relu(self.conv(cor_flo))

        out, motion_hidden_state = torch.split(out, [128-2*2-1-2*3, mhs_channel], dim=1)
        return torch.cat([out, flow, disp, sf], dim=1), motion_hidden_state
    

class BasicUpdateBlock_video(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock_video, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_video(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        self.conv_sf = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 3*2, isReLU=False)
            )
        self.conv_d1 = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9*2, 1, padding=0))   # 64*9*2    16*16*9*2


    def forward(self, net, inp, corr_f, corr_b, sf_f, sf_b, disp, flow_f, flow_b, motion_hidden_state, forward_motion_hidden_state, backward_motion_hidden_state):
        motion_features, motion_hidden_state = self.encoder(corr_f, corr_b, sf_f, sf_b, disp, flow_f, flow_b, 
                                                            motion_hidden_state, forward_motion_hidden_state, backward_motion_hidden_state)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        disp = self.conv_d1(net)
        sf = self.conv_sf(net)

        # scale mask to balence gradients
        mask = self.mask(net.detach())
        return net, mask, disp, sf, motion_hidden_state
    


class BasicMotionEncoder_video3_f(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_video3_f, self).__init__()
        self.cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(self.cor_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2*2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convsf1 = nn.Conv2d(3*2, 128, 7, padding=3)
        self.convsf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64*3+192, 128-2*2-1-2*3 , 3, padding=1)

    def forward(self, corr, sf, disp, flow):
        corr_f, corr_b = torch.split(corr, [self.cor_planes, self.cor_planes], dim=1)
        corr_f = F.relu(self.convc1(corr_f))
        corr_b = F.relu(self.convc1(corr_b))
        cor = torch.cat([corr_f, corr_b], dim=1)
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        d = F.relu(self.convd1(disp))
        d = F.relu(self.convd2(d))
        scene_f = F.relu(self.convsf1(sf))
        scene_f = F.relu(self.convsf2(scene_f))

        cor_flo = torch.cat([cor, flo, d, scene_f], dim=1)
        out = F.relu(self.conv(cor_flo))

        # out, motion_hidden_state = torch.split(out, [128-2*2-1-2*3, mhs_channel], dim=1)
        return torch.cat([out, flow, disp, sf], dim=1), 0
    

class BasicUpdateBlock_video_3f(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock_video_3f, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_video3_f(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        self.conv_sf = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 3*2, isReLU=False)
            )
        self.conv_d1 = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9*2, 1, padding=0))   # 64*9*2    16*16*9*2

    def forward(self, net, inp, corr, sf, disp, flow):
        motion_features, _ = self.encoder(corr, sf, disp, flow)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        disp = self.conv_d1(net)
        sf = self.conv_sf(net)

        # scale mask to balence gradients
        mask = self.mask(net.detach())
        return net, mask, disp, sf

class BasicMotionEncoder_lstm(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder_lstm, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convsf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convsf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64+128+192 + mhs_channel , 128-6, 3, padding=1) #  + mhs_channel

        self.convmhs1 = nn.Conv2d(mhs_channel*2, 128, 7, padding=3)
        self.convmhs2 = nn.Conv2d(128, mhs_channel, 3, padding=1)

        self.conv_c_init =  conv(mhs_channel , mhs_channel) # + mhs_channel  128-2*2-1-2*3
        self.conv_lstm = conv(mhs_channel + mhs_channel, 4 * mhs_channel, isReLU=False)
        self.cell_state = None
        self.featprop_softsplat = FeatureProp_Softsplat(padding_mode="zeros")

    def forward_lstm(self, input_tensor, h_cur, c_cur):

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv_lstm(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, mhs_channel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = nn.LeakyReLU(0.1, inplace=False)(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * nn.LeakyReLU(0.1, inplace=False)(c_next)

        return h_next, c_next
    
    def forward(self, corr, sf, disp, flow, 
                motion_hidden_state, mhs_reverse, 
                forward_mhs_pr, fl_pr, dp_pr, fmap0, fmap1):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        d = F.relu(self.convd1(disp))
        d = F.relu(self.convd2(d))
        scene_f = F.relu(self.convsf1(sf))
        scene_f = F.relu(self.convsf2(scene_f))

        # motion_hidden_state = self.motion_model(motion_hidden_state, mhs_reverse)
        motion_hidden_state = F.relu(self.convmhs1(torch.cat([mhs_reverse, motion_hidden_state], dim=1)))
        motion_hidden_state = F.relu(self.convmhs2(motion_hidden_state))

        if forward_mhs_pr is not None:
            # forward-warp the hidden state and cell state using the estimated flow and disp
            h_pre = self.featprop_softsplat(forward_mhs_pr, fl_pr, dp_pr, fmap0, fmap1)
            c_pre = self.featprop_softsplat(self.cell_state, fl_pr, dp_pr, fmap0, fmap1)
            motion_hidden_state, self.cell_state = self.forward_lstm(motion_hidden_state, h_pre, c_pre)

        if forward_mhs_pr is None:  # initializing cell state in the begining
            self.cell_state = self.conv_c_init(motion_hidden_state)

        cor_flo = torch.cat([cor, flo, d, scene_f, motion_hidden_state], dim=1)
        out = F.relu(self.conv(cor_flo))

        # out, motion_hidden_state = torch.split(out, [128-2*2-1-2*3, mhs_channel], dim=1)
        # return torch.cat([out, flow, disp, sf], dim=1), motion_hidden_state

        return torch.cat([out, flow, disp, sf], dim=1), 0
    

class BasicUpdateBlock_lstm(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock_lstm, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder_lstm(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        self.conv_sf = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 3, isReLU=False)
            )
        self.conv_d1 = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))   # 64*9    16*16*9
        self.conv_mhs = nn.Sequential(
            conv(hidden_dim, 128),
            conv(128, 128),
            conv(128, mhs_channel, isReLU=False)
        )

    def forward(self, net, inp, corr_f, flow_f, disp, sf_f, forward_motion_hidden_state, backward_motion_hidden_state,
                forward_mhs_pr = None, fl_pr = None, dp_pr = None, fmap0 = None, fmap1 = None):
        motion_features, _ = self.encoder(corr_f, sf_f, disp, flow_f, forward_motion_hidden_state, backward_motion_hidden_state, 
                                          forward_mhs_pr, fl_pr, dp_pr, fmap0, fmap1)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        disp = self.conv_d1(net)
        sf = self.conv_sf(net)
        motion_hidden_state = self.conv_mhs(net)

        # scale mask to balence gradients
        mask = self.mask(net.detach())
        return net, mask, disp, sf, motion_hidden_state
    
    
class BasicMotionEncoder_corr(nn.Module):
    def __init__(self, args, with_mhs = False):
        super(BasicMotionEncoder_corr, self).__init__()
        self.with_mhs = with_mhs
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2*2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convsf1 = nn.Conv2d(3*2, 128, 7, padding=3)
        self.convsf2 = nn.Conv2d(128, 64, 3, padding=1)
        if self.with_mhs:
            self.convmhs1 = nn.Conv2d(mhs_channel*2, 128, 7, padding=3)
            self.convmhs2 = nn.Conv2d(128, mhs_channel, 3, padding=1)
            self.conv = nn.Conv2d(64+128+192 + mhs_channel , 128-2*2-1-3*2, 3, padding=1) #  + mhs_channel

        else:
            self.conv = nn.Conv2d(64*3+192, 128-2*2-1-3*2, 3, padding=1)   #  + mhs_channel*2

    def forward(self, corr_f, corr_b, sf_f, sf_b, disp, flow_f, flow_b, mhs_reverse = None , mhs = None):
        corr_f = F.relu(self.convc1(corr_f))
        corr_b = F.relu(self.convc1(corr_b))
        cor = torch.cat([corr_f, corr_b], dim=1)
        cor = F.relu(self.convc2(cor))
        flow = torch.cat([flow_f, flow_b], dim=1)
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        d = F.relu(self.convd1(disp))
        d = F.relu(self.convd2(d))
        sf = torch.cat([sf_f, sf_b], dim=1)
        scene_f = F.relu(self.convsf1(sf))
        scene_f = F.relu(self.convsf2(scene_f))

        if self.with_mhs:
            mhs = F.relu(self.convmhs1(torch.cat([mhs_reverse, mhs], dim=1)))
            mhs = F.relu(self.convmhs2(mhs))
            cor_flo = torch.cat([cor, flo, d, scene_f, mhs], dim=1)
        else:
            cor_flo = torch.cat([cor, flo, d, scene_f], dim=1)

        out = F.relu(self.conv(cor_flo))
        # out, motion_hidden_state = torch.split(out, [128-2*2-1-3*2, mhs_channel], dim=1)
        return torch.cat([out, flow, disp, sf], dim=1), 0
    

class BasicUpdateBlock_corr(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128, with_mhs = False):
        super(BasicUpdateBlock_corr, self).__init__()
        self.args = args
        self.with_mhs = with_mhs
        self.encoder = BasicMotionEncoder_corr(args, self.with_mhs)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        self.conv_sf = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 3, isReLU=False)
            )
        self.conv_d1 = nn.Sequential(
            conv(128, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))   # 64*9    16*16*9
        if self.with_mhs:
            self.conv_mhs = nn.Sequential(
                conv(hidden_dim, 128),
                conv(128, 128),
                conv(128, mhs_channel, isReLU=False)
            )

    def forward(self, net, inp, corr_f, corr_b, flow_f, flow_b, disp, sf_f, sf_b, mhs_reverse = None , mhs = None):
        motion_features, _ = self.encoder(corr_f, corr_b, sf_f, sf_b, disp, flow_f, flow_b, mhs_reverse, mhs)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)

        disp = self.conv_d1(net)
        sf = self.conv_sf(net)
        mask = self.mask(net.detach())
        if self.with_mhs:
            motion_hidden_state = self.conv_mhs(net)
            return net, mask, disp, sf, motion_hidden_state
        
        return net, mask, disp, sf#, motion_hidden_state


def get_deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )
class MFUSE(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, join="add", inbetween="conv"):
        super(MFUSE,self).__init__()

        self.conv0a = conv(in_channels, 64)
        self.conv0b = conv(64, 64)
        self.down01 = conv(64, 64, stride=2)
        self.conv1a = conv(64, 128)
        self.conv1b = conv(128, 128)
        self.down12 = conv(128, 128, stride=2)
        self.conv2a = conv(128, 256)
        self.conv2b = conv(256, 256)

        self.up21 = get_deconv(256, 128)
        self.conv1c = conv(128, 128)
        self.conv1d = conv(128, 128)

        self.up10 = get_deconv(128, 64)
        self.flow_head = nn.Sequential(
            conv(64, 64),
            conv(64, 64),
            conv(64, 3, isReLU=False),
        )
        # self.disp_head = nn.Sequential(
        #     conv(64, 64),
        #     conv(64, 64),
        #     conv(64, 1, isReLU=False),
        # )

        # self.conv0c = conv(64, 64)
        # self.conv0d = conv(64,64)
        # self.conv0e = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out0 = self.conv0b(self.conv0a(x))
        out1 = self.conv1b(self.conv1a(self.down01(out0)))
        out2 = self.conv2b(self.conv2a(self.down12(out1)))

        up1 = self.up21(out2)
        join1 = out1 + up1

        up0 = self.up10(self.conv1d(self.conv1c(join1)))
        join0 = out0 + up0

        # flow0 = self.conv0e(self.conv0d(self.conv0c(join0)))
        flow = self.flow_head(join0)
        # disp = self.disp_head(join0)

        return flow, 0   #   disp