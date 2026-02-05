import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from models.extractor import BasicEncoder
from models.corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, proj_coords_grid, upsample_x8
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing
from utils.sceneflow_util import projectSceneFlow2Flow, flow_warp

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        
def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
        (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()

from models.forwardwarp_package.forward_warp import forward_warp
def _adaptive_disocc_detection(flow):

    # init mask
    b, _, h, w, = flow.size()
    mask = torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device).float().requires_grad_(False)    
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1) 
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)
        
    return disocc_map

from models import mhs_channel
from models.my_models.update import BasicUpdateBlock, conv
from models.my_models.gma import Attention
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.sigmoid = torch.nn.Sigmoid()
        self.disp_head =  nn.Sequential(
            conv(256, 128),
            conv(128, 1, isReLU=False),
            )
        
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if not hasattr(self.args, 'dropout'):
            self.args.dropout = 0

        if not hasattr(self.args, 'alternate_corr'):
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.init_hidden_state = nn.Parameter(torch.randn(1, 1, mhs_channel, 1, 1))
        self.curr_epoch = -1

        self.att = Attention(args=self.args, dim=128, heads=1, max_pos_size=160, dim_head=128)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def cvx_up_pred(self, pred, mask):
        N, dim, H, W = pred.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_pred = F.unfold(pred, [3,3], padding=1)
        up_pred = up_pred.view(N, dim, 9, 1, 1, H, W)
        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, dim, 8*H, 8*W)

    def sceneF2opticalF(self, disp, sf, K, input_size):
        _, _, h, w = disp.size()
        disp = disp * w
        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h
        local_scale[:, 1] = w
        rel_scale = local_scale / input_size
        intrinsic_s = intrinsic_scale(K, rel_scale[:, 0], rel_scale[:, 1])
        proj_flow, coord = projectSceneFlow2Flow(intrinsic_s, sf, disp)
        return proj_flow, coord

    def forward_3_frames(self, img0, img1, img2, K, input_size):
        # K = torch.repeat_interleave(K, 2, dim=0)
        # input_size = torch.repeat_interleave(input_size, 2, dim=0)  #  .repeat(1, 2).reshape(-1, 2)
        """ Estimate optical flow between pair of frames """
        iters = self.args.iters
        hdim = self.hidden_dim
        cdim = self.context_dim
        B,  _, H, W = img0.shape
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            self.fnet.frames = 3
            fmap0, fmap1, fmap2 = self.fnet([img0, img1, img2])

        fmap0 = fmap0.float()
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        b, c, h, w = fmap0.shape

        corr_fn_forward = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        corr_fn_backward = CorrBlock(fmap1, fmap0, radius=self.args.corr_radius)
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(img1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net_f = torch.tanh(net)
            net_b = net_f.clone()
            inp_f = torch.relu(inp)
            inp_b = inp_f
            attention = self.att(inp)

        # RAFT SceneFlow
        sf_f = torch.zeros(B, 3, H//8, W//8).to(img0.device)
        sf_b = torch.zeros(B, 3, H//8, W//8).to(img0.device)
        disp = self.disp_head(fmap1)
        forward_motion_hidden_state, backward_motion_hidden_state = self.init_hidden_state.repeat(b, 2, 1, h, w).split(1, dim=1)
        forward_motion_hidden_state = forward_motion_hidden_state.reshape(b, -1, h, w)
        backward_motion_hidden_state = backward_motion_hidden_state.reshape(b, -1, h, w)

        coords0_f = coords_grid(B, H//8, W//8).to(img0.device)
        coords0_b = coords_grid(B, H//8, W//8).to(img0.device)
        _, coords1_f = self.sceneF2opticalF(disp * 0.3, sf_f, K, input_size) # multiplied by 0.3: ijcvRevision
        _, coords1_b = self.sceneF2opticalF(disp * 0.3, sf_b, K, input_size) # multiplied by 0.3: ijcvRevision

        disp_predictions = []
        flows_f12 = []
        flows_b10 = []
        for itr in range(iters):
            coords1_f = coords1_f.detach()
            coords1_b = coords1_b.detach()
            disp = disp.detach()
            sf_f = sf_f.detach()
            sf_b = sf_b.detach()
            corr_f = corr_fn_forward(coords1_f) # index correlation volume
            corr_b = corr_fn_backward(coords1_b) # index correlation volume
            flow_f = coords1_f - coords0_f
            flow_b = coords1_b - coords0_b
            
            forward_12_23 = backward_motion_hidden_state * -1.0  # 12 23
            backward_10_21 = forward_motion_hidden_state * -1.0  # 10 21

            with autocast(enabled=self.args.mixed_precision):
                net_combined = torch.cat([net_f, net_b], dim=0)
                inp_combined = torch.cat([inp_f, inp_b], dim=0)
                corr_combined = torch.cat([corr_f, corr_b], dim=0)
                flow_combined = torch.cat([flow_f, flow_b], dim=0)
                disp_combined = torch.cat([disp, disp], dim=0)
                sf_combined = torch.cat([sf_f, sf_b], dim=0)
                hidden_state_combined = torch.cat([forward_motion_hidden_state, backward_motion_hidden_state], dim=0)
                motion_info_combined = torch.cat([forward_12_23, backward_10_21], dim=0)
                attention_combined = torch.cat([attention, attention], dim=0)

                combined_output = self.update_block(net_combined, inp_combined, corr_combined, flow_combined, disp_combined, sf_combined, 
                                                    motion_info_combined, hidden_state_combined, attention_combined)

                net_f, net_b = combined_output[0].chunk(2, dim=0)
                up_mask_f, up_mask_b = combined_output[1].chunk(2, dim=0)
                disp_d_f, disp_d_b = combined_output[2].chunk(2, dim=0)
                sf_f_d, sf_b_d = combined_output[3].chunk(2, dim=0)
                forward_motion_hidden_state, backward_motion_hidden_state = combined_output[4].chunk(2, dim=0)

                
            disp = disp + (disp_d_f + disp_d_b) / 2.0
            sf_f = sf_f + sf_f_d
            sf_b = sf_b + sf_b_d
            _, coords1_f = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf_f, K, input_size) 
            _, coords1_b = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf_b, K, input_size)
            
            if up_mask_f is None:
                disp_up = upsample_x8(self.sigmoid(disp) * 0.3)
                sf_f_up = upsample_x8(sf_f)
            else:
                disp_up = self.cvx_up_pred(self.sigmoid(disp) * 0.3, up_mask_f)
                sf_f_up = self.cvx_up_pred(sf_f, up_mask_f)
                # sf_b_up = self.cvx_up_pred(sf_b, up_mask_b)

            disp_predictions.append(disp_up)
            flows_f12.append(sf_f_up)
            # flows_b10.append(sf_b_up)
          
        return flows_f12, disp_predictions


    def forward_4_frames(self, img0, img1, img2, img3, K, input_size):
        K = torch.repeat_interleave(K, 2, dim=0)
        input_size = torch.repeat_interleave(input_size, 2, dim=0)  #  .repeat(1, 2).reshape(-1, 2)
        """ Estimate optical flow between pair of frames """
        iters = self.args.iters
        hdim = self.hidden_dim
        cdim = self.context_dim
        B,  _, H, W = img0.shape
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            self.fnet.frames = 4
            fmap0, fmap1, fmap2, fmap3 = self.fnet([img0, img1, img2, img3])

        fmap0 = fmap0.float()
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        fmap3 = fmap3.float()
        b, c, h, w = fmap0.shape
        fmaps = torch.stack((fmap0, fmap1, fmap2, fmap3), dim=1)  #  b, N, c, h, w
        images = torch.stack((img0, img1, img2, img3), dim=1)  #  b, N, c, h, w

        corr_fn_forward = CorrBlock(fmaps[:, 1:3, ...].reshape(b*2, -1, h, w), fmaps[:, 2:4, ...].reshape(b*2, -1, h, w), radius=self.args.corr_radius)
        corr_fn_backward = CorrBlock(fmaps[:, 1:3, ...].reshape(b*2, -1, h, w), fmaps[:, 0:2, ...].reshape(b*2, -1, h, w), radius=self.args.corr_radius)
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(images[:, 1:3, ...].reshape(B*2, 3, H, W))
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net_f = torch.tanh(net)
            net_b = net_f.clone()
            inp_f = torch.relu(inp)
            inp_b = inp_f
            attention = self.att(inp)

        # RAFT SceneFlow
        sf_f = torch.zeros(B*2, 3, H//8, W//8).to(img0.device)
        sf_b = torch.zeros(B*2, 3, H//8, W//8).to(img0.device)
        disp = self.disp_head(fmaps[:, 1:3, ...].reshape(b*2, -1, h, w))

        forward_motion_hidden_state, backward_motion_hidden_state = self.init_hidden_state.repeat(b, 4, 1, h, w).split(2, dim=1) # [:2]
        coords0_f = coords_grid(B*2, H//8, W//8).to(img0.device)
        coords0_b = coords_grid(B*2, H//8, W//8).to(img0.device)
        _, coords1_f = self.sceneF2opticalF(disp * 0.3, sf_f, K, input_size) # multiplied by 0.3: ijcvRevision
        _, coords1_b = self.sceneF2opticalF(disp * 0.3, sf_b, K, input_size) # multiplied by 0.3: ijcvRevision

        disp1_predictions = []
        disp2_predictions = []
        flows_f12 = []
        flows_f23 = []
        flows_b10 = []
        flows_b21 = []

        for itr in range(iters):
            coords1_f = coords1_f.detach()
            coords1_b = coords1_b.detach()
            disp = disp.detach()
            sf_f = sf_f.detach()
            sf_b = sf_b.detach()
            corr_f = corr_fn_forward(coords1_f) # index correlation volume
            corr_b = corr_fn_backward(coords1_b) # index correlation volume
            flow_f = coords1_f - coords0_f
            flow_b = coords1_b - coords0_b
            
            # # MIP from m2flow
            ##############################
            # m_delta = forward_motion_hidden_state.reshape(b, 2, -1, h, w) + backward_motion_hidden_state.reshape(b, 2, -1, h, w) # b 2 24 h w
            # delta_1 = m_delta[:, 0, ...].float()  # b 24 h w
            # delta_2_hat = (flow_warp(delta_1, flow_b.reshape(b, 2, 2, h, w)[:, 1, ...], pad='border')) # *vis_mask2

            # delta_2 = m_delta[:, 1, ...].float()  # b 24 h w
            # delta_1_hat = (flow_warp(delta_2, flow_f.reshape(b, 2, 2, h, w)[:, 0, ...], pad='border')) # *vis_mask1
            # delta_1_2_hat = torch.stack((delta_1_hat, delta_2_hat), dim=1)

            # forward_12_23 = backward_motion_hidden_state.reshape(b*2, -1, h, w) * -1.0  # 12 23
            # forward_12_23 = forward_12_23 + delta_1_2_hat.reshape(b*2, -1, h, w)

            # backward_10_21 = forward_motion_hidden_state.reshape(b*2, -1, h, w) * -1.0  # 10 21
            # backward_10_21 = backward_10_21 + delta_1_2_hat.reshape(b*2, -1, h, w)
            
            forward_12_23 = backward_motion_hidden_state * -1.0  # 12 23
            backward_10_21 = forward_motion_hidden_state * -1.0  # 10 21

            with autocast(enabled=self.args.mixed_precision):
                net_f, up_mask_f, disp_d_f, sf_f_d, forward_motion_hidden_state = self.update_block(net_f, inp_f, corr_f, flow_f, disp, sf_f, 
                                                                                                    forward_12_23.reshape(b*2, -1, h, w), 
                                                                                                    forward_motion_hidden_state.reshape(b*2, -1, h, w),
                                                                                                    attention
                                                                                                    )
                net_b, up_mask_b, disp_d_b, sf_b_d, backward_motion_hidden_state = self.update_block(net_b, inp_b, corr_b, flow_b, disp, sf_b,
                                                                                                     backward_10_21.reshape(b*2, -1, h, w), 
                                                                                                     backward_motion_hidden_state.reshape(b*2, -1, h, w),
                                                                                                     attention
                                                                                                    )
                
            disp = disp + (disp_d_f + disp_d_b) / 2.0
            sf_f = sf_f + sf_f_d
            sf_b = sf_b + sf_b_d
            _, coords1_f = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf_f, K, input_size) 
            _, coords1_b = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf_b, K, input_size)
            
            if up_mask_f is None:
                disp_up = upsample_x8(self.sigmoid(disp) * 0.3)
                sf_f_up = upsample_x8(sf_f)
            else:
                disp_up = self.cvx_up_pred(self.sigmoid(disp) * 0.3, up_mask_f)
                sf_f_up = self.cvx_up_pred(sf_f, up_mask_f)
                sf_b_up = self.cvx_up_pred(sf_b, up_mask_b)

            disp_up = disp_up.reshape(b, 2, -1, H, W)
            disp1_predictions.append(disp_up[:, 0, ...])
            disp2_predictions.append(disp_up[:, 1, ...])
            sf_f_up = sf_f_up.reshape(b, 2, -1, H, W)
            flows_f12.append(sf_f_up[:, 0, ...])
            flows_f23.append(sf_f_up[:, 1, ...])
            sf_b_up = sf_b_up.reshape(b, 2, -1, H, W)
            flows_b10.append(sf_b_up[:, 0, ...])
            flows_b21.append(sf_b_up[:, 1, ...])
          
        return flows_f12, flows_f23, flows_b10, flows_b21, disp1_predictions, disp2_predictions

    def forward(self, input_dict):

        output_dict = {}
        k1 = input_dict['input_k_l1_aug']
        k2 = input_dict['input_k_l2_aug']
        # assert torch.equal(k1, k2), "k1 and k2 are not equal"
        input_size = input_dict['aug_size']

        if self.training:
            ## Left
            self.curr_epoch = input_dict['curr_epoch']
            output_dict['flows_f12'], output_dict['flows_f23'], output_dict['flows_b10'], output_dict['flows_b21'], \
            output_dict['disp_l1'], output_dict['disp_l2'] = self.forward_4_frames(input_dict['input_l0_aug'], input_dict['input_l1_aug'], input_dict['input_l2_aug'], 
                                                                        input_dict['input_l3_aug'], k1, input_size)
            output_dict_r = {}
            input_r0_flip = torch.flip(input_dict['input_r0_aug'], [3])
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            input_r3_flip = torch.flip(input_dict['input_r3_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]
            assert torch.equal(k_r1_flip, k_r2_flip), "k_r1_flip and k_r2_flip are not equal"
            
            with torch.no_grad():
                flows_f12, flows_f23, flows_b10, flows_b21, disp_l1, disp_l2 = self.forward_4_frames(input_r0_flip, input_r1_flip, input_r2_flip, input_r3_flip, k_r1_flip, input_size)

            output_dict_r['flows_f12'] = [ flow_horizontal_flip(flows_f12[ii]) for ii in range(len(flows_f12)) ]
            output_dict_r['flows_f23'] = [ flow_horizontal_flip(flows_f23[ii]) for ii in range(len(flows_f23)) ]
            output_dict_r['flows_b10'] = [ flow_horizontal_flip(flows_b10[ii]) for ii in range(len(flows_b10)) ]
            output_dict_r['flows_b21'] = [ flow_horizontal_flip(flows_b21[ii]) for ii in range(len(flows_b21)) ]
            output_dict_r['disp_l1'] = [ torch.flip(disp_l1[ii], [3]) for ii in range(len(disp_l1)) ]
            output_dict_r['disp_l2'] = [ torch.flip(disp_l2[ii], [3]) for ii in range(len(disp_l2)) ]

            output_dict['output_dict_r'] = output_dict_r

        if self.training and self.args.finetuning:
            input_l0_flip = torch.flip(input_dict['input_l0_aug'], [3])
            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]

            flows_f12, disp_l1 = self.forward_3_frames(input_l0_flip, input_l1_flip, input_l2_flip, k_l1_flip, input_size)

            flows_f12_pp = []
            disp_l1_pp = []

            for ii in range(0, len(flows_f12)):
                flows_f12_pp.append(post_processing(output_dict['flows_f12'][ii], flow_horizontal_flip(flows_f12[ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(disp_l1[ii], [3])))

            output_dict['flows_f12_pp'] = flows_f12_pp
            output_dict['disp_l1_pp'] = disp_l1_pp

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self.args.evaluation or  (self.args.eval_on_train and not self.training):
            # last_iter = self.args.iters
            # self.args.iters = 10
            ## Left
            output_dict['flows_f12'], output_dict['disp_l1'] = self.forward_3_frames(input_dict['input_l0_aug'], input_dict['input_l1_aug'], \
                                                                input_dict['input_l2_aug'], k1, input_size)
            input_l0_flip = torch.flip(input_dict['input_l0_aug'], [3])
            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]

            flows_f12, disp_l1 = self.forward_3_frames(input_l0_flip, input_l1_flip, input_l2_flip, k_l1_flip, input_size)

            flows_f12_pp = []
            disp_l1_pp = []

            for ii in range(0, len(flows_f12)):
                flows_f12_pp.append(post_processing(output_dict['flows_f12'][ii], flow_horizontal_flip(flows_f12[ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(disp_l1[ii], [3])))

            output_dict['flows_f12_pp'] = flows_f12_pp
            output_dict['disp_l1_pp'] = disp_l1_pp


        return output_dict
