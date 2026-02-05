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

from models import mhs_channel
from models.my_models.update import BasicUpdateBlock, conv, BasicUpdateBlock_corr
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
        self.only3frames = False

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock_corr(self.args, hidden_dim=hdim)

        self.curr_epoch = -1
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
            inp_b = inp_f.clone()

        # RAFT SceneFlow
        sf_f = torch.zeros(B*2, 3, H//8, W//8).to(img0.device)
        sf_b = torch.zeros(B*2, 3, H//8, W//8).to(img0.device)
        with autocast(enabled=self.args.mixed_precision):
            disp = self.disp_head(fmaps[:, 1:3, ...].reshape(b*2, -1, h, w))
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

            with autocast(enabled=self.args.mixed_precision):
                net_f, up_mask_f, disp_d_f, sf_f_d = self.update_block(net_f, inp_f, corr_f, corr_b, flow_f, -flow_b, 
                                                                                                    disp, sf_f, -sf_b)
                net_b, up_mask_b, disp_d_b, sf_b_d = self.update_block(net_b, inp_b, corr_b, corr_f, flow_b, -flow_f, 
                                                                                                     disp, sf_b, -sf_f)
                
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

    def run_raft(self, img0, img1, img2, img3, k, input_size):
        # b, c, h, w = img1.size() 
        flows_f12, flows_f23, flows_b10, flows_b21, disp_l1, disp_l2 = self.forward_4_frames(img0, img1, img2, img3, k, input_size)

        return flows_f12, flows_f23, flows_b10, flows_b21, disp_l1, disp_l2 # disp_predictions, sf_predictions
    
    def forward(self, input_dict):
        output_dict = {}
        k1 = input_dict['input_k_l1_aug']
        k2 = input_dict['input_k_l2_aug']
        assert torch.equal(k1, k2), "k1 and k2 are not equal"
        input_size = input_dict['aug_size']

        if self.training or (not self.args.finetuning and not self.args.evaluation and not self.args.eval_on_train):
            ## Left
            self.curr_epoch = input_dict['curr_epoch']
            output_dict['flows_f12'], output_dict['flows_f23'], output_dict['flows_b10'], output_dict['flows_b21'], \
            output_dict['disp_l1'], output_dict['disp_l2'] = self.run_raft(input_dict['input_l0_aug'], input_dict['input_l1_aug'], input_dict['input_l2_aug'], 
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
                flows_f12, flows_f23, flows_b10, flows_b21, disp_l1, disp_l2 = self.run_raft(input_r0_flip, input_r1_flip, input_r2_flip, input_r3_flip, k_r1_flip, input_size)

            output_dict_r['flows_f12'] = [ flow_horizontal_flip(flows_f12[ii]) for ii in range(len(flows_f12)) ]
            output_dict_r['flows_f23'] = [ flow_horizontal_flip(flows_f23[ii]) for ii in range(len(flows_f23)) ]
            output_dict_r['flows_b10'] = [ flow_horizontal_flip(flows_b10[ii]) for ii in range(len(flows_b10)) ]
            output_dict_r['flows_b21'] = [ flow_horizontal_flip(flows_b21[ii]) for ii in range(len(flows_b21)) ]
            output_dict_r['disp_l1'] = [ torch.flip(disp_l1[ii], [3]) for ii in range(len(disp_l1)) ]
            output_dict_r['disp_l2'] = [ torch.flip(disp_l2[ii], [3]) for ii in range(len(disp_l2)) ]
            output_dict['output_dict_r'] = output_dict_r

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self.args.evaluation or self.args.finetuning or (self.args.eval_on_train and not self.training):
            ## Left
            output_dict['flows_f12'], output_dict['flows_f23'], output_dict['flows_b10'], output_dict['flows_b21'], \
            output_dict['disp_l1'], output_dict['disp_l2'] = self.run_raft(input_dict['input_l0_aug'], input_dict['input_l1_aug'], input_dict['input_l2_aug'], 
                                                                        input_dict['input_l3_aug'], k1, input_size)
            input_l0_flip = torch.flip(input_dict['input_l0_aug'], [3])
            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            input_l3_flip = torch.flip(input_dict['input_l3_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            flows_f12, flows_f23, flows_b10, flows_b21, disp_l1, disp_l2 = self.run_raft(input_l0_flip, input_l1_flip, input_l2_flip, input_l3_flip, k_l1_flip, input_size)

            flows_f12_pp = []
            flows_f23_pp = []
            flows_b10_pp = []
            flows_b21_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(flows_f12)):
                
                flows_f12_pp.append(post_processing(output_dict['flows_f12'][ii], flow_horizontal_flip(flows_f12[ii])))
                flows_f23_pp.append(post_processing(output_dict['flows_f23'][ii], flow_horizontal_flip(flows_f23[ii])))
                flows_b10_pp.append(post_processing(output_dict['flows_b10'][ii], flow_horizontal_flip(flows_b10[ii])))
                flows_b21_pp.append(post_processing(output_dict['flows_b21'][ii], flow_horizontal_flip(flows_b21[ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(disp_l1[ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(disp_l2[ii], [3])))

            output_dict['flows_f12_pp'] = flows_f12_pp
            output_dict['flows_f23_pp'] = flows_f23_pp
            output_dict['flows_b10_pp'] = flows_b10_pp
            output_dict['flows_b21_pp'] = flows_b21_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            # output_dict['flow_b_pp'] = flow_b_pp
            # output_dict['disp_l1_pp'] = disp_l1_pp
            # output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict
