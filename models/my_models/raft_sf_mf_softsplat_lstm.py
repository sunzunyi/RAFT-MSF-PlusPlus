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
from models.my_models.update import BasicUpdateBlock, conv, BasicUpdateBlock_lstm
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
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, frames=3)        
        # self.fnet = twins_svt_large(pretrained=False)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock_lstm(self.args, hidden_dim=hdim)

        self.init_hidden_state = nn.Parameter(torch.randn(1, mhs_channel, 1, 1))
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

    def cvx_up_pred_x16(self, pred, mask):
        N, dim, H, W = pred.shape
        mask = mask.view(N, 1, 9, 16, 16, H, W)
        mask = torch.softmax(mask, dim=2)
        up_pred = F.unfold(pred, [3,3], padding=1)
        up_pred = up_pred.view(N, dim, 9, 1, 1, H, W)
        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, dim, 16*H, 16*W)
    
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

    def forward_3_frames(self, img0, img1, img2, K, input_size, sf_pr=None, dp_pr=None, x_outs_f_pr=None, x_outs_b_pr=None):
        """ Estimate optical flow between pair of frames """
        iters = self.args.iters
        hdim = self.hidden_dim
        cdim = self.context_dim
        B,  _, H, W = img0.shape
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
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
            net_b = net_f # .clone()
            inp_f = torch.relu(inp)
            inp_b = inp_f # .clone()

        # RAFT SceneFlow
        sf_f = torch.zeros(B, 3, H//8, W//8).to(img0.device)
        sf_b = torch.zeros(B, 3, H//8, W//8).to(img0.device)
        with autocast(enabled=self.args.mixed_precision):
            disp = self.disp_head(fmap1)
            forward_motion_hidden_state, backward_motion_hidden_state = self.init_hidden_state.repeat(b*2, 1, h, w).split(b, dim=0) # [:2]
            # forward_motion_hidden_state, backward_motion_hidden_state = None, None
        
        coords0_f = coords_grid(B, H//8, W//8).to(img0.device)
        coords0_b = coords_grid(B, H//8, W//8).to(img0.device)
        _, coords1_f = self.sceneF2opticalF(disp * 0.3, sf_f, K, input_size) # multiplied by 0.3: ijcvRevision
        _, coords1_b = self.sceneF2opticalF(disp * 0.3, sf_b, K, input_size) # multiplied by 0.3: ijcvRevision

        disps_l = []
        flows_f = []
        flows_b = []
        x_outs_f = []
        x_outs_b = []
        if sf_pr != None:
            fl_pr, _ = projectSceneFlow2Flow(K, sf_pr, dp_pr * dp_pr.size(-1), input_size)
            fl_pr = fl_pr.detach()
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

            forward_reverse = -forward_motion_hidden_state
            backward_reverse = -backward_motion_hidden_state

            if sf_pr is not None: 
                forward_mhs_pr = x_outs_f_pr[itr]
                backward_mhs_pr = x_outs_b_pr[itr]
                with autocast(enabled=self.args.mixed_precision):
                    net_f, up_mask_f, disp_d_f, sf_f_d, forward_motion_hidden_state = self.update_block(net_f, inp_f, corr_f, flow_f, disp, sf_f,
                                                                                                        forward_motion_hidden_state, backward_reverse,
                                                                                                        forward_mhs_pr, fl_pr, dp_pr, fmap0, fmap1)
                    net_b, up_mask_b, disp_d_b, sf_b_d, backward_motion_hidden_state = self.update_block(net_b, inp_b, corr_b, flow_b, disp, sf_b,
                                                                                                        backward_motion_hidden_state, forward_reverse,
                                                                                                        backward_mhs_pr, fl_pr, dp_pr, fmap0, fmap1)
                
            else:
                with autocast(enabled=self.args.mixed_precision):
                    net_f, up_mask_f, disp_d_f, sf_f_d, forward_motion_hidden_state = self.update_block(net_f, inp_f, corr_f, flow_f, disp, sf_f,
                                                                                                        forward_motion_hidden_state, backward_reverse)
                    net_b, up_mask_b, disp_d_b, sf_b_d, backward_motion_hidden_state = self.update_block(net_b, inp_b, corr_b, flow_b, disp, sf_b,
                                                                                                        backward_motion_hidden_state, forward_reverse)
                
            disp = disp + (disp_d_f + disp_d_b) / 2.0
            sf_f = sf_f + sf_f_d
            sf_b = sf_b + sf_b_d
            _, coords1_f = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf_f, K, input_size) 
            _, coords1_b = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf_b, K, input_size)
            
            disp_up = self.cvx_up_pred(self.sigmoid(disp) * 0.3, up_mask_f)
            sf_f_up = self.cvx_up_pred(sf_f, up_mask_f)
            sf_b_up = self.cvx_up_pred(sf_b, up_mask_b)

            disps_l.append(disp_up)
            flows_f.append(sf_f_up)
            flows_b.append(sf_b_up)
            x_outs_f.append(forward_motion_hidden_state)
            x_outs_b.append(backward_motion_hidden_state)

        return flows_f, flows_b, disps_l, x_outs_f, x_outs_b

    def run_raft(self, img0, img1, img2, img3, k, input_size):
        # b, c, h, w = img1.size() 

        flows_f12, flows_b10, disp_l1, x_outs_f, x_outs_b = self.forward_3_frames(img0, img1, img2, k, input_size)
        flows_f23, flows_b21, disp_l2, x_outs_f, x_outs_b = self.forward_3_frames(img1, img2, img3, k, input_size, flows_f12[-1], disp_l1[-1], x_outs_f, x_outs_b)

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

            # When using the three-frame model, the input should be four consecutive images:
            # 09, 10, 11, 12, where the flow from 10 -> 11 has ground truth for evaluation.
            # When using the four-frame model, the input should be four consecutive images:
            # 08, 09, 10, 11, where the flow from 10 -> 11 has ground truth for evaluation.
            # Modify the input images in kitti_2015_train & kitti_2015_test accordingly if needed.

            for ii in range(0, len(flows_f23)):
                
                # flows_f12_pp.append(post_processing(output_dict['flows_f12'][ii], flow_horizontal_flip(flows_f12[ii])))
                flows_f23_pp.append(post_processing(output_dict['flows_f23'][ii], flow_horizontal_flip(flows_f23[ii])))
                # flows_b10_pp.append(post_processing(output_dict['flows_b10'][ii], flow_horizontal_flip(flows_b10[ii])))
                # flows_b21_pp.append(post_processing(output_dict['flows_b21'][ii], flow_horizontal_flip(flows_b21[ii])))
                # disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(disp_l1[ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(disp_l2[ii], [3])))

            output_dict['flows_f12_pp'] = flows_f23_pp
            output_dict['disp_l1_pp'] = disp_l2_pp

        return output_dict
