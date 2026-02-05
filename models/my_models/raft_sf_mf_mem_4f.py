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
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    print('no flash attention installed')
    # assert False, 'no flash attention installed'
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
from models.my_models.update_mem import BasicUpdateBlock, conv, Attention
import math
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

        # if 'dropout' not in self.args:
        self.args.dropout = 0

        # if 'alternate_corr' not in self.args:
        self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.init_hidden_state = nn.Parameter(torch.randn(1, 1, mhs_channel, 1, 1))
        self.conv1x1 = conv(1, 1, kernel_size=1)     #    , padding_mode=padding_mode
        self.att = Attention(args=self.args, dim=self.context_dim, heads=1, max_pos_size=160, dim_head=self.context_dim)
        self.curr_epoch = -1
        
    def encode_context(self, frame):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        # shape is b*c*h*w
        cnet = self.cnet(frame)
        # if self.cfg.cnet == 'twins':
        #     cnet = self.proj(cnet)

        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        query, key = self.att.to_qk(inp).chunk(2, dim=1)
        # query = query * self.att.scale
        if need_reshape:
            # B*C*T*H*W
            query = query.view(b, t, *query.shape[-3:]).transpose(1, 2).contiguous()
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            net = net.view(b, t, *net.shape[-3:])
            inp = inp.view(b, t, *inp.shape[-3:])

        return query, key, net, inp

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def encode_features(self, frame, flow_init=None):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        fmaps = self.fnet(frame).float()
        # if self.cfg.fnet == 'twins':
        #     fmaps = self.channel_convertor(fmaps)
        if need_reshape:
            # B*T*C*H*W
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            frame = frame.view(b, t, *frame.shape[-3:])
            coords0, coords1 = self.initialize_flow(frame[:, 0, ...])
        else:
            coords0, coords1 = self.initialize_flow(frame)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        return coords0, coords1, fmaps
    
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

    def predict_flow(self, net, inp, coords0, coords1, sf, disp, K, input_size, 
                     fmaps, query, ref_keys, ref_values, test_mode=False):
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...],
                            num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        flow_predictions = []
        disp_predictions = []
        query = query.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2) # 2, 128, 32, 104 -> 2, 3328, 1, 128
        ref_keys = ref_keys.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2) # 2, 128, 2, 32, 104 -> 2, 6656, 1, 128
        for iter_ in range(self.args.iters): # self.cfg.decoder_depth  训练12   推理15
            coords1 = coords1.detach()
            sf = sf.detach()
            disp = disp.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            motion_features, current_value = self.update_block.get_motion_and_value(flow, disp, sf, corr)
            current_value = current_value.unsqueeze(2) # 2, 128, 1, 32, 104
            value = current_value if ref_values is None else torch.cat([ref_values, current_value], dim=2) # 2, 128, 1, 32, 104
            # get global motion
            # B, L, N, C
            value = value.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)  # 2, 3328, 1, 128
            train_avg_length = (256 * 832 // 64) * 3 / 2
            scale = self.att.scale * math.log(ref_keys.shape[1], train_avg_length)
            hidden_states = flash_attn_func(query.contiguous(), ref_keys.contiguous(), value.contiguous(), dropout_p=0.0, softmax_scale=scale, causal=False) # batch_size, seqlen, nheads, headdim
            hidden_states = hidden_states.contiguous()
            hidden_states = hidden_states.squeeze(2).permute(0, 2, 1).reshape(motion_features.shape)

            motion_features_global = motion_features + self.update_block.aggregator.gamma * hidden_states
            net, up_mask, disp_d, sf_d = self.update_block(net, inp, motion_features, motion_features_global)
            disp = disp + disp_d
            sf = sf + sf_d
            _, coords1 = self.sceneF2opticalF(self.sigmoid(disp) * 0.3, sf, K, input_size) 
            # upsample predictions
            sf_up = self.cvx_up_pred(sf, up_mask)
            disp_up = self.cvx_up_pred(self.sigmoid(disp) * 0.3, up_mask)

            flow_predictions.append(sf_up)
            disp_predictions.append(disp_up)

        return flow_predictions, disp_predictions, current_value
        
    def forward_4_frames(self, img0, img1, img2, img3, K, input_size):
        B,  _, H, W = img0.shape
        images = torch.stack((img0, img1, img2, img3), dim=1)  #  b, N, c, h, w
        with autocast(enabled=self.args.mixed_precision): 
            # B*C*N-1*H*W,                    B*N-1*C*H*W
            query, key, net, inp = self.encode_context(images[:, :-1, ...])
            coords0, coords1, fmaps = self.encode_features(images)
        b, N, c, h, w = fmaps.shape
        flows_f01 = torch.zeros(B, 3, H//8, W//8).to(img0.device)
        flows_f12 = torch.zeros(B, 3, H//8, W//8).to(img0.device)
        flows_f23 = torch.zeros(B, 3, H//8, W//8).to(img0.device)
        with autocast(enabled=self.args.mixed_precision): 
            disp0, disp1, disp2 = self.disp_head(fmaps[:, 0:3, ...].reshape(b*3, -1, h, w)).reshape(b, 3, h, w).split(1, dim=1)
        with autocast(enabled=self.args.mixed_precision): # dtype=torch.bfloat16
            values = None
            # f0 f1 #################################################
            _, coords1 = self.sceneF2opticalF(disp0 * 0.3, flows_f01, K, input_size) # multiplied by 0.3: ijcvRevision
            ref_values = values
            ref_keys = key[:, :, :1]
            flows_f01, disp0, current_value = self.predict_flow(net[:, 0], inp[:, 0], coords0, coords1, flows_f01, disp0, K, input_size,
                                                        fmaps[:, 0:0 + 2], query[:, :, 0], ref_keys, ref_values)
            values = current_value if values is None else torch.cat([values, current_value], dim=2)
            # f1 f2 #################################################
            _, coords1 = self.sceneF2opticalF(disp1 * 0.3, flows_f12, K, input_size) # multiplied by 0.3: ijcvRevision
            ref_values = values
            ref_keys = key[:, :, :2]
            flows_f12, disp1, current_value = self.predict_flow(net[:, 1], inp[:, 1], coords0, coords1, flows_f12, disp1, K, input_size,
                                                        fmaps[:, 1:1 + 2], query[:, :, 1], ref_keys, ref_values)
            values = torch.cat([values, current_value], dim=2)
            # f2 f3 #################################################
            _, coords1 = self.sceneF2opticalF(disp2 * 0.3, flows_f23, K, input_size) # multiplied by 0.3: ijcvRevision
            ref_values = values
            ref_keys = key[:, :, :3]
            flows_f23, disp2, current_value = self.predict_flow(net[:, 2], inp[:, 2], coords0, coords1, flows_f23, disp2, K, input_size,
                                                        fmaps[:, 2:2 + 2], query[:, :, 2], ref_keys, ref_values)
            # values = current_value if values is None else torch.cat([values, current_value], dim=2)
        return flows_f01, flows_f12, flows_f23, disp0, disp1, disp2

    def forward_3_frames(self, img1, img2, img3, K, input_size):
        B,  _, H, W = img1.shape
        images = torch.stack((img1, img2, img3), dim=1)  #  b, N, c, h, w
        with autocast(enabled=self.args.mixed_precision): 
            # B*C*N-1*H*W,                    B*N-1*C*H*W
            query, key, net, inp = self.encode_context(images[:, :-1, ...])
            coords0, coords1, fmaps = self.encode_features(images)
        b, N, c, h, w = fmaps.shape
        flows_f12 = torch.zeros(B, 3, H//8, W//8).to(img1.device)
        flows_f23 = torch.zeros(B, 3, H//8, W//8).to(img1.device)
        with autocast(enabled=self.args.mixed_precision): 
            disp1, disp2 = self.disp_head(fmaps[:, 0:2, ...].reshape(b*2, -1, h, w)).reshape(b, 2, h, w).split(1, dim=1)
        with autocast(enabled=self.args.mixed_precision): # dtype=torch.bfloat16
            values = None
            # f1 f2 #################################################
            _, coords1 = self.sceneF2opticalF(disp1 * 0.3, flows_f12, K, input_size) # multiplied by 0.3: ijcvRevision
            ref_values = values
            ref_keys = key[:, :, :1]
            flows_f12, disp1, current_value = self.predict_flow(net[:, 0], inp[:, 0], coords0, coords1, flows_f12, disp1, K, input_size,
                                                        fmaps[:, 0:0 + 2], query[:, :, 0], ref_keys, ref_values)
            values = current_value if values is None else torch.cat([values, current_value], dim=2)
            # f2 f3 #################################################
            _, coords1 = self.sceneF2opticalF(disp2 * 0.3, flows_f23, K, input_size) # multiplied by 0.3: ijcvRevision
            ref_values = values
            ref_keys = key[:, :, :2]
            flows_f23, disp2, current_value = self.predict_flow(net[:, 1], inp[:, 1], coords0, coords1, flows_f23, disp2, K, input_size,
                                                        fmaps[:, 1:1 + 2], query[:, :, 1], ref_keys, ref_values)
            # values = current_value if values is None else torch.cat([values, current_value], dim=2)
        return flows_f12, flows_f23, disp1, disp2

    def run_raft(self, img0, img1, img2, img3, k, input_size):
    
        flows_f12, flows_f23, disp1_f, disp2_f = self.forward_3_frames(img1, img2, img3, k, input_size)
        flows_b21, flows_b10, disp2_b, disp1_b = self.forward_3_frames(img2, img1, img0, k, input_size)
        # return flows_f12, flows_f23, None, None, disp1_f, disp2_f
        return flows_f12, flows_f23, flows_b10, flows_b21, disp1_f, disp2_b


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
            
            # When using the three-frame model, the input should be four consecutive images:
            # 09, 10, 11, 12, where the flow from 10 -> 11 has ground truth for evaluation.
            # When using the four-frame model, the input should be four consecutive images:
            # 08, 09, 10, 11, where the flow from 10 -> 11 has ground truth for evaluation.
            # Modify the input images in kitti_2015_train & kitti_2015_test accordingly if needed.

            # 3-frame model
            _, output_dict['flows_f12'], output_dict['disp_l1'], _ = self.forward_3_frames(input_dict['input_l0_aug'], input_dict['input_l1_aug'], input_dict['input_l2_aug'], 
                                                                           k1, input_size)
            # # 4-frame model
            # _, _, output_dict['flows_f12'], _, _, output_dict['disp_l1'] = self.forward_4_frames(input_dict['input_l0_aug'], input_dict['input_l1_aug'], input_dict['input_l2_aug'], 
            #                                                                                  input_dict['input_l3_aug'], k1, input_size)

            input_l0_flip = torch.flip(input_dict['input_l0_aug'], [3])
            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            input_l3_flip = torch.flip(input_dict['input_l3_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            # 3-frame model
            _, flows_f12, disp_l1, _ = self.forward_3_frames(input_l0_flip, input_l1_flip, input_l2_flip, k_l1_flip, input_size)

            # # 4-frame model
            # _, _, flows_f12, _, _, disp_l1= self.forward_4_frames(input_l0_flip, input_l1_flip, input_l2_flip, input_l3_flip, 
            #                                                                                  k_l1_flip, input_size)


            flows_f12_pp = []
            flows_f23_pp = []
            flows_b10_pp = []
            flows_b21_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(flows_f23)):
                
                flows_f12_pp.append(post_processing(output_dict['flows_f12'][ii], flow_horizontal_flip(flows_f12[ii])))
                # flows_f23_pp.append(post_processing(output_dict['flows_f23'][ii], flow_horizontal_flip(flows_f23[ii])))
                # flows_b10_pp.append(post_processing(output_dict['flows_b10'][ii], flow_horizontal_flip(flows_b10[ii])))
                # flows_b21_pp.append(post_processing(output_dict['flows_b21'][ii], flow_horizontal_flip(flows_b21[ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(disp_l1[ii], [3])))
                # disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(disp_l2[ii], [3])))

            output_dict['flows_f12_pp'] = flows_f12_pp
            # output_dict['flows_f23_pp'] = flows_f23_pp
            # output_dict['flows_b10_pp'] = flows_b10_pp
            # output_dict['flows_b21_pp'] = flows_b21_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            # output_dict['disp_l2_pp'] = disp_l2_pp
            # output_dict['flow_b_pp'] = flow_b_pp
            # output_dict['disp_l1_pp'] = disp_l1_pp
            # output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict

