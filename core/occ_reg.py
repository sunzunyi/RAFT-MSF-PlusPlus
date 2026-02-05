import torch
import numpy as np
import torch.nn.functional as F
import random
from core.tools import _disp2depth_kitti_K
# Adapted from the homography smoothing in UnSAMFlow

def disp2depth_kitti(pred_disp, k_value): # k_value B -> B 1 1 1

    pred_depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + 1e-6)
    pred_depth = torch.clamp(pred_depth, 1e-3, 80)

    return pred_depth

def get_pixelgrid(b, h, w):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

    ones = torch.ones_like(grid_h)
    pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False) # .cuda()

    return pixelgrid

def pixel2pts(intrinsics, depth):
    b, _, h, w = depth.size()

    pixelgrid = get_pixelgrid(b, h, w).to(depth.device)

    depth_mat = depth.view(b, 1, -1)
    pixel_mat = pixelgrid.view(b, 3, -1)
    pts_mat = torch.matmul(torch.inverse(intrinsics.cpu()).cuda(), pixel_mat) * depth_mat # intrinsics.cpu()
    pts = pts_mat.view(b, -1, h, w)

    return pts, pixelgrid

def pixel2pts_ms(intrinsic, output_disp):
    output_depth = disp2depth_kitti(output_disp, intrinsic[:, 0, 0])
    pts, _ = pixel2pts(intrinsic, output_depth)

    return pts


def RigidMotionLoss_Fullseg(input_dict):
    flow = input_dict['flow']
    B, C, H, W = flow.shape
    DEVICE = flow.device
    k = input_dict['k']
    disp = input_dict['disp']
    full_seg = input_dict['fullseg']
    occ_mask = input_dict['occ_map']

    diff_map = input_dict['diff_map']
    diff_map_mask = diff_map < 0.025
    
    out_depth_l1 = _disp2depth_kitti_K(disp * W, k[:, 0, 0])
    # B 1 H W
    out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
    out_depth_l1_mask = out_depth_l1 < 75
    
    occ_mask = 1-occ_mask.float() # 1 for occluded, 0 for non-occluded
    loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)

    ptl1 = pixel2pts_ms(k, disp * W) # 1 3 H W
    pts_tform_l2 = ptl1 + flow
    ptl1_bz = ptl1.permute(0, 2, 3, 1)
    pts_tform_l2_bz = pts_tform_l2.permute(0, 2, 3, 1)
    for i in range(B):

        # if not full_seg[i].any():
        #     continue
        ## find regions to refine
        n = int(full_seg[i].max().item() + 1)
        if n <= 1:
            continue
        occ_mask_ids = full_seg[i, occ_mask[i].to(bool)].to(int)
        occ_mask_id_count = torch.eye(n, dtype=bool, device=DEVICE)[occ_mask_ids].sum(
            axis=0
        )

        id_order = occ_mask_id_count.argsort(descending=True)
        refine_id = id_order[id_order > 0][
            :6
        ]  # we disregard the `0` mask id because it is just the non-masked region, not one object
        refine_id = refine_id.tolist()

        ## start refining
        ptl1 = ptl1_bz[i:i+1]
        pts_tform_l2 = pts_tform_l2_bz[i:i+1]

        for id in refine_id:
            reliable_mask = (
                (out_depth_l1_mask * diff_map_mask * (1 - occ_mask))[i, full_seg[i] == id].bool().detach()
            )
            if reliable_mask.float().sum() < 4 or reliable_mask.float().mean() < 0.2:
                continue
            full_seg_mask = full_seg[i] == id # 1 h w

            reliable_noocc = full_seg_mask.unsqueeze(0)*(out_depth_l1_mask * diff_map_mask * (1 - occ_mask))[i:i+1]
            depth_mean = (out_depth_l1[i:i+1] * reliable_noocc).sum() / reliable_noocc.sum()

            # # logger.myinfo("depth_mean: %s" % depth_mean)
            if depth_mean > 25:
                continue

            full_seg_mask = full_seg_mask.unsqueeze(0).permute(0, 2, 3, 1).repeat(1, 1, 1, 3)    # .permute(1, 2, 0) # h w 1

            ptl1_mask = ptl1[full_seg_mask].view(-1, 3)
            pts_tform_l2_mask = pts_tform_l2[full_seg_mask].view(-1, 3)

            ptl1_mask_reliable = ptl1_mask[reliable_mask].detach()
            pts_tform_l2_mask_reliable = pts_tform_l2_mask[reliable_mask].detach()

            centroid_A = ptl1_mask_reliable.mean(dim=0) # Nx3 -> 3
            centroid_B = pts_tform_l2_mask_reliable.mean(dim=0)
            H_ori = torch.matmul((ptl1_mask_reliable - centroid_A).T, (pts_tform_l2_mask_reliable - centroid_B))
            U, S, V = torch.svd(H_ori)
            R = torch.matmul(V, U.T).detach()
            if torch.det(R) < 0:
                V[:, -1] = V[:, -1] * -1
                R = torch.matmul(V, U.T).detach()
            t = centroid_B - torch.matmul(R, centroid_A) # 3

            ptl1_mask_tform = torch.matmul(R, ptl1_mask.T) + t.view(-1, 1) # 3xN + 3x1 -> 3xN
            ptl1_mask_tform = ptl1_mask_tform.T
            ptloss = torch.sum((ptl1_mask_tform - pts_tform_l2_mask).abs()) / (H * W)
            loss += ptloss

    return loss / B