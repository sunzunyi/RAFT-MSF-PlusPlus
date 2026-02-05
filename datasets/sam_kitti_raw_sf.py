# Generate SAM segmentation mask for KITTI dataset
# Reference `UnSAMFlow/sam_inference.py`.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from tqdm import tqdm
import numpy as np
from torchvision.utils import draw_segmentation_masks
import torchvision.utils as vutils
import glob
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import random
# from pycocotools import mask as mask_utils
import imageio
import torch
torch.set_num_threads(4)

sam = sam_model_registry["default"](checkpoint="segment-anything/checkpoints/sam_vit_h_4b8939.pth")
sam = sam.cuda()

images_root =  "/your_kitti_root/"


# Fine-tuning requires segmentation annotations from KITTI 2015 MV.
# Please follow the existing code to load dataset and process the segmentation masks.

index_file = open('datasets/index_txt/kitti_train_scenes_all.txt', 'r')
scene_list = [line.rstrip() for line in index_file.readlines()]
scene_list = sorted(scene_list) 

_image_list = []
dir_list = []
view1 = 'image_02/data'
view2 = 'image_03/data'
ext = '.jpg'
for scene in scene_list:
    for view in [view1, view2]:
        date = scene[:10]
        img_dir = os.path.join(images_root, date, scene, view)
        img_list = sorted(glob.glob(img_dir + '/*' + ext))
        for i in range(len(img_list)):
            _image_list.append([img_list[i]])
            dir_list.append([date, scene, view, os.path.basename(img_list[i])[:-4]])
print(f"Total images found: {len(_image_list)}")
print(f"Total directories found: {len(dir_list)}")

print(len(_image_list))

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

total_images = len(_image_list)
for idx, image_l in tqdm(enumerate(_image_list), desc="Processing images", total=total_images):

    image = cv2.imread(image_l[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (375, 1242, 3)
    masks = mask_generator.generate(image)
    # print(len(masks))
    # print(masks[0].keys())
    if len(masks) == 0:
        continue
    
    masks_map = torch.stack([
        torch.from_numpy(mask["segmentation"]).float().cuda() for mask in masks
    ])  # (N, H, W)

    H, W = masks_map.shape[1:]
    masks_map = masks_map.permute(1, 2, 0)  # (H, W, N)

    # Calculate the area of the region
    masks_area = torch.tensor([mask["area"] for mask in masks], dtype=torch.float32).cuda()

    # Remove masks that are equal to the entire image size
    valid_masks = masks_area < H * W
    masks_map = masks_map[:, :, valid_masks]
    masks_area = masks_area[valid_masks]

    # Sort by area in descending order
    area_order = torch.argsort(masks_area, descending=True)
    masks_area = masks_area[area_order]
    masks_map = masks_map[:, :, area_order]

    # Add background mask
    background_mask = torch.ones((H, W, 1), device="cuda")
    masks_map_aug = torch.cat((background_mask, masks_map), dim=-1)  # (H, W, N+1)
    masks_area_aug = torch.cat((torch.tensor([H * W], device="cuda"), masks_area))  # (N+1)

    # Generate unified mask
    unified_mask = torch.argmin(
        masks_map_aug * masks_area_aug[None, None, :] +
        (1 - masks_map_aug) * (H * W + 1),
        dim=-1
    )  # (H, W)

    # Map to unique classes
    unique_classes = torch.unique(unified_mask)
    mapping = torch.zeros(unique_classes.max().item() + 1, device="cuda", dtype=torch.int32)
    for i, cl in enumerate(unique_classes):
        mapping[cl] = i
    new_mask = mapping[unified_mask]  # (H, W)

    if new_mask.max() > 255:  # Rarely occurs
        print(f"More than 256 masks detected for image {dir_list[idx]}")
        new_mask[new_mask > 255] = 0

    new_mask = new_mask.cpu().numpy().astype(np.uint8)

    # Save the result
    save_path = os.path.join('/your_dir/kitti_raw_sam1_scenceflow/', *dir_list[idx]) + '.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, new_mask)
