from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict, get_date_from_width



class KITTI_2015_Test(data.Dataset):
    def __init__(self,
                 args,
                 root):

        self._args = args


        images_l_root = os.path.join(root, "multi_view/testing", "image_2") # "data_scene_flow" , 
        images_r_root = os.path.join(root, "multi_view/testing", "image_3")
        
        ## loading image -----------------------------------
        if not os.path.isdir(images_l_root):
            raise ValueError("Image directory %s not found!", images_l_root)
        if not os.path.isdir(images_r_root):
            raise ValueError("Image directory %s not found!", images_r_root)

        # ----------------------------------------------------------
        # Construct list of indices for training/validation
        # ----------------------------------------------------------
        num_images = 200
        list_of_indices = range(num_images)

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and disp/flow
        # ----------------------------------------------------------
        path_dir = os.path.dirname(os.path.realpath(__file__))
        self._image_list = []
        self._flow_list = []
        self._disp_list = []
        img_ext = '.png'

        for ii in list_of_indices:

            file_idx = '%.6d' % ii

            # im_l0 = os.path.join(images_l_root, file_idx + "_08" + img_ext) # Four-frame model predicts f23    10->11
            # im_l1 = os.path.join(images_l_root, file_idx + "_09" + img_ext)
            # im_l2 = os.path.join(images_l_root, file_idx + "_10" + img_ext)
            # im_l3 = os.path.join(images_l_root, file_idx + "_11" + img_ext)

            # im_r0 = os.path.join(images_r_root, file_idx + "_08" + img_ext)
            # im_r1 = os.path.join(images_r_root, file_idx + "_09" + img_ext)
            # im_r2 = os.path.join(images_r_root, file_idx + "_10" + img_ext)
            # im_r3 = os.path.join(images_r_root, file_idx + "_11" + img_ext)

            im_l0 = os.path.join(images_l_root, file_idx + "_09" + img_ext)  # Three-frame model predicts f12    10->11 , not use _12
            im_l1 = os.path.join(images_l_root, file_idx + "_10" + img_ext)
            im_l2 = os.path.join(images_l_root, file_idx + "_11" + img_ext)
            im_l3 = os.path.join(images_l_root, file_idx + "_12" + img_ext)

            im_r0 = os.path.join(images_r_root, file_idx + "_09" + img_ext)
            im_r1 = os.path.join(images_r_root, file_idx + "_10" + img_ext)
            im_r2 = os.path.join(images_r_root, file_idx + "_11" + img_ext)
            im_r3 = os.path.join(images_r_root, file_idx + "_12" + img_ext)

            # file_list = [im_l0, im_l1, im_l2, im_l3, im_r0, im_r1, im_r2, im_r3]
            # for _, item in enumerate(file_list):
            #     if not os.path.isfile(item):
            #         raise ValueError("File not exist: %s", item)

            self._image_list.append([im_l0, im_l1, im_l2, im_l3, im_r0, im_r1, im_r2, im_r3])

        self._size = len(self._image_list)
        assert len(self._image_list) != 0

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])


    def __getitem__(self, index):

        index = index % self._size
        im_l0_filename = self._image_list[index][0]
        im_l1_filename = self._image_list[index][1]
        im_l2_filename = self._image_list[index][2]
        im_l3_filename = self._image_list[index][3]
        im_r0_filename = self._image_list[index][4]
        im_r1_filename = self._image_list[index][5]
        im_r2_filename = self._image_list[index][6]
        im_r3_filename = self._image_list[index][7]

        # read float32 images and flow
        im_l0_np = read_image_as_byte(im_l0_filename)
        im_l1_np = read_image_as_byte(im_l1_filename)
        im_l2_np = read_image_as_byte(im_l2_filename)
        im_l3_np = read_image_as_byte(im_l3_filename)
        im_r0_np = read_image_as_byte(im_r0_filename)
        im_r1_np = read_image_as_byte(im_r1_filename)
        im_r2_np = read_image_as_byte(im_r2_filename)
        im_r3_np = read_image_as_byte(im_r3_filename)
        
        # example filename
        basename = os.path.basename(im_l2_filename)[:6]

        # find intrinsic
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[get_date_from_width(im_l1_np.shape[1])]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[get_date_from_width(im_r1_np.shape[1])]).float()

        im_l0 = self._to_tensor(im_l0_np)
        im_l1 = self._to_tensor(im_l1_np)
        im_l2 = self._to_tensor(im_l2_np)
        im_l3 = self._to_tensor(im_l3_np)
        im_r0 = self._to_tensor(im_r0_np)
        im_r1 = self._to_tensor(im_r1_np)
        im_r2 = self._to_tensor(im_r2_np)
        im_r3 = self._to_tensor(im_r3_np)

        # input size
        h_orig, w_orig, _ = im_l1_np.shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        example_dict = {
            "input_l0": im_l0,
            "input_l1": im_l1,
            "input_l2": im_l2,
            "input_l3": im_l3,
            "input_r0": im_r0,
            "input_r1": im_r1,
            "input_r2": im_r2,
            "input_r3": im_r3,
            "index": index,
            "basename": basename,
            "input_k_l1": k_l1,
            "input_k_l2": k_l1,
            "input_k_r1": k_r1,
            "input_k_r2": k_r1,
            "input_size": input_im_size
        }

        return example_dict

    def __len__(self):
        return self._size

