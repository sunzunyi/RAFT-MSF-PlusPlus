## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import numpy as np
import colorama
import logging
import collections
import torch
import torch.distributed
import torch.nn as nn

from core import logger, tools
from core.tools import MovingAverage, log

# for evaluation
from utils.flow import flow_to_png_middlebury, write_flow_png, write_depth_png
import matplotlib.pyplot as plt
from skimage import io
import os
from torch.cuda.amp import GradScaler
# for tensorboardX
from torch.utils.tensorboard import SummaryWriter

from core.occ_reg import RigidMotionLoss_Fullseg
from core.tools import _disp2depth_kitti_K
import math
# --------------------------------------------------------------------------------
# Exponential moving average smoothing factor for speed estimates
# Ranges from 0 (average speed) to 1 (current/instantaneous speed) [default: 0.3].
# --------------------------------------------------------------------------------
TQDM_SMOOTHING = 0


# -------------------------------------------------------------------------------------------
# Magic progressbar for inputs of type 'iterable'
# -------------------------------------------------------------------------------------------
def create_progressbar(iterable,
                       desc="",
                       train=False,
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       postfix=False):

    # ---------------------------------------------------------------
    # Pick colors
    # ---------------------------------------------------------------
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    cyan = colorama.Fore.CYAN
    dim = colorama.Style.DIM
    green = colorama.Fore.GREEN

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}:%s " % (cyan, reset, bright, reset)     # description
    bar_format += "{percentage:3.0f}%"                                      # percentage
    bar_format += "%s|{bar}|%s " % (dim, reset)                             # bar
    bar_format += " {n_fmt}/{total_fmt}  "                                  # i/n counter
    bar_format += "{elapsed}<{remaining}"                                   # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}  "                                   # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}  "
    bar_format += "%s{postfix}%s" % (green, reset)                          # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,                          # Prefix for the progress bar
        "total": len(iterable),                # The number of expected iterations
        "leave": True,                         # Leave progress bar when done
        "miniters": 1 if train else None,      # Minimum display update interval in iterations
        "unit": unit,                          # String be used to define the unit of each iteration
        "initial": initial,                    # The initial counter value.
        "dynamic_ncols": True,                 # Allow window resizes
        "smoothing": TQDM_SMOOTHING,           # Moving average smoothing factor for speed estimates
        "bar_format": bar_format,              # Specify a custom bar string formatting
        "position": offset,                    # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close
    }

    return tools.tqdm_with_logging(**tqdm_args)


def tensor2float_dict(tensor_dict):
    return {key: tensor.item() for key, tensor in tensor_dict.items()}


def format_moving_averages_as_progress_dict(moving_averages_dict={},
                                            moving_averages_postfix="avg"):
    progress_dict = collections.OrderedDict([
        (key + moving_averages_postfix, "%1.4f" % moving_averages_dict[key].mean())
        for key in sorted(moving_averages_dict.keys())
    ])
    return progress_dict


def format_learning_rate(lr):
    if np.isscalar(lr):
        return "{}".format(lr)
    else:
        return "{}".format(str(lr[0]) if len(lr) == 1 else lr)


class TrainingEpoch:
    def __init__(self,
                 args,
                 loader,
                 scheduler=None,
                 augmentation=None,
                 tbwriter=None,
                 add_progress_stats={},
                 desc="Training Epoch"):

        self._args = args
        self._desc = desc
        self._loader = loader
        self._augmentation = augmentation
        self._add_progress_stats = add_progress_stats
        self._tbwriter = tbwriter
        self.scaler = GradScaler(enabled=self._args.mixed_precision)
        self.scheduler = scheduler
        self.sam_aug = False
        self.obj_cache = None
        self.start_occ_reg = False
        self.start_chain_aug = False
        self.start_ap_aug = False
        self.arflow_aug = False

    def _step(self, example_dict, model_and_loss, optimizer):

        # Get input and target tensor keys
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # Possibly transfer to Cuda
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=True)

        # Optionally perform augmentations
        if self._augmentation is not None:
            with torch.no_grad():
                example_dict = self._augmentation(example_dict)

        # Convert inputs/targets to variables that require gradients
        for key, tensor in example_dict.items():
            if key in input_keys:
                example_dict[key] = tensor.requires_grad_(True)
            elif key in target_keys:
                example_dict[key] = tensor.requires_grad_(False)

        # Reset gradients
        optimizer.zero_grad()

        loss_dict_total = []
        recons_imgs_total = []
        output_dict_total = []
        occ_dict_total = []
        for bz_idx in range(0, example_dict['input_l0_aug'].shape[0], 4): # Use gradient accumulation if GPU memory is insufficient
            example_dict_bz = {}      #     {}    example_dict
            for key, tensor in example_dict.items():
                if isinstance(tensor, (int, float, list)):
                    example_dict_bz[key] = tensor
                else:
                    example_dict_bz[key] = tensor[bz_idx:bz_idx+4]

            loss_dict, recons_imgs, output_dict, occ_dict = model_and_loss(example_dict_bz)
            loss_dict_total.append(loss_dict)
            recons_imgs_total.append(recons_imgs)
            output_dict_total.append(output_dict)
            occ_dict_total.append(occ_dict)

            # Check total_loss for NaNs
            training_loss = loss_dict[self._args.training_key]

            if self.start_occ_reg:
                store_dict = {
                    'fullseg': example_dict_bz['input_sam_fullseg_l1_aug'],
                    'k': example_dict_bz['input_k_l1_aug'],
                    'flow': output_dict['flows_f12'][-1],
                    'disp': output_dict['disp_l1'][-1],
                    'occ_map': occ_dict["occ_map_f"],
                    'diff_map': occ_dict['pts_diff1_aug'],
                    'img': example_dict_bz['input_l1_aug'],
                }
                
                rm_loss_f = RigidMotionLoss_Fullseg(store_dict) * 1.0
                training_loss += rm_loss_f
                loss_dict['rmlf'] = rm_loss_f

                store_dict = {
                    'fullseg': example_dict_bz['input_sam_fullseg_l2_aug'],
                    'k': example_dict_bz['input_k_l2_aug'],
                    'flow': output_dict['flows_b21'][-1],
                    'disp': output_dict['disp_l2'][-1],
                    'occ_map': occ_dict["occ_map_b"],
                    'diff_map': occ_dict['pts_diff2_aug'],
                    'img': example_dict_bz['input_l2_aug'],
                }
                rm_loss_b = RigidMotionLoss_Fullseg(store_dict) * 1.0
                training_loss += rm_loss_b
                loss_dict['rmlb'] = rm_loss_b


            self.scaler.scale(training_loss).backward()


        self.scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model_and_loss._model.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        if self._args.lr_scheduler == 'OneCycleLR':
            self.scheduler.step()

        loss_dict = {}
        recons_imgs = {}
        output_dict = {}
        occ_dict = {}
        for key in loss_dict_total[0].keys():
            loss_dict[key] = torch.mean(torch.stack([d[key] for d in loss_dict_total]), dim=0)

        # Return success flag, loss and output dictionary
        return loss_dict, recons_imgs, output_dict

    def run(self, model_and_loss, optimizer, epoch, mode):

        n_iter = 0
        model_and_loss.train()
        moving_averages_dict = None
        progressbar_args = {
            "iterable": self._loader,
            "desc": self._desc,
            "train": True,
            "offset": 0,
            "logging_on_update": False,
            "logging_on_close": True,
            "postfix": True
        }

        # Perform training steps
        with create_progressbar(**progressbar_args) as progress:
            for i, example_dict in enumerate(progress):

                n_iter = i + ((epoch-1) * len(progress))
                # perform step
                example_dict['curr_epoch'] = epoch
                loss_dict_per_step, recons_imgs, output_dict = self._step(example_dict, model_and_loss, optimizer)
                # convert
                loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                # Possibly initialize moving averages and  Add moving averages
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: MovingAverage() for key in loss_dict_per_step.keys()
                    }

                for key, loss in loss_dict_per_step.items():
                    if math.isnan(loss): # if torch.isnan(loss):
                        continue
                    if key not in moving_averages_dict:
                        moving_averages_dict[key] = MovingAverage()
                    moving_averages_dict[key].add_average(loss, addcount=self._args.batch_size)

                # view statistics in progress bar
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix="")    # "_ema"

                progress.set_postfix(progress_stats)

                # if self._args.evaluation is False and self._args.is_main_process:
                #     if n_iter % self._args.tbIter == 0:
                #         log(self._tbwriter, output_dict, moving_averages_dict, example_dict, recons_imgs, n_iter, mode)


        # Return loss and output dictionary
        ema_loss_dict = { key: ma.mean() for key, ma in moving_averages_dict.items() }
        return ema_loss_dict, output_dict


class EvaluationEpoch:
    def __init__(self,
                 args,
                 loader,
                 augmentation=None,
                 tbwriter=None,
                 add_progress_stats={},
                 desc="Evaluation Epoch"):
        self._args = args
        self._desc = desc
        self._loader = loader
        self._add_progress_stats = add_progress_stats
        self._augmentation = augmentation
        self._tbwriter = tbwriter
        self._save_output = False
        if self._args.save_flow or self._args.save_disp or self._args.save_disp2:
            self._save_output = True

    def save_outputs(self, example_dict, output_dict):

        # save occ
        save_root_flow = self._args.save + '/save/flow/'
        save_root_disp = self._args.save + '/save/disp_0/'
        save_root_disp2 = self._args.save + '/save/disp_1/'

        if self._args.save_flow:
            out_flow = output_dict["out_flow_pp"].data.cpu().numpy() # predicted flow
            # out_flow = example_dict["target_flow"].data.cpu().numpy() # gt flow
            b_size = output_dict["out_flow_pp"].data.size(0)
            file_names_flow = []
            for ii in range(0, b_size):
                file_name_flow = save_root_flow + '/' + str(example_dict["basename"][ii])
                file_names_flow.append(file_name_flow)
                directory_flow = os.path.dirname(file_name_flow)
                if not os.path.exists(directory_flow):
                    os.makedirs(directory_flow)
                    print(directory_flow, ' has been created.')

            for ii in range(0, b_size):
                # # Vis
                # flow_f_rgb = flow_to_png_middlebury(out_flow[ii, ...])
                # file_name_flo_vis = file_names_flow[ii] + '_flow.png'
                # io.imsave(file_name_flo_vis, flow_f_rgb)
                # Png output
                file_name = file_names_flow[ii] + '_10.png'
                write_flow_png(file_name, out_flow[ii, ...].swapaxes(0, 1).swapaxes(1, 2))

                if "out_flow_pp_b" in output_dict:
                    out_flow_b = output_dict["out_flow_pp_b"].data.cpu().numpy()
                    file_name = file_names_flow[ii] + '_10_b.png'
                    write_flow_png(file_name, out_flow_b[ii,...].swapaxes(0, 1).swapaxes(1, 2))

        if self._args.save_disp:

            b_size = output_dict["out_disp_l_pp"].data.size(0)
            out_disp = output_dict["out_disp_l_pp"].data.cpu().numpy() # predicted disp
            # out_disp = example_dict["target_disp"].data.cpu().numpy() # gt disp
            file_names_disp = []

            for ii in range(0, b_size):
                file_name_disp = save_root_disp + '/' + str(example_dict["basename"][ii])
                file_names_disp.append(file_name_disp)
                directory_disp = os.path.dirname(file_name_disp)
                if not os.path.exists(directory_disp):
                    os.makedirs(directory_disp)
                    print(directory_disp, ' has been created.')

            for ii in range(0, b_size):
                # # Vis
                # disp_ii = out_disp[ii, 0, ...]
                # norm_disp = (disp_ii / disp_ii.max() * 255).astype(np.uint8)
                # plt.imsave(file_names_disp[ii] + '_disp.jpg', norm_disp, cmap='plasma')
                # Png
                file_name = file_names_disp[ii] + '_10.png'
                write_depth_png(file_name, out_disp[ii, 0, ...])


        if self._args.save_disp2:

            b_size = output_dict["out_disp_l_pp_next"].data.size(0)
            out_disp2 = output_dict["out_disp_l_pp_next"].data.cpu().numpy()
            #out_disp2 = output_dict["target_disp2_occ"] # gt disp2
            file_names_disp2 = []

            for ii in range(0, b_size):
                file_name_disp2 = save_root_disp2 + '/' + str(example_dict["basename"][ii])
                file_names_disp2.append(file_name_disp2)
                directory_disp2 = os.path.dirname(file_name_disp2)
                if not os.path.exists(directory_disp2):
                    os.makedirs(directory_disp2)
                    print(directory_disp2, ' has been created.')

            for ii in range(0, b_size):
                # # Vis
                # disp2_ii = out_disp2[ii, 0, ...]
                # norm_disp2 = (disp2_ii / disp2_ii.max() * 255).astype(np.uint8)
                # plt.imsave(file_names_disp2[ii] + '_disp2.jpg', norm_disp2, cmap='plasma')
                # Png
                file_name2 = file_names_disp2[ii] + '_10.png'
                write_depth_png(file_name2, out_disp2[ii, 0, ...])
                
                if "out_disp_l_pp_next_b" in output_dict:
                    out_disp2_b = output_dict["out_disp_l_pp_next_b"].data.cpu().numpy()
                    file_name2 = file_names_disp2[ii] + '_10_b.png'
                    write_depth_png(file_name2, out_disp2_b[ii, 0, ...])


    def _step(self, example_dict, model_and_loss):

        # Get input and target tensor keys
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # Possibly transfer to Cuda
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=True)

        # Optionally perform augmentations
        if self._augmentation is not None:
            example_dict = self._augmentation(example_dict)

        # Run forward pass to get losses and outputs.
        loss_dict, recons_imgs, output_dict, _ = model_and_loss(example_dict)

        return loss_dict, recons_imgs, output_dict

    def run(self, model_and_loss, epoch):

        with torch.no_grad():

            # Tell model that we want to evaluate
            model_and_loss.eval()

            # Keep track of moving averages
            moving_averages_dict = None

            # Progress bar arguments
            progressbar_args = {
                "iterable": self._loader,
                "desc": self._desc,
                "train": False,
                "offset": 0,
                "logging_on_update": False,
                "logging_on_close": True,
                "postfix": True
            }

            # Perform evaluation steps
            with create_progressbar(**progressbar_args) as progress:
                for example_dict in progress:

                    # Perform forward evaluation step
                    loss_dict_per_step, recons_imgs, output_dict = self._step(example_dict, model_and_loss)

                    # Save results
                    if self._save_output:
                        self.save_outputs(example_dict, output_dict)

                    # Convert loss dictionary to float
                    loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                    # Possibly initialize moving averages
                    if moving_averages_dict is None:
                        moving_averages_dict = {
                            key: MovingAverage() for key in loss_dict_per_step.keys()
                        }

                    # Add moving averages
                    for key, loss in loss_dict_per_step.items():
                        moving_averages_dict[key].add_average(loss, addcount=self._args.batch_size_val)

                    # view statistics in progress bar
                    progress_stats = format_moving_averages_as_progress_dict(
                        moving_averages_dict=moving_averages_dict,
                        moving_averages_postfix="")

                    progress.set_postfix(progress_stats)

            # Record average losses
            avg_loss_dict = { key: ma.mean() for key, ma in moving_averages_dict.items() }
            
            return avg_loss_dict, output_dict

from torch.utils.data import ConcatDataset
def exec_runtime(args,
                 checkpoint_saver,
                 model_and_loss,
                 optimizer,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 sampler_train,
                 training_augmentation,
                 validation_augmentation):

    # ----------------------------------------------------------------------------------------------
    # Tensorboard writer
    # ----------------------------------------------------------------------------------------------
    
    if args.evaluation is False and args.is_main_process:
        tensorBoardWriter = SummaryWriter(args.save + '/writer')
    else:
        tensorBoardWriter = None


    if train_loader is not None:
        training_module = TrainingEpoch(
            args, 
            desc="   Train",
            loader=train_loader,
            scheduler=lr_scheduler,
            augmentation=training_augmentation,
            tbwriter=tensorBoardWriter)

    if validation_loader is not None:
        evaluation_module = EvaluationEpoch(
            args,
            desc="Validate",
            loader=validation_loader,
            augmentation=validation_augmentation,
            tbwriter=tensorBoardWriter)

    # --------------------------------------------------------
    # Log some runtime info
    # --------------------------------------------------------
    with logger.LoggingBlock("Runtime", emph=True):
        logger.myinfo("start_epoch: %i" % args.start_epoch)
        logger.myinfo("total_epochs: %i" % args.total_epochs)

    # ---------------------------------------
    # Total progress bar arguments
    # ---------------------------------------
    progressbar_args = {
        "desc": "Progress",
        "initial": args.start_epoch - 1,
        "invert_iterations": True,
        "iterable": range(1, args.total_epochs + 1),
        "logging_on_close": True,
        "logging_on_update": True,
        "postfix": False,
        "unit": "ep"
    }

    # --------------------------------------------------------
    # Total progress bar
    # --------------------------------------------------------
    print(''), logging.logbook('')
    total_progress = create_progressbar(**progressbar_args)
    print("\n")

    # --------------------------------------------------------
    # Remember validation loss
    # --------------------------------------------------------
    best_validation_loss = float("inf") if args.validation_key_minimize else -float("inf")
    best_epoch = 0
    store_as_best = False


    for epoch in range(args.start_epoch, args.total_epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        if args.start_occ_reg <= epoch:
            datasets_to_check = []
            if isinstance(training_module._loader.dataset, ConcatDataset):
                datasets_to_check = training_module._loader.dataset.datasets
            else:
                datasets_to_check = [training_module._loader.dataset]

            for dataset in datasets_to_check:
                if hasattr(dataset, 'start_occ_reg'):
                    dataset.start_occ_reg = True
                    logger.myinfo(f"Start occ reg for {dataset.__class__.__name__}")

            training_module._loader.dataset.start_occ_reg = True
            training_module.start_occ_reg = True
            logger.myinfo("Start occ reg from epoch %i , The set value is %i" % (epoch, args.start_occ_reg))
            args.start_occ_reg = 10000

        with logger.LoggingBlock("Epoch %i/%i" % (epoch, args.total_epochs), emph=True):

            # --------------------------------------------------------
            # Always report learning rate
            # --------------------------------------------------------
            if lr_scheduler is None:
                logger.myinfo("lr: %s" % format_learning_rate(args.optimizer_lr))
            else:
                logger.myinfo("lr: %s" % format_learning_rate(lr_scheduler.get_last_lr()[0]))

            # -------------------------------------------
            # Create and run a training epoch
            # -------------------------------------------
            if train_loader is not None:
                mode = 'train'
                avg_loss_dict, _ = training_module.run(model_and_loss=model_and_loss, optimizer=optimizer, epoch=epoch, mode=mode)

                if args.evaluation is False and args.is_main_process:
                    tensorBoardWriter.add_scalar('Train/Loss', avg_loss_dict[args.training_key], epoch)
            # torch.cuda.empty_cache()
            # -------------------------------------------
            # Create and run a validation epoch
            # -------------------------------------------
            if validation_loader is not None and args.is_main_process:  # and args.is_main_process

                if hasattr(model_and_loss._model, 'require_forward_param_sync'):
                    save_flag = model_and_loss._model.require_forward_param_sync
                    model_and_loss._model.require_forward_param_sync = False

                # ---------------------------------------------------
                # Construct holistic recorder for epoch
                # ---------------------------------------------------
                avg_loss_dict, output_dict = evaluation_module.run(model_and_loss=model_and_loss, epoch=epoch)

                if hasattr(model_and_loss._model, 'require_forward_param_sync'):
                    model_and_loss._model.require_forward_param_sync = save_flag

                # --------------------------------------------------------
                # Tensorboard X writing
                # --------------------------------------------------------
                if args.evaluation is False and args.is_main_process:
                    tensorBoardWriter.add_scalar('Val/Metric', avg_loss_dict[args.validation_key], epoch)

                # ----------------------------------------------------------------
                # Evaluate whether this is the best validation_loss
                # ----------------------------------------------------------------
                validation_loss = avg_loss_dict[args.validation_key]
                if args.validation_key_minimize:
                    store_as_best = validation_loss < best_validation_loss
                else:
                    store_as_best = validation_loss > best_validation_loss
                if store_as_best:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                
                # print validation loss and best loss, epoch
                logger.myinfo("Validation Loss: %1.4f at epoch %i, Best Validation Loss: %1.4f at epoch %i" % 
                              (validation_loss, epoch, best_validation_loss, best_epoch))
            # torch.cuda.empty_cache()
            # --------------------------------------------------------
            # Update standard learning scheduler
            # --------------------------------------------------------
            if lr_scheduler is not None and (args.lr_scheduler == 'MultiStepLR' or args.lr_scheduler == 'CosineAnnealingLR'):
                lr_scheduler.step(epoch)


            # ----------------------------------------------------------------
            # Also show best loss on total_progress
            # ----------------------------------------------------------------
            total_progress_stats = {
                "best_" + args.validation_key + "_avg": "%1.4f" % best_validation_loss
            }
            total_progress.set_postfix(total_progress_stats)

            # # ----------------------------------------------------------------
            # # Bump total progress
            # # ----------------------------------------------------------------
            total_progress.update()
            print('')

            # ----------------------------------------------------------------
            # Store checkpoint
            # ----------------------------------------------------------------
            if checkpoint_saver is not None and args.is_main_process:
                directory = args.save
                checkpoint_saver.save_latest(
                    directory=directory,
                    model_and_loss=model_and_loss,
                    stats_dict=dict(avg_loss_dict, epoch=epoch, validation_loss = validation_loss),
                    store_as_best=store_as_best,
                    lr_scheduler=lr_scheduler,
                    optimizer=optimizer,
                    )

            # ----------------------------------------------------------------
            # Vertical space between epochs
            # ----------------------------------------------------------------
            print(''), logging.logbook('')

    # ----------------------------------------------------------------
    # Finish
    # ----------------------------------------------------------------
    if args.evaluation is False and args.is_main_process:
        tensorBoardWriter.close()

    total_progress.close()
    logger.myinfo("Finished.")
