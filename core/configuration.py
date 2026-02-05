## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import logging
import shutil
import random
import fnmatch

from core import logger, tools

from torch.utils.data.sampler import RandomSampler
from datasets.custom_batchsampler import CustomBatchSampler

# ---------------------------------------------------
# Class that contains both the network model and loss
# ---------------------------------------------------
class ModelAndLoss(nn.Module):
    def __init__(self, args, model, training_loss, evaluation_loss=None):
        super(ModelAndLoss, self).__init__()
        self._model = model
        self._training_loss = training_loss
        self._evaluation_loss = evaluation_loss

    @property
    def training_loss(self):
        return self._training_loss

    @property
    def evaluation_loss(self):
        return self._evaluation_loss

    @property
    def model(self):
        return self._model

    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    # -------------------------------------------------------------
    # Note: We merge inputs and targets into a single dictionary !
    # -------------------------------------------------------------
    def forward(self, example_dict):
        # -------------------------------------
        # Run forward pass
        # -------------------------------------
        output_dict = self._model(example_dict)

        # -------------------------------------
        # Compute losses
        # -------------------------------------
        occ_dict = {}
        if self.training:
            loss_dict, recons_imgs, occ_dict = self._training_loss(output_dict, example_dict)
        else:
            loss_dict, recons_imgs = self._evaluation_loss(output_dict, example_dict)

        # -------------------------------------
        # Return losses and outputs
        # -------------------------------------
        return loss_dict, recons_imgs, output_dict, occ_dict


def configure_runtime_augmentations(args):
    with logger.LoggingBlock("Runtime Augmentations", emph=True):

        training_augmentation = None
        validation_augmentation = None

        # ----------------------------------------------------
        # Training Augmentation
        # ----------------------------------------------------
        if args.training_augmentation is not None:
            kwargs = tools.kwargs_from_args(args, "training_augmentation")
            logger.myinfo("training_augmentation: %s" % args.training_augmentation)
            for param, default in sorted(kwargs.items()):
                logger.myinfo("  %s: %s" % (param, default))
            kwargs["args"] = args
            training_augmentation = tools.instance_from_kwargs(
                args.training_augmentation_class, kwargs)
            if args.cuda:
                training_augmentation = training_augmentation.cuda()

        else:
            logger.myinfo("training_augmentation: None")

        # ----------------------------------------------------
        # Validation Augmentation
        # ----------------------------------------------------
        if args.validation_augmentation is not None:
            kwargs = tools.kwargs_from_args(args, "validation_augmentation")
            logger.myinfo("validation_augmentation: %s" % args.validation_augmentation)
            for param, default in sorted(kwargs.items()):
                logger.myinfo("  %s: %s" % (param, default))
            kwargs["args"] = args
            validation_augmentation = tools.instance_from_kwargs(
                args.validation_augmentation_class, kwargs)
            if args.cuda:
                validation_augmentation = validation_augmentation.cuda()

        else:
            logger.myinfo("validation_augmentation: None")

    return training_augmentation, validation_augmentation


def configure_model_and_loss(args):

    # ----------------------------------------------------
    # Dynamically load model and loss class with parameters
    # passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    # ----------------------------------------------------
    device = torch.device(args.device)
    with logger.LoggingBlock("Model and Loss", emph=True):

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------
        kwargs = tools.kwargs_from_args(args, "model")
        kwargs["args"] = args
        model = tools.instance_from_kwargs(args.model_class, kwargs)
        model.to(device)
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)  #  , broadcast_buffers=False
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False)  #  , broadcast_buffers=False
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.myinfo('******number of params ins TransDOSE:%d' % n_parameters)

        # ----------------------------------------------------
        # Training loss
        # ----------------------------------------------------
        training_loss = None
        if args.training_loss is not None:
            kwargs = tools.kwargs_from_args(args, "training_loss")
            kwargs["args"] = args
            training_loss = tools.instance_from_kwargs(args.training_loss_class, kwargs).to(device)

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------
        validation_loss = None
        if args.validation_loss is not None:
            kwargs = tools.kwargs_from_args(args, "validation_loss")
            kwargs["args"] = args
            validation_loss = tools.instance_from_kwargs(args.validation_loss_class, kwargs).to(device)

        # ----------------------------------------------------
        # Model and loss
        # ----------------------------------------------------
        model_and_loss = ModelAndLoss(args, model, training_loss, validation_loss)

        # -----------------------------------------------------------
        # If Cuda, transfer model to Cuda and wrap with DataParallel.
        # -----------------------------------------------------------
        # ---------------------------------------------------------------
        # Report some network statistics
        # ---------------------------------------------------------------
        logger.myinfo("Batch Size: %i" % args.batch_size)
        logger.myinfo("GPGPU: Cuda") if args.cuda else logger.myinfo("GPGPU: off")
        logger.myinfo("Network: %s" % args.model)
        logger.myinfo("Number of parameters: %i" % tools.x2module(model_and_loss).num_parameters())
        if training_loss is not None:
            logger.myinfo("Training Key: %s" % args.training_key)
            logger.myinfo("Training Loss: %s" % args.training_loss)
        if validation_loss is not None:
            logger.myinfo("Validation Key: %s" % args.validation_key)
            logger.myinfo("Validation Loss: %s" % args.validation_loss)

    return model_and_loss


def configure_random_seed(args):
    with logger.LoggingBlock("Random Seeds", emph=True):
        base_seed = args.seed
        rank = args.rank if args.distributed else 0

        seed = base_seed + rank * 10 +2

        # python
        random.seed(seed)
        logger.myinfo(f"Rank {rank} Python seed: {seed}")

        # numpy
        seed_np = seed + 1
        np.random.seed(seed_np)
        logger.myinfo(f"Rank {rank} Numpy seed: {seed_np}")

        # torch CPU
        seed_torch_cpu = seed + 2
        torch.manual_seed(seed_torch_cpu)
        logger.myinfo(f"Rank {rank} Torch CPU seed: {seed_torch_cpu}")

        # torch CUDA (single GPU)
        seed_torch_cuda = seed + 3
        torch.cuda.manual_seed(seed_torch_cuda)
        logger.myinfo(f"Rank {rank} Torch CUDA seed: {seed_torch_cuda}")

        # # torch CUDA all GPUs
        # seed_torch_cuda_all = seed + 4
        # torch.cuda.manual_seed_all(seed_torch_cuda_all)
        # logger.myinfo(f"Rank {rank} Torch CUDA all seed: {seed_torch_cuda_all}")



# --------------------------------------------------------------------------
# Checkpoint loader/saver.
# --------------------------------------------------------------------------
class CheckpointSaver:
    def __init__(self,
                 prefix="checkpoint",
                 latest_postfix="_latest",
                 best_postfix="_best",
                 model_key="state_dict",
                 extension=".ckpt"):

        self._prefix = prefix
        self._model_key = model_key
        self._latest_postfix = latest_postfix
        self._best_postfix = best_postfix
        self._extension = extension

    # the purpose of rewriting the loading function is we sometimes want to
    # initialize parameters in modules without knowing the dimensions at runtime
    #
    # This function here will resize these parameters to whatever size required.
    #
    def stripModule(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k.replace(".module", "")
            new_state_dict[k] = v
        return new_state_dict

    def _load_state_dict_into_module(self, state_dict, module, strict=True):
        is_parallel = isinstance(module._model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
        if not is_parallel:
            state_dict = self.stripModule(state_dict)

        module.load_state_dict(state_dict, strict=True)

        # model_state = module.state_dict()
        # # filter out the keys that are not in the model
        # filtered_checkpoint = {k: v for k, v in state_dict.items()
        #                     if k in model_state and model_state[k].shape == v.shape}

        # model_state.update(filtered_checkpoint)
        # module.load_state_dict(model_state, strict=False)
        
    def restore(self, filename, model_and_loss, include_params="*", exclude_params=()):
        # -----------------------------------------------------------------------------------------
        # Make sure file exists
        # -----------------------------------------------------------------------------------------
        if not os.path.isfile(filename):
            logger.myinfo("Could not find checkpoint file '%s'!" % filename)
            quit()

        # -----------------------------------------------------------------------------------------
        # Load checkpoint from file including the state_dict
        # -----------------------------------------------------------------------------------------
        checkpoint_with_state = torch.load(filename, map_location='cuda')

        # -----------------------------------------------------------------------------------------
        # Load filtered state dictionary
        # -----------------------------------------------------------------------------------------
        state_dict = checkpoint_with_state[self._model_key]
        restore_keys = tools.filter_list_of_strings(
            state_dict.keys(),
            include=include_params,
            exclude=exclude_params)
        state_dict = {key: value for key, value in state_dict.items() if key in restore_keys}
        self._load_state_dict_into_module(state_dict, model_and_loss)
        logger.myinfo("  Restore keys:")
        for key in restore_keys:
            logger.myinfo("    %s" % key)

        # -----------------------------------------------------------------------------------------
        # Get checkpoint statistics without the state dict
        # -----------------------------------------------------------------------------------------
        checkpoint_stats = {
            key: value for key, value in checkpoint_with_state.items() if key != self._model_key
        }

        return checkpoint_stats, filename

    def restore_latest(self, directory, model_and_loss, include_params="*", exclude_params=()):
        latest_checkpoint_filename = os.path.join(
            directory, self._prefix + self._latest_postfix + self._extension)
        return self.restore(latest_checkpoint_filename, model_and_loss, include_params, exclude_params)

    def restore_best(self, directory, model_and_loss, include_params="*", exclude_params=()):
        best_checkpoint_filename = os.path.join(
            directory, self._prefix + self._best_postfix + self._extension)
        return self.restore(best_checkpoint_filename, model_and_loss, include_params, exclude_params)

    def save_latest(self, directory, model_and_loss, stats_dict, store_as_best=False, lr_scheduler=None, optimizer=None):
        # -----------------------------------------------------------------------------------------
        # Make sure directory exists
        # -----------------------------------------------------------------------------------------
        tools.ensure_dir(directory)

        # -----------------------------------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------------------------------
        save_dict = dict(stats_dict)
        save_dict[self._model_key] = model_and_loss.state_dict()
        save_dict["optimizer"] = optimizer.state_dict() if optimizer is not None else None
        save_dict["lr_scheduler"] = lr_scheduler.state_dict() if lr_scheduler is not None else None

        # latest_checkpoint_filename = os.path.join(directory, self._prefix + self._latest_postfix + self._extension)
        latest_checkpoint_filename = directory + '/latest.ckpt'
        torch.save(save_dict, latest_checkpoint_filename)

        # -----------------------------------------------------------------------------------------
        # Possibly store as best
        # -----------------------------------------------------------------------------------------
        if store_as_best:
            # best_checkpoint_filename = os.path.join(directory, self._prefix + self._best_postfix + self._extension)
            best_checkpoint_filename = directory + f"/best-{save_dict['epoch']}-{save_dict['validation_loss']:.3f}.ckpt"
            logger.myinfo("Saved checkpoint as best model..")
            shutil.copyfile(latest_checkpoint_filename, best_checkpoint_filename)

            ckpts = sorted([x for x in os.listdir(directory) if "best-" in x], key=lambda x: int(x.split('-')[1]), reverse=False)
            if len(ckpts) >= 10:
                os.remove(os.path.join(directory, ckpts[0]))

def configure_checkpoint_saver(args, model_and_loss):
    with logger.LoggingBlock("Checkpoint", emph=True):
        checkpoint_saver = CheckpointSaver()
        checkpoint_stats = None

        if args.checkpoint is None:
            logger.myinfo("No checkpoint given.")
            logger.myinfo("Starting from scratch with random initialization.")

        elif os.path.isfile(args.checkpoint):
            checkpoint_stats, filename = checkpoint_saver.restore(
                filename=args.checkpoint,
                model_and_loss=model_and_loss,
                include_params=args.checkpoint_include_params,
                exclude_params=args.checkpoint_exclude_params)

        elif os.path.isdir(args.checkpoint):
            if args.checkpoint_mode in ["resume_from_best"]:
                logger.myinfo("Loading best checkpoint in %s" % args.checkpoint)
                checkpoint_stats, filename = checkpoint_saver.restore_best(
                    directory=args.checkpoint,
                    model_and_loss=model_and_loss,
                    include_params=args.checkpoint_include_params,
                    exclude_params=args.checkpoint_exclude_params)

            elif args.checkpoint_mode in ["resume_from_latest"]:
                logger.myinfo("Loading latest checkpoint in %s" % args.checkpoint)
                checkpoint_stats, filename = checkpoint_saver.restore_latest(
                    directory=args.checkpoint,
                    model_and_loss=model_and_loss,
                    include_params=args.checkpoint_include_params,
                    exclude_params=args.checkpoint_exclude_params)
            else:
                logger.myinfo("Unknown checkpoint_restore '%s' given!" % args.checkpoint_restore)
                quit()
        else:
            logger.myinfo("Could not find checkpoint file or directory '%s'" % args.checkpoint)
            quit()

    return checkpoint_saver, checkpoint_stats


# -------------------------------------------------------------------------------------------------
# Configure data loading
# -------------------------------------------------------------------------------------------------
def configure_data_loaders(args):
    with logger.LoggingBlock("Datasets", emph=True):

        def _sizes_to_str(value):
            if np.isscalar(value):
                return '[1L]'
            else:
                return ' '.join([str([d for d in value.size()])])

        def _log_statistics(dataset, prefix, name):
            with logger.LoggingBlock("%s Dataset: %s" % (prefix, name)):
                example_dict = dataset[0]  # get sizes from first dataset example
                for key, value in sorted(example_dict.items()):
                    if key in ["index", "basename"]:  # no need to display these
                        continue
                    if isinstance(value, str):
                        logger.myinfo("{}: {}".format(key, value))
                    else:
                        logger.myinfo("%s: %s" % (key, _sizes_to_str(value)))
                logger.myinfo("num_examples: %i" % len(dataset))

        # -----------------------------------------------------------------------------------------
        # GPU parameters
        # -----------------------------------------------------------------------------------------
        gpuargs = {"num_workers": args.num_workers, "pin_memory": True} if args.cuda else {}

        train_loader = None
        validation_loader = None
        inference_loader = None

        # -----------------------------------------------------------------------------------------
        # Training dataset
        # -----------------------------------------------------------------------------------------
        if args.training_dataset is not None:

            # ----------------------------------------------
            # Figure out training_dataset arguments
            # ----------------------------------------------
            kwargs = tools.kwargs_from_args(args, "training_dataset")
            kwargs["is_cropped"] = True
            kwargs["args"] = args

            # ----------------------------------------------
            # Create training dataset
            # ----------------------------------------------
            train_dataset = tools.instance_from_kwargs(args.training_dataset_class, kwargs)
            if args.distributed:
                sampler_train = DistributedSampler(train_dataset, shuffle=True)
            else:
                sampler_train = None

            # ----------------------------------------------
            # Create training loader
            # ----------------------------------------------            
            if args.training_dataset == 'KITTI_Comb_Train' or args.training_dataset == 'KITTI_Comb_Full' :
                if args.distributed:
                    sampler1 = DistributedSampler(train_dataset.dataset1, shuffle=True)
                    sampler2 = DistributedSampler(train_dataset.dataset2, shuffle=True)
                    custom_batch_sampler = CustomBatchSampler([sampler1, sampler2])
                else:
                    custom_batch_sampler = CustomBatchSampler([RandomSampler(train_dataset.dataset1), RandomSampler(train_dataset.dataset2)])
                train_loader = DataLoader(dataset=train_dataset, batch_sampler=custom_batch_sampler, **gpuargs)

            else:
                # train_loader = DataLoader(
                #     train_dataset,
                #     batch_size=args.batch_size,
                #     sampler=sampler_train,
                #     drop_last=True,
                #     **gpuargs)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    sampler=sampler_train,
                    shuffle=(sampler_train is None),
                    drop_last=True,
                    **gpuargs)

            _log_statistics(train_dataset, prefix="Training", name=args.training_dataset)

        # -----------------------------------------------------------------------------------------
        # Validation dataset
        # -----------------------------------------------------------------------------------------
        if args.validation_dataset is not None:
            
            if args.validation_dataset == 'KITTI_2015_Train_Full_mnsf' or args.validation_dataset == 'KITTI_2015_Train_Full_mnsf_4f' \
                                                                            or args.validation_dataset == 'KITTI_Comb_Val':
                args.eval_on_train = True
            else:
                args.eval_on_train = False
            # ----------------------------------------------
            # Figure out validation_dataset arguments
            # ----------------------------------------------
            kwargs = tools.kwargs_from_args(args, "validation_dataset")
            kwargs["is_cropped"] = True
            kwargs["args"] = args

            # ----------------------------------------------
            # Create validation dataset
            # ----------------------------------------------
            validation_dataset = tools.instance_from_kwargs(args.validation_dataset_class, kwargs)
            
            # if args.distributed:
            #     sampler_eval = DistributedSampler(validation_dataset, shuffle=False)
            # else:
            sampler_eval = None

            # ----------------------------------------------
            # Create validation loader
            # ----------------------------------------------
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size_val,
                sampler=sampler_eval,
                shuffle=(sampler_eval is not None),
                drop_last=False,
                **gpuargs)

            _log_statistics(validation_dataset, prefix="Validation", name=args.validation_dataset)

        if args.training_dataset is None:
            sampler_train = None
    return train_loader, validation_loader, sampler_train


# ------------------------------------------------------------
# Generator for trainable parameters by pattern matching
# ------------------------------------------------------------
def _print_trainable_params(model_and_loss, match="*"):
    sum = 0
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                logger.myinfo(name)
                logger.myinfo(str(p.numel()))
                print(name)
                print(p.numel())
                sum += p.numel()
    logger.myinfo(str(sum))

def _generate_trainable_params(model_and_loss, match="*"):
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                yield p


def _param_names_and_trainable_generator(model_and_loss, match="*"):
    names = []
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                names.append(name)

    return names, _generate_trainable_params(model_and_loss, match=match)


# -------------------------------------------------------------------------------------------------
# Build optimizer:
# -------------------------------------------------------------------------------------------------
def configure_optimizer(args, model_and_loss):
    optimizer = None
    with logger.LoggingBlock("Optimizer", emph=True):
        if args.optimizer is not None:
            if model_and_loss.num_parameters() == 0:
                logger.myinfo("No trainable parameters detected.")
                logger.myinfo("Setting optimizer to None.")
            else:
                logger.myinfo(args.optimizer)

                # -------------------------------------------
                # Figure out all optimizer arguments
                # -------------------------------------------
                all_kwargs = tools.kwargs_from_args(args, "optimizer")
                # all_kwargs['betas'] = (0.5, 0.999)  # default for Adam

                # -------------------------------------------
                # Get the split of param groups
                # -------------------------------------------
                kwargs_without_groups = {
                    key: value for key,value in all_kwargs.items() if key != "group"
                }
                param_groups = all_kwargs["group"]

                # ----------------------------------------------------------------------
                # Print arguments (without groups)
                # ----------------------------------------------------------------------
                for param, default in sorted(kwargs_without_groups.items()):
                    logger.myinfo("%s: %s" % (param, default))

                # ----------------------------------------------------------------------
                # Construct actual optimizer params
                # ----------------------------------------------------------------------
                kwargs = dict(kwargs_without_groups)
                if param_groups is None:
                    # ---------------------------------------------------------
                    # Add all trainable parameters if there is no param groups
                    # ---------------------------------------------------------
                    all_trainable_parameters = _generate_trainable_params(model_and_loss)
                    kwargs["params"] = all_trainable_parameters
                else:
                    # -------------------------------------------
                    # Add list of parameter groups instead
                    # -------------------------------------------
                    trainable_parameter_groups = []
                    dnames, dparams = _param_names_and_trainable_generator(model_and_loss)
                    dnames = set(dnames)
                    dparams = set(list(dparams))
                    with logger.LoggingBlock("parameter_groups:"):
                        for group in param_groups:
                            #  log group settings
                            group_match = group["params"]
                            group_args = {
                                key: value for key, value in group.items() if key != "params"
                            }

                            with logger.LoggingBlock("%s: %s" % (group_match, group_args)):
                                # retrieve parameters by matching name
                                gnames, gparams = _param_names_and_trainable_generator(
                                    model_and_loss, match=group_match)
                                # log all names affected
                                for n in sorted(gnames):
                                    logger.myinfo(n)
                                # set generator for group
                                group_args["params"] = gparams
                                # append parameter group
                                trainable_parameter_groups.append(group_args)
                                # update remaining trainable parameters
                                dnames -= set(gnames)
                                dparams -= set(list(gparams))

                        # append default parameter group
                        trainable_parameter_groups.append({"params": list(dparams)})
                        # and log its parameter names
                        with logger.LoggingBlock("default:"):
                            for dname in sorted(dnames):
                                logger.myinfo(dname)

                    # set params in optimizer kwargs
                    kwargs["params"] = trainable_parameter_groups

                # -------------------------------------------
                # Create optimizer instance
                # -------------------------------------------
                optimizer = tools.instance_from_kwargs(args.optimizer_class, kwargs)

    return optimizer


# -------------------------------------------------------------------------------------------------
# Configure learning rate scheduler
# -------------------------------------------------------------------------------------------------
def configure_lr_scheduler(args, optimizer):
    lr_scheduler = None

    with logger.LoggingBlock("Learning Rate Scheduler", emph=True):
        logger.myinfo("class: %s" % args.lr_scheduler)

        if args.lr_scheduler is not None:

            # ----------------------------------------------
            # Figure out lr_scheduler arguments
            # ----------------------------------------------
            kwargs = tools.kwargs_from_args(args, "lr_scheduler")
            
            # -------------------------------------------
            # Print arguments
            # -------------------------------------------
            for param, default in sorted(kwargs.items()):
                logger.myinfo("%s: %s" % (param, default))

            # -------------------------------------------
            # Add optimizer
            # -------------------------------------------
            kwargs["optimizer"] = optimizer

            # -------------------------------------------
            # Create lr_scheduler instance
            # -------------------------------------------
            lr_scheduler = tools.instance_from_kwargs(args.lr_scheduler_class, kwargs)

    return lr_scheduler
