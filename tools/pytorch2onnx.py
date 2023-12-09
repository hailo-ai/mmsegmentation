# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
import torch.nn as nn

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.registry import RUNNERS

import onnx
from onnxsim import simplify

from collections import OrderedDict

def load_pretrained_weights_soft(model, checkpoint, logger):

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        logger.warning(
            'The pretrained weights cannot be loaded, '
            'please check the key names manually '
        )
    else:
        logger.info('Successfully loaded pretrained weights')
        if len(discarded_layers) > 0:
            logger.warning(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


def dummy_prune_ckpt(ckpt, prune_ratio=0.5, random_prune=False):
    for k, v in ckpt['state_dict'].items():
        if k.startswith('backbone.') and k.endswith('.rbr_dense.conv.weight'):
            if random_prune:  # Sparsify layer randomly:
                v = random_prune_layer(v, prune_ratio)
            else:  # Sparsify layer according to magnitude:
                v = dummy_prune_layer(v, prune_ratio)
    calc_sparsity(ckpt['state_dict'])
    return ckpt


def random_prune_layer(layer, prune_ratio=0.5):
    """
    Randomly prune (set to zero) a fraction of elements in a PyTorch tensor.

    Args:
        layer (torch.Tensor): Input tensor of shape [B, C, H, W].
        prune_ratio (float): Fraction of elements to set to zero.

    Returns:
        torch.Tensor: Pruned tensor with the same shape as the input.
    """
    # Determine the number of elements to prune
    num_elements = layer.numel()
    num_prune = int(prune_ratio * num_elements)

    # Create a mask with zeros and ones to select the elements to prune
    mask = torch.ones(num_elements)
    mask[:num_prune] = 0
    mask = mask[torch.randperm(num_elements)]  # Shuffle the mask randomly
    mask = mask.view(layer.shape)

    # Apply the mask to the input tensor to prune it
    layer *= mask
    return layer


def dummy_prune_layer(layer, prune_ratio=0.5):
    # Flatten the tensor
    flattened_layer = layer.flatten()
    # Get the absolute values
    abs_values = torch.abs(flattened_layer)
    # Get indices sorted by absolute values
    sorted_indices = torch.argsort(abs_values)
    # Determine the threshold index
    threshold_index = int(prune_ratio * len(sorted_indices))
    # Set values below the threshold to zero
    flattened_layer[sorted_indices[:threshold_index]] = 0
    # Reshape the tensor back to its original shape
    pruned_tensor = flattened_layer.reshape(layer.shape)

    return pruned_tensor

def calc_sparsity(model_dict, logger):
    weights_layers_num, total_weights, total_zeros = 0, 0, 0
    for k, v in model_dict.items():
        if k.startswith('backbone.') and k.endswith('weight'):
            weights_layers_num += 1
            total_weights += v.numel()
            total_zeros += (v.numel() - v.count_nonzero())
            zeros_ratio = (v.numel() - v.count_nonzero()) / v.numel() * 100.0
            logger.info(f"[{weights_layers_num:>2}] {k:<51}:: {v.numel() - v.count_nonzero():<5} / {v.numel():<7} ({zeros_ratio:<4.1f}%) are zeros")
    logger.info(f"Model has {weights_layers_num} weight layers")
    logger.info(f"Overall Sparsity is roughly: {100 * total_zeros / total_weights:.1f}%")


def parse_args():
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--no_simplify', action='store_false')
    parser.add_argument('--postprocess', action='store_true', default=False)
    parser.add_argument('--shape', nargs=2, type=int, default=[1024, 1920])
    parser.add_argument('-o', '--opset', type=int, default=13)
    parser.add_argument('--out_name', default='fcn.onnx', type=str, help="Name for the onnx output")
    parser.add_argument('--soft_weights_loading', action='store_true', default=False)
    parser.add_argument('--dummy_prune_ratio', type=float, default=0.0, help="Applies dummy pruning with ratio")
    parser.add_argument('--random_prune', action='store_true', default=False, help="Set method to prune as random (default: Minimum absolute value)")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


class ModelWithPostProc(torch.nn.Module):
        def __init__(self, model, args):
            super(ModelWithPostProc, self).__init__()
            self.model = model
            self.post_proc_flag = args.postprocess
            self.shape = args.shape
            self.bilinear_resize = nn.Upsample(size=self.shape, mode='bilinear', align_corners=True)

        def forward(self, x):
            x = self.model(x)
            if self.post_proc_flag:
                x = self.bilinear_resize(x)
                if x.shape[1] > 1:
                    x = x.argmax(dim=1, keepdim=True)
            return x


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    model = runner.model
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if args.soft_weights_loading:
            if args.dummy_prune_ratio > 0.0:
                ckpt = dummy_prune_ckpt(ckpt, args.dummy_prune_ratio, args.random_prune)
            load_pretrained_weights_soft(model, ckpt, runner.logger)
        else:
            if 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
    
    runner.logger.info("Switching to deployment model")
    # if repvgg style -> deploy
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    calc_sparsity(model.state_dict(), runner.logger)

    # to onnx
    model.eval()
    if args.postprocess:
        runner.logger.info("Adding Postprocess (Resize+ArgMax) to the model")
    model_with_postprocess = ModelWithPostProc(model, args)
    model_with_postprocess.eval()

    imgs = torch.zeros(1,3, args.shape[0], args.shape[1], dtype=torch.float32).to(device)
    outputs = model_with_postprocess(imgs)

    torch.onnx.export(model_with_postprocess,
                      imgs, args.out_name,
                      input_names=['test_input'],
                      output_names=['output'],
                      training=torch.onnx.TrainingMode.PRESERVE,
                      opset_version=args.opset)

    # if also simplify
    if args.no_simplify:
        model_onnx = onnx.load(args.out_name)
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, args.out_name)
        runner.logger.info(f"Simplified model saved at: {args.out_name}")
    else:
        runner.logger.info(f"Model saved at: {args.out_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        epilog='Example: CUDA_VISIBLE_DEVICES=0 python tools/pytorch2onnx.py configs/fcn/fcn_hailo_10classes.py --checkpoint work_dirs/fcn_hailo/iter_173760.pth --shape 736 960 --postprocess --soft_weights_loading --out_name fcn_hailo.onnx')
    main()
