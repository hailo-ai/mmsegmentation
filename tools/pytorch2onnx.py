# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.nn as nn

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

import onnx
from onnxsim import simplify
from mmseg.models.utils import resize

import torch.nn.functional as F

from collections import OrderedDict
import warnings

def load_pretrained_weights_soft(model, checkpoint):

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
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'
        )
    else:
        print('Successfully loaded pretrained weights')
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


def dummy_prune_ckpt(ckpt, prune_ratio=0.5):
    pass
    for k, v in ckpt['state_dict'].items():
        if k.startswith('backbone.') and k.endswith('.rbr_dense.conv.weight'):
            # Sparsify layer:
            v = dummy_prune_layer(v, prune_ratio)
    calc_sparsity(ckpt['state_dict'])
    return ckpt

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

def calc_sparsity(model_dict):
    weights_layers_num, total_weights, total_zeros = 0, 0, 0
    for k, v in model_dict.items():
        if k.startswith('backbone.') and k.endswith('weight'):
            weights_layers_num += 1
            total_weights += v.numel()
            total_zeros += (v.numel() - v.count_nonzero())
            zeros_ratio = (v.numel() - v.count_nonzero()) / v.numel() * 100.0
            print(f"[{weights_layers_num:>2}] {k:<51}:: {v.numel() - v.count_nonzero():<5} / {v.numel():<7} ({zeros_ratio:<4.1f}%) are zeros")
    print(f"Model has {weights_layers_num} weight layers")
    print(f"Overall Sparsity is roughly: {100 * total_zeros / total_weights:.1f}%")


def parse_args():
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--no_simplify', action='store_false')
    parser.add_argument('--no_postprocess', action='store_true', default=False)
    parser.add_argument('--shape', nargs=2, type=int, default=[1024, 1920])
    parser.add_argument('--out_name', default='fcn.onnx', type=str, help="Name for the onnx output")
    parser.add_argument('--soft_weights_loading',action='store_true', default=False)
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
            self.post_proc_flag = not(args.no_postprocess)
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

    # start training
    model = runner.model
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if args.soft_weights_loading:
            ckpt = dummy_prune_ckpt(ckpt, 0.6)
            load_pretrained_weights_soft(model, ckpt)
        else:
            if 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
    
    print("Switching to deployment model")
    # if repvgg style -> deploy
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    calc_sparsity(model.state_dict())

    # to onnx
    model.eval()
    model_with_postprocess = ModelWithPostProc(model, args)
    model_with_postprocess.eval()
    imgs = torch.zeros(1,3, args.shape[0], args.shape[1], dtype=torch.float32).to(device)
    outputs = model_with_postprocess(imgs)

    torch.onnx.export(model_with_postprocess, imgs, args.out_name, input_names=['test_input'], output_names=['output'], training=torch.onnx.TrainingMode.PRESERVE, opset_version=13)
    print('model saved at: ', args.out_name)

    # if also simplify
    if args.no_simplify:
        model_onnx = onnx.load(args.out_name)
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, args.out_name[0:-5] + '_simplify.onnx')
        print('model simplified saved at: ', args.out_name[0:-5] + '_simplify.onnx')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Example: CUDA_VISIBLE_DEVICES=0 python tools/pytorch2onnx.py configs/fcn/fcn8_r18_hailo.py --checkpoint work_dirs/fcn8_r18_hailo_iterbased/epoch_1.pth --out_name my_fcn_model.onnx --shape 600 800')
    main()
