import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import time
import numpy as np

from .fbnet_blocks import *
from .fbnet_arch import predefine_archs

import logging
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, kaiming_init
from collections import OrderedDict
import logging

# from .utils import load_checkpoint

# from ..registry import BACKBONES

from ..builder import BACKBONES
# @BACKBONES.register_module

try:
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
except:
    pass

def load_checkpoint(model,
                    filename,
                    strict=False,
                    logger=None):


    checkpoint = torch.load(filename)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    Args:
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    state_dict_modify = state_dict.copy()
    for name, param in state_dict.items():
        ''' for mobilenet v2
        if 'features' in name:
            name = name.replace('features.','features')
        '''
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if 'conv2' in name and 'layer4.0.conv2_d2.weight' in own_state.keys():
            d1 = name.replace('conv2', 'conv2_d1')
            d1_c = own_state[d1].size(0)
            own_state[d1].copy_(param[:d1_c,:,:,:])
            state_dict_modify[d1] = param[:d1_c,:,:,:]

            d2 = name.replace('conv2', 'conv2_d2')
            d2_c = own_state[d2].size(0)
            own_state[d2].copy_(param[d1_c:d1_c+d2_c,:,:,:])
            state_dict_modify[d2] = param[d1_c:d1_c+d2_c,:,:,:]

            d3 = name.replace('conv2', 'conv2_d3')
            own_state[d3].copy_(param[d1_c+d2_c:,:,:,:])
            state_dict_modify[d3] = param[d1_c+d2_c:,:,:,:]
        else:
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} and '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    '''
    if 'layer4.0.conv2_d2.weight' in own_state.keys():
        missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    else:
        # for mobilenetv2
        own_state_set = []
        for name in set(own_state.keys()):
            own_state_set.append(name.replace('features','features.'))
        missing_keys = set(own_state_set) - set(state_dict.keys())
    '''
    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            print(err_msg)

@BACKBONES.register_module()
class FBNet(nn.Module):
    def __init__(self, arch='my_search_result_2', out_indices=(5, 9, 17, 22), frozen_stages=-1):
        super(FBNet, self).__init__()
        print('Model is {}.'.format(arch))
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.arch = arch
        self.input_size = 800

        self.build_backbone(self.arch, self.input_size)

    def build_backbone(self, arch, input_size):
        genotypes = predefine_archs[arch]['genotypes'] 
        strides = predefine_archs[arch]['strides'] 
        out_channels = predefine_archs[arch]['out_channels']
        
        self.layers = nn.ModuleList()
        self.layers.append(ConvBNReLU(input_size, in_channels=3, out_channels=out_channels[0], kernel_size=3, stride=strides[0], padding=1, 
                      bias=True, relu_type='relu', bn_type='bn'))
        input_size = input_size // strides[0]

        _in_channels = out_channels[0]
        for genotype, stride, _out_channels in zip(genotypes[1:], strides[1:], out_channels[1:]):
            if genotype.endswith('sb'):
                self.layers.append(SUPER_PRIMITIVES[genotype](input_size, _in_channels, _out_channels, stride))
            else:
                self.layers.append(PRIMITIVES[genotype](input_size, _in_channels, _out_channels, stride))
            input_size = input_size // stride
            _in_channels = _out_channels

        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                m._specify_ddp_gpu_num(1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, alphas=None):
        outs = []
        cnt = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs