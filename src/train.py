# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================
"""Training code"""
import argparse
import random

import numpy as np
import mindspore
from mindspore import context
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

from datasets import coco, cocopanoptic
from models import detr
from models import matcher
from models.segmentation import DETRsegm
from models.loss import CustomWithLossCell


def parse_args():
    """Training parameters"""
    parser = argparse.ArgumentParser(description='train DTER')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--lr', default=0.00001, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-6, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--coco_dir', default='./coco', type=str)
    parser.add_argument('--pano_dir', default='./coco_panoptic', type=str)
    parser.add_argument('--resnet', default='resnet50', choices=['resnet50', 'resnet101'], type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--is_segmentation', action='store_true')
    parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
    return parser.parse_args()


def train(args):
    """
    Training code

    Args:
        args: Training parameters.
    """
    print('args:', args)
    mindspore.set_seed(42)
    # np.random.seed(42)
    # random.seed(42)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device)
    is_segmentation = args.is_segmentation
    if is_segmentation:
        dataset = cocopanoptic.build(img_set='train', batch=args.batch, shuffle=True,
                                     coco_dir=args.coco_dir, pano_dir=args.pano_dir)
    else:
        dataset = coco.build(img_set='train', batch=args.batch, shuffle=True,
                                  coco_dir=args.coco_dir)
    print('Build detr network......')
    if is_segmentation:
        net = detr.bulid_detr(resnet=args.resnet, return_interm_layers=is_segmentation,
                              num_classes=250, is_dilation=args.dilation)
        net = DETRsegm(net, freeze_detr=False)
    else:
        net = detr.bulid_detr(resnet=args.resnet, return_interm_layers=False,
                              num_classes=91, is_dilation=args.dilation)
    params = []
    for p in net.trainable_params():
        if "query_embed" not in p.name:
            params.append(p)
    optim = nn.AdamWeightDecay(params=params, learning_rate=args.lr, weight_decay=args.weight_decay)
    criterion = matcher.build_criterion(is_segmentation=is_segmentation)
    net_with_loss = CustomWithLossCell(net, criterion)
    model = mindspore.Model(network=net_with_loss, optimizer=optim)
    config_ck = CheckpointConfig(save_checkpoint_steps=2, keep_checkpoint_max=2)
    ckpoint = ModelCheckpoint(prefix="detr", directory=args.checkpoint_path,
                              config=config_ck)
    print('Start to train......')
    model.train(args.epoch, dataset, callbacks=[ckpoint, LossMonitor()], dataset_sink_mode=False)


if __name__ == '__main__':
    train(parse_args())
