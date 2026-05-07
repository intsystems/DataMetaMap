# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import torch.utils.model_zoo as model_zoo

import torchvision.models.resnet as resnet
import torch

from data_meta_map.task2vec import ProbeNetwork

_MODELS = {}


def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn


class ResNet(resnet.ResNet, ProbeNetwork):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        # Saves the ordered list of layers. We need this to forward from an arbitrary intermediate layer.
        self.layers = [
            self.conv1, self.bn1, self.relu,
            self.maxpool, self.layer1, self.layer2,
            self.layer3, self.layer4, self.avgpool,
            lambda z: torch.flatten(z, 1), self.fc
        ]

    @property
    def classifier(self):
        return self.fc

    # @ProbeNetwork.classifier.setter
    # def classifier(self, val):
    #     self.fc = val

    # Modified forward method that allows to start feeding the cached activations from an intermediate
    # layer of the network
    def forward(self, x, start_from=0):
        """Replaces the default forward so that we can forward features starting from any intermediate layer."""
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x


@_add_model
def resnet18(pretrained=False, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model: ProbeNetwork = ResNet(
        resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet18-5c106cde.pth')
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


@_add_model
def resnet34(pretrained=False, num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


from data_meta_map.dataset2vec.model import Dataset2Vec as _Dataset2VecModel
from data_meta_map.dataset2vec.config import (
    Dataset2VecConfig as _Dataset2VecConfig,
    OptimizerConfig as _OptimizerConfig,
)


@_add_model
def dataset2vec(config=None, optimizer_config=None, **_):
    """Constructs a Dataset2Vec model for tabular dataset embedding.

    Args:
        config: Dataset2Vec architecture configuration. If None, uses defaults.
        optimizer_config: Optimizer configuration. If None, uses defaults.
        **_: Absorbs unused kwargs from get_model() (pretrained, num_classes).
    """
    cfg = config if config is not None else _Dataset2VecConfig()
    opt_cfg = optimizer_config if optimizer_config is not None else _OptimizerConfig()
    return _Dataset2VecModel(cfg, opt_cfg)


def get_model(model_name, pretrained=False, num_classes=1000):
    try:
        return _MODELS[model_name](pretrained=pretrained, num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Architecture {model_name} not implemented.")
