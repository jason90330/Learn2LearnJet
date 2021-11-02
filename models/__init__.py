from __future__ import absolute_import
from .EfficientMetaNet import *

__factory = {
    'FeatExtractor': FeatExtractor,
    'FeatEmbedder': FeatEmbedder,
    'DepthEstmator': DepthEstmator,
}


def names():
    return sorted(__factory.keys())


def create(name, pretrain = True, in_ch=6, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name == "Eff_FeatExtractor":
        if pretrain:
            return FeatExtractor.from_pretrained('efficientnet-b0',in_channels=in_ch)
        else:
            return FeatExtractor.from_name('efficientnet-b0',in_channels=in_ch)

    elif name == "Eff_FeatEmbedder":
        if pretrain:
            return FeatEmbedder.from_pretrained('efficientnet-b0')#normal
        else:
            # return FeatEmbedder.from_name('efficientnet-b0',num_classes=1)#CelebA_efficient_b0_wo_arf_equal
            return FeatEmbedder.from_name('efficientnet-b0',num_classes=2)#wo arcface

    elif name == "Eff_DepthEstmator":
        return DepthEstmator()

    elif name not in __factory:
        raise KeyError("Unknown model:", name)

    else:
        return __factory[name](*args, **kwargs)
