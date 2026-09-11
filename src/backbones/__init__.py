import sys
sys.path.append('./src/backbones')

import torch
import torch.nn as nn

from .efficientnet import build_efficient
        
def get_backbone_feature_shape(model_type, img_size=256):
    """Native feature-grid shape produced by BackboneWrapper: all outblocks
    are resampled to a common stride-16 grid, so grid_hw = img_size // 16
    regardless of img_size (img_size defaults to 256 to preserve legacy
    callers/checkpoints that never passed img_size explicitly)."""
    if model_type == "efficientnet-b4":
        grid = img_size // 16
        return (272, grid, grid)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_efficientnet(model_name, **kwargs):
    return build_efficient(model_name, **kwargs)

def get_backbone(**kwargs):
    model_name = kwargs['model_type']
    if 'efficientnet' in model_name:
        net =  get_efficientnet(model_name, **kwargs)
        return BackboneWrapper(net, scale_factors=[0.125, 0.25, 0.5, 1.0])
    else:
        raise ValueError(f"Invalid backbone model: {model_name}")

class BackboneWrapper(nn.Module):
    def __init__(self, backbone, scale_factors=None, target_size=None):
        super(BackboneWrapper, self).__init__()
        self.backbone = backbone
        self.scale_factors = scale_factors
        self.target_size = target_size
        assert scale_factors is not None or target_size is not None, "Either scale_factors or target_size must be provided"
        
        self.downsamples = nn.ModuleList()
        if scale_factors is not None:
            for scale_factor in scale_factors:
                self.downsamples.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
        elif target_size is not None:
            for _ in range(len(self.backbone.outblocks)):
                self.downsamples.append(nn.Upsample(size=target_size, mode='bilinear'))
        
    def forward(self, x):
    
        out = self.backbone(x)
        if isinstance(out, dict):
            y = out["features"]
        else:
            y = out
        concat_y = torch.cat([downsample(y[i]) for i, downsample in enumerate(self.downsamples)], dim=1)
        return concat_y, y