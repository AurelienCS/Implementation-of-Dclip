import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer 
from copy import deepcopy

_tokenizer = _Tokenizer()

def patch_vit_resolution(model, new_resolution=448):
    vision = model.visual
    if vision.input_resolution != new_resolution:
        patch_size = vision.conv1.stride[0] 
        old_gs = vision.input_resolution // patch_size
        new_gs = new_resolution // patch_size
        
        pos_embed = vision.positional_embedding
        cls_pos = pos_embed[:1, :]
        spatial_pos = pos_embed[1:, :]
        
        spatial_pos = spatial_pos.reshape(1, old_gs, old_gs, -1).permute(0, 3, 1, 2)
        spatial_pos = F.interpolate(spatial_pos, size=(new_gs, new_gs), mode='bicubic', align_corners=False)
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(new_gs*new_gs, -1)
        
        vision.positional_embedding = nn.Parameter(torch.cat([cls_pos, spatial_pos], dim=0))
        vision.input_resolution = new_resolution

    def spatial_forward(x):
        x = vision.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([vision.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + vision.positional_embedding.to(x.dtype)
        x = vision.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = vision.transformer(x)
        x = x.permute(1, 0, 2)
        x = x[:, 1:, :] 
        if vision.proj is not None:
            x = x @ vision.proj
        return x

    vision.forward = spatial_forward
    return model

def load_clip_to_cpu(backbone_name): 
    url = clip._MODELS[backbone_name] 
    import os
    download_root = os.path.expanduser("/tempory/cg/deepl/dclip/clip")
    os.makedirs(download_root, exist_ok=True)
    model_path = clip._download(url, root=download_root)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    model = patch_vit_resolution(model, new_resolution=448)
    return model