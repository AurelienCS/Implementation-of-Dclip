import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer 
from copy import deepcopy

_tokenizer = _Tokenizer()

class VOCHFDataset(Dataset):
    def __init__(self, hf_ds, class_names, transform=None):
        self.ds, self.class_names, self.transform = hf_ds, class_names, transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        
        img = item["image"].convert("RGB")
        if self.transform: 
            img = self.transform(img)

        vec = torch.zeros(len(self.class_names))
        
        label_list = item.get("classes", [])
        
        for i in label_list:
            if i < len(self.class_names):
                vec[i] = 1.0
                
        return img, vec

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



def visualize_comparison_clip_dclip(clip_model, preprocess, dclip_model, image, class_names, device="cuda"):
    clip_model.eval()
    dclip_model.eval()
    
    if isinstance(class_names, str):
        class_names = [class_names]
    else:
        class_names = list(class_names) 
        
    if len(class_names) == 0:
        print("Aucune classe valide trouvée.")
        return
        
    title_str = ", ".join(class_names)
    
    # Image de base pour l'affichage (Standardisée à 448x448)
    img_display = np.array(image.resize((448, 448)))

    with torch.no_grad():
        # Extraction CLIP
        text_prompts = [f"a photo of a {c}" for c in class_names]
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_features_clip = F.normalize(clip_model.encode_text(text_tokens), dim=-1)

        img_tensor_clip = preprocess(image).unsqueeze(0).to(device)
        visual = clip_model.visual
        x_clip = img_tensor_clip.type(clip_model.dtype)
        
        # Passage dans le ViT de CLIP
        x_clip = visual.conv1(x_clip) 
        grid_size_clip = x_clip.shape[-1]
        x_clip = x_clip.reshape(x_clip.shape[0], x_clip.shape[1], -1).permute(0, 2, 1)
        x_clip = torch.cat([visual.class_embedding.to(x_clip.dtype) + torch.zeros(x_clip.shape[0], 1, x_clip.shape[-1], dtype=x_clip.dtype, device=x_clip.device), x_clip], dim=1)
        x_clip = x_clip + visual.positional_embedding.to(x_clip.dtype)
        x_clip = visual.ln_pre(x_clip)
        x_clip = x_clip.permute(1, 0, 2)
        x_clip = visual.transformer(x_clip)
        x_clip = x_clip.permute(1, 0, 2)
        
        spatial_features_clip = visual.ln_post(x_clip[:, 1:, :])
        if visual.proj is not None:
            spatial_features_clip = spatial_features_clip @ visual.proj
        img_norm_clip = F.normalize(spatial_features_clip, dim=-1)

        # Fusion des heatmaps CLIP
        maps_clip = []
        for i in range(len(class_names)):
            target = text_features_clip[i].unsqueeze(0)
            attn = (img_norm_clip[0] * target).sum(dim=-1)
            maps_clip.append(attn)
            
        combined_clip = torch.stack(maps_clip, dim=0).max(dim=0)[0]
        attn_grid_clip = combined_clip.view(1, 1, grid_size_clip, grid_size_clip)
        
        # Interpolation à 448x448 pour l'affichage
        mask_clip = F.interpolate(attn_grid_clip, size=(448, 448), mode='bilinear', align_corners=False)
        mask_clip = mask_clip.squeeze().cpu().numpy()
        mask_clip = (mask_clip - mask_clip.min()) / (mask_clip.max() - mask_clip.min() + 1e-8)

        # Extraction DCLIP
        transform_dclip = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        img_tensor_dclip = transform_dclip(image).unsqueeze(0).to(device)
        
        raw_img_feats = dclip_model.image_encoder(img_tensor_dclip.type(dclip_model.dtype))
        img_proj = dclip_model.image_projector(raw_img_feats)
        img_norm_dclip = F.normalize(img_proj, dim=-1)
        
        text_proj = dclip_model.text_projector(dclip_model.fixed_text_features)
        text_norm_dclip = F.normalize(text_proj, dim=-1)

        valid_classes = [c for c in class_names if c in dclip_model.classnames]
        K = len(dclip_model.classnames)
        
        maps_dclip = []
        for class_name in valid_classes:
            cls_idx = list(dclip_model.classnames).index(class_name)
            pos_idx = K + cls_idx 
            
            target = text_norm_dclip[pos_idx].unsqueeze(0)
            attn = (img_norm_dclip[0] * target).sum(dim=-1)
            maps_dclip.append(attn)

        if maps_dclip:
            combined_dclip = torch.stack(maps_dclip, dim=0).max(dim=0)[0]
            grid_size_dclip = int(combined_dclip.shape[0]**0.5)
            attn_grid_dclip = combined_dclip.view(1, 1, grid_size_dclip, grid_size_dclip)
            
            mask_dclip = F.interpolate(attn_grid_dclip, size=(448, 448), mode='bilinear', align_corners=False)
            mask_dclip = mask_dclip.squeeze().cpu().numpy()
            mask_dclip = (mask_dclip - mask_dclip.min()) / (mask_dclip.max() - mask_dclip.min() + 1e-8)
        else:
            mask_dclip = np.zeros((448, 448))


    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # : Image Originale
    ax[0].imshow(img_display)
    ax[0].axis("off")
    
    # Overlay CLIP Standard
    ax[1].imshow(img_display)
    ax[1].imshow(mask_clip, cmap="jet", alpha=0.55)
    ax[1].set_title("Overlay CLIP Standard", fontsize=14)
    ax[1].axis("off")
    
    # Overlay DCLIP
    ax[2].imshow(img_display)
    ax[2].imshow(mask_dclip, cmap="jet", alpha=0.55)
    ax[2].set_title("Overlay DCLIP", fontsize=14)
    ax[2].axis("off")
    
    fig.suptitle(f"Comparaison image originale et heatmap avec CLIP/DCLIP, classes : {title_str}", 
                 fontsize=18, fontweight='bold', y=1)
    
    plt.tight_layout()
    plt.show()



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
