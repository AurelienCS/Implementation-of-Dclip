import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from utils import load_clip_to_cpu

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
        
class FixedPromptEncoder(nn.Module):
    def __init__(self, classnames):  
        super().__init__()
        self.classnames = classnames
        self.texts_pos = [f"a photo of a {name}." for name in classnames]
        self.texts_neg = [f"not a photo of a {name}." for name in classnames]
        self.tokenized_prompts = clip.tokenize(self.texts_neg + self.texts_pos)

    def forward(self, clip_model): 
        device = next(clip_model.parameters()).device
        with torch.no_grad():
            prompts = clip_model.token_embedding(self.tokenized_prompts.to(device))
        return prompts, self.tokenized_prompts.to(device)
        
class DClip(nn.Module):
    def __init__(self, classnames, clip_model): #m
        super().__init__()
        self.prompt_encoder = FixedPromptEncoder(classnames)
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = 7
        self.dtype = clip_model.dtype
        self.classnames = classnames

        print(f"\n--- [INIT DCLIP] ---")
        print(f"Logit Scale: {self.logit_scale}")
        
        with torch.no_grad():
            device = next(self.clip_model.parameters()).device
            tokenized = self.prompt_encoder.tokenized_prompts.to(device)
            text_features = self.clip_model.encode_text(tokenized)
            # text_features = F.normalize(text_features.float(), dim=-1)
            self.register_buffer("fixed_text_features", text_features)
            print(f"Fixed Text Features shape: {self.fixed_text_features.shape} (Attendu: [2*N, 512])")

        # Vérification dimension d'entrée
        in_dim = 512
        if hasattr(clip_model.visual, 'proj') and clip_model.visual.proj is not None:
             in_dim = clip_model.visual.proj.shape[1]
        
        print(f"Detected ViT Output Dim: {in_dim}")

        self.image_projector = nn.Linear(in_dim, 256)
        self.text_projector = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 256)
        )
                
    def forward(self, image):
        B = image.shape[0]

        # Vit
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
        
        # Proj text et image
        image_features_proj = self.image_projector(image_features)
        image_features_proj = F.normalize(image_features_proj, dim=-1)
        projected_text_features = self.text_projector(self.fixed_text_features)
        projected_text_norm = F.normalize(projected_text_features, dim=-1) 
        
        image_features_proj = image_features_proj.transpose(1, 2) 
        score = F.conv1d(image_features_proj, projected_text_norm.unsqueeze(-1))
        weights = F.softmax(score * self.logit_scale, dim=-1) # logit scale 
        
        # Agrégation
        aggregated = (score * weights).sum(dim=-1) * 5.0 
        
        # 6. Reshape Final [B, 2, K]
        logits = aggregated.view(B, 2, -1) 

        return logits, projected_text_features

def dclip(classnames, clip_type="ViT-B/32", **kwargs): 
    clip_model = load_clip_to_cpu(clip_type) 
    clip_model.float()
    model = DClip(classnames, clip_model) 

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.image_projector.parameters():
        param.requires_grad_(True)
    for param in model.text_projector.parameters():
        param.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #m
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model