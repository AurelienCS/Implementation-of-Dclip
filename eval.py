import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def evaluate_dclip_model(model, dataloader, class_names, device="cuda"):
    model.eval()
    all_targets = []
    all_scores  = []
    all_probs   = []

    softmax_dual = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)

            probs = softmax_dual(logits) 
            probs_pos = probs[:, 1, :]

            all_scores.append(logits[:, 1, :].cpu())
            all_probs.append(probs_pos.cpu())
            all_targets.append(labels.cpu())

    all_scores  = torch.cat(all_scores).numpy()
    all_probs   = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    APs = [average_precision_score(all_targets[:, i], all_scores[:, i]) 
           for i in range(len(class_names)) if np.sum(all_targets[:, i]) > 0]
    mAP = np.mean(APs)

    best_f1 = 0
    best_threshold = 0.5
    for t in np.linspace(0.05, 0.95, 37):
        preds = (all_probs >= t).astype(float)
        tp = (preds * all_targets).sum()
        fp = (preds * (1 - all_targets)).sum()
        fn = ((1 - preds) * all_targets).sum()
        
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    final_preds = (all_probs >= best_threshold).astype(float)
    total_tp = (final_preds * all_targets).sum()
    total_fp = (final_preds * (1 - all_targets)).sum()
    CP = total_tp / (total_tp + total_fp + 1e-8)

    print(f"  mAP    : {mAP:.4f}")
    # print(f"  Best T : {best_threshold:.2f} (F1: {best_f1:.4f})")
    print(f"  CP     : {CP:.4f}")

    return mAP, CP

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits [B, 2, N_classes]
        y: targets [B, N_classes]
        """
        # Calculating Probabilities
        x_softmax = self.softmax(x)
        
        xs_pos = x_softmax[:, 1, :]
        xs_neg = x_softmax[:, 0, :]
        
        y = y.reshape(-1)
        xs_pos = xs_pos.reshape(-1)
        xs_neg = xs_neg.reshape(-1)

        xs_pos = xs_pos[y!=-1]
        xs_neg = xs_neg[y!=-1]
        y = y[y!=-1]

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        final_loss = -loss.sum()
        return final_loss

class AsymmetricLossOptimized(nn.Module):
    """
    ASL (sigmoid-based) for multi-label tasks
    - uses asymmetric focusing on positives/negatives
    - works better for multi-label CLIP/DCLIP setups
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits, targets):
        """
        logits: [B, N_classes] raw logits
        targets: [B, N_classes] binary multi-label targets (0 or 1)
        """
        logits = logits[:, 1, :]
        # Sigmoid probabilities
        x_sigmoid = self.sigmoid(logits)

        # Clip negative probs
        if self.clip is not None and self.clip > 0:
            x_sigmoid = torch.clamp(x_sigmoid, min=self.clip, max=1.0)

        # Basic CE
        loss_pos = targets * torch.log(x_sigmoid.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - x_sigmoid).clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt = x_sigmoid * targets + (1 - x_sigmoid) * (1 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

        
class MFILoss(nn.Module):
    def __init__(self, lambda_coeff=0.2):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, x):
        if x.shape[0] < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # pour offdiag loss (collapse prevention)
        x = F.normalize(x, dim=-1)
        S = x @ x.t()
        n = x.size(0)

        # Collapse Prevention
        diag_loss = torch.sum((torch.diagonal(S) - 1) ** 2)

        # MFI Reduction
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        off_diag_loss = torch.sum(S[mask] ** 2)

        total_loss = diag_loss + self.lambda_coeff * off_diag_loss
        return total_loss

class MFILossOpti(nn.Module): # on ne calcule pas la diag car on a déjà normalisé
    def __init__(self, lambda_coeff=0.2):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, x):
        """
        x: projected_text_features [2*K, d]
        """
        if x.shape[0] < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        x = F.normalize(x, dim=-1)
        # Similarité
        S = x @ x.t()
        n = x.size(0)

        # hors-diag
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        off_diag_sims = S[mask]
        loss_mfi = torch.sum(off_diag_sims ** 2)

        return self.lambda_coeff * loss_mfi

def save_checkpoint(model, optimizer, epoch, mAP, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mAP': mAP,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('mAP', 0.0)