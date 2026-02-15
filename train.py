import torch
import torch.nn as nn
from tqdm import tqdm
from eval import evaluate_dclip_model, AsymmetricLossOptimized, MFILossOpti, save_checkpoint
from dclip import dclip

def train_dclip(
    model, 
    train_loader, 
    test_loader, 
    class_names, 
    num_epochs=50, 
    lr=0.002, 
    alpha=7e-5, 
    eval_period=5, # 0: aucune, 1: chaque epoch, etc.
    device="cuda", 
    verbose=True
):
    """
    Boucle d'entraînement pour D-CLIP avec période d'évaluation paramétrable.
    """
    # ASL 
    asl_criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=1, clip=0.05).to(device)
    # MFI 
    mfi_criterion = MFILossOpti(lambda_coeff=0.2).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], # juste unfrozen param
        lr=lr,
        momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_map = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_asl, epoch_mfi, epoch_loss = 0.0, 0.0, 0.0 #m
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=not verbose, ncols=100)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
        
            # Forward
            logits, class_text_features = model(images)
        
            # loss
            loss_asl = asl_criterion(logits, labels)
            loss_mfi = mfi_criterion(class_text_features)
            loss = loss_asl + alpha * loss_mfi
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_asl += loss_asl.item() 
            epoch_mfi += (alpha * loss_mfi.item()) 
            epoch_loss += loss.item() 
            
            if verbose:
                pbar.set_postfix({
                    "ASL": f"{loss_asl.item():.4f}", 
                    "α*MFI": f"{(alpha*loss_mfi.item()):.5f}", 
                    "Loss": f"{loss.item():.4f}"
                })

        if verbose: 
            n = len(train_loader) 
            print(f"Avg Epoch {epoch}: ASL={epoch_asl/n:.4f} | α*MFI={epoch_mfi/n:.5f} | Total={epoch_loss/n:.4f}") #m

        scheduler.step()
        
        # eval
        should_eval = (eval_period > 0) and (epoch % eval_period == 0 or epoch == 1 or epoch == num_epochs)
        
        if should_eval:
            if verbose: print(f"\n--- Evaluation (Epoch {epoch}) ---")
            current_map, cp = evaluate_dclip_model(model, test_loader, class_names, device=device)
            
            if current_map > best_map:
                best_map = current_map
                save_checkpoint(model, optimizer, epoch, best_map, "best_model.pth")
            
            if verbose: 
                print(f"mAP: {current_map:.4f} | CP: {cp:.4f} (Best: {best_map:.4f})")

    if verbose: print(f"\nTraining Finished! Best mAP: {best_map:.4f}")
    return model, best_map