import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    Extremely effective for highly imbalanced 3D segmentation tasks (like aneurysms).
    """
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


class TverskyBCELoss(nn.Module):
    """
    Combined Tversky and Binary Cross Entropy Loss.
    Tversky index allows us to penalize False Negatives (FN) more heavily than 
    False Positives (FP) by setting beta > alpha, which directly drives up Sensitivity.
    """
    def __init__(self, alpha=0.3, beta=0.7, bce_weight=0.3, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()
        
        tversky_num = tp + self.smooth
        tversky_den = tp + self.alpha * fp + self.beta * fn + self.smooth
        
        tversky_index = tversky_num / tversky_den
        tversky_loss = 1.0 - tversky_index
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * tversky_loss


def compute_metrics(logits, targets, threshold=0.5, smooth=1e-5):
    """
    Computes standard segmentation metrics: Dice, Sensitivity, Specificity.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    
    dice = (2.0 * tp + smooth) / (preds_flat.sum().item() + targets_flat.sum().item() + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return {
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


class AneurysmTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3, device='cuda', checkpoint_path='best_model.pth', use_amp=True, threshold=0.5):
        """
        Trainer class for Aneurysm Detection.
        Supports Automatic Mixed Precision (AMP) for fast training on high-end GPUs like A100.
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        
        # Default to Tversky Loss with FN penalization (alpha=0.3, beta=0.7)
        self.criterion = TverskyBCELoss(alpha=0.3, beta=0.7, bce_weight=0.3)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        
        # Enable AMP if on CUDA
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) enabled for GPU acceleration.")
        
        self.best_val_dice = 0.0

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        metrics_sum = {'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0}
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, vesselness = self.model(images)
                loss = self.criterion(logits, targets)
            
            # Backward pass with scaled gradients
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Compute training metrics using custom threshold
            batch_metrics = compute_metrics(logits, targets, threshold=self.threshold)
            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]
                
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        
        return avg_loss, avg_metrics

    def validate(self):
        self.model.eval()
        epoch_loss = 0.0
        metrics_sum = {'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0}
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits, vesselness = self.model(images)
                    loss = self.criterion(logits, targets)
                
                epoch_loss += loss.item()
                
                # Compute validation metrics using custom threshold
                batch_metrics = compute_metrics(logits, targets, threshold=self.threshold)
                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics[k]
                    
        num_batches = len(self.val_loader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        
        return avg_loss, avg_metrics

    def fit(self, epochs=20):
        history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_sensitivity': [], 'val_sensitivity': [],
            'train_specificity': [], 'val_specificity': []
        }
        
        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            self.scheduler.step(val_metrics['dice'])
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dice'].append(train_metrics['dice'])
            history['val_dice'].append(val_metrics['dice'])
            history['train_sensitivity'].append(train_metrics['sensitivity'])
            history['val_sensitivity'].append(val_metrics['sensitivity'])
            history['train_specificity'].append(train_metrics['specificity'])
            history['val_specificity'].append(val_metrics['specificity'])
            
            print(f"Epoch {epoch:02d}/{epochs:02d}:")
            print(f"  [Train] Loss: {train_loss:.4f} | Dice: {train_metrics['dice']:.4f} | Sens: {train_metrics['sensitivity']:.4f} | Spec: {train_metrics['specificity']:.4f}")
            print(f"  [Val  ] Loss: {val_loss:.4f} | Dice: {val_metrics['dice']:.4f} | Sens: {val_metrics['sensitivity']:.4f} | Spec: {val_metrics['specificity']:.4f}")
            
            # Save best checkpoint
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': self.best_val_dice,
                }, self.checkpoint_path)
                print(f"  => Saved new best checkpoint with Validation Dice: {self.best_val_dice:.4f}")
                
        return history
