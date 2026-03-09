import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NanoTacotron
from data_loader import VoiceCloningDataset, collate_fn, chars
import os
import sys

# Check for Mixed Precision support
try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

# Hyperparameters
BATCH_SIZE = 4  # Reduced for RTX 3050 (4GB VRAM) to prevent OOM
ACCUM_STEPS = 8 # Gradient Accumulation: Effective Batch Size = 32 (4 * 8)
LEARNING_RATE = 1e-4
EPOCHS = 100
CHECKPOINT_DIR = "checkpoints"
DATASET_PATH = "dataset" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("-" * 50)
print(f"Training Config:")
print(f"Device: {device}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Accumulation Steps: {ACCUM_STEPS}")
print(f"Effective Batch Size: {BATCH_SIZE * ACCUM_STEPS}")
print(f"Mixed Precision (AMP): {'Enabled' if HAS_AMP else 'Disabled'}")
print("-" * 50)

def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    """Saves the training state comprehensively."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss
    }, path)
    print(f"Saved Checkpoint: {path}")

def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Loads a checkpoint and restores training state."""
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    # Load Model
    # stored_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Legacy support for old checkpoints (raw state dict)
        model.load_state_dict(checkpoint, strict=False)

    # Load Optimizer
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Load Scaler
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    # Get Epoch
    start_epoch = checkpoint.get('epoch', 0)
    # If loading from a completed epoch, start safely at next
    if 'epoch' in checkpoint:
        start_epoch += 1 
        
    return start_epoch

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Prepare Data
    if not os.path.exists(DATASET_PATH):
        print(f"WARNING: Dataset path '{DATASET_PATH}' not found.")
        print("Please create a folder named 'dataset' and put your speaker folders inside.")
        return
        
    dataset = VoiceCloningDataset(DATASET_PATH)
    if len(dataset) == 0:
        print("ERROR: No .wav files found in the dataset folder!")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    # 2. Init Model
    model = NanoTacotron(num_chars=len(chars)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=HAS_AMP)
    
    criterion_mel = nn.MSELoss()
    
    # 3. Resume Logic
    start_epoch = 0
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    if checkpoints:
        # Sort by numerical epoch index if possible, else modification time
        try:
            # Assumes format "checkpoint_epoch_X.pth"
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        except ValueError:
            # Fallback to last modified
            latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
            
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
        try:
            start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scaler)
            print(f"Resumed training from Epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            print("Starting fresh.")

    # 4. Training Loop
    print("Starting Training...")
    model.train()
    optimizer.zero_grad() # Initialize zero grad
    
    try:
        for epoch in range(start_epoch, EPOCHS):
            total_loss = 0
            
            for i, batch in enumerate(dataloader):
                # Unpack batch
                text, mel_target, mel_ref = batch
                
                text = text.to(device)
                mel_target = mel_target.to(device)
                mel_ref = mel_ref.to(device)
                
                # Forward Pass with Mixed Precision
                with autocast(enabled=HAS_AMP):
                    # We pass mel_target for teacher forcing, but we only calculate loss on output
                    # Returns: mel_out (pre-postnet), mel_final (post-postnet), gate_out, alignment
                    mel_out, mel_final, gate_out, _ = model(text, mel_target, ref_mel=mel_ref)
                    
                    # Loss Calculation
                    # 1. Mel Loss (Before PostNet)
                    loss_mel_pre = criterion_mel(mel_out, mel_target)
                    # 2. Mel Loss (After PostNet)
                    loss_mel_post = criterion_mel(mel_final, mel_target)
                    
                    # Combine losses (Standard Tacotron 2 practice)
                    loss = loss_mel_pre + loss_mel_post
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / ACCUM_STEPS

                # Backward Pass with Scaler
                scaler.scale(loss).backward()
                
                # Step Step: Execute optimization if accumulation is done OR it's the last batch
                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Track separate un-scaled loss for display
                total_loss += loss.item() * ACCUM_STEPS
                
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(dataloader)}], Loss: {loss.item() * ACCUM_STEPS:.4f}")
                    
            # Save Checkpoint at end of epoch
            save_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scaler, epoch, total_loss / len(dataloader), save_path)
            
    except KeyboardInterrupt:
        print("\nUsing Ctrl+C detected. Saving 'checkpoint_interrupted.pth'...")
        save_path = os.path.join(CHECKPOINT_DIR, "checkpoint_interrupted.pth")
        save_checkpoint(model, optimizer, scaler, epoch, 0.0, save_path)
        print("Training interrupted. Progress saved.")
        sys.exit(0)

if __name__ == "__main__":
    train()
