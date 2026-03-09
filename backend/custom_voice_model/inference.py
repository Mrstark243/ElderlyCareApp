import torch
import torchaudio
import numpy as np
import os
import argparse
from model import NanoTacotron
from data_loader import chars, text_to_sequence
import soundfile as sf

# Config
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = NanoTacotron(num_chars=len(chars)).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("Error: No checkpoints found! You need to train the model first.")
        return None

    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    if not checkpoints:
         print("Error: checkpoinst folder is empty.")
         return None
         
    # Filter for standard epoch checkpoints "checkpoint_epoch_X.pth"
    epoch_checkpoints = []
    for f in checkpoints:
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
            try:
                epoch_num = int(f.split('_')[-1].split('.')[0])
                epoch_checkpoints.append((epoch_num, f))
            except ValueError:
                continue
                
    if not epoch_checkpoints:
        # Fallback to simple modification time if no numbered checkpoints found
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
    else:
        # Sort by epoch number
        epoch_checkpoints.sort(key=lambda x: x[0])
        latest_checkpoint = epoch_checkpoints[-1][1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    
    print(f"Loading model from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

def process_audio(wav_path):
    """
    Load and preprocess reference audio for the encoder.
    """
    if not os.path.exists(wav_path):
        print(f"Reference audio not found: {wav_path}")
        return None
        
    wav_numpy, sample_rate = sf.read(wav_path)
    waveform = torch.from_numpy(wav_numpy).float()
    if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
    
    # Resample to 22050 (Model standard)
    if sample_rate != 22050:
         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=22050)
         waveform = resampler(waveform)

    # Convert to Mel
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256
    )
    mel_spec = mel_transform(waveform).squeeze(0).transpose(0, 1) # [Time, Mels]

    # Convert to Log-Mel (dB) matches data_loader.py
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    return mel_spec.unsqueeze(0).to(DEVICE) # [1, Time, Mels]

def griffin_lim(mel_spec):
    """
    Convert Mel Spectrogram back to Audio using Griffin-Lim algorithm.
    mel_spec: [1, Mels, Time]
    """
    print("Running Griffin-Lim Vocoder (converting picture to sound)...")
    
    # Inverse Mel Scale
    inv_mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1, n_mels=80, sample_rate=22050
    ).to(DEVICE)
    
    # Griffin Lim
    griffin_lim_transform = torchaudio.transforms.GriffinLim(
        n_fft=1024, hop_length=256, n_iter=60 # 60 iters for better quality
    ).to(DEVICE)
    
    # Mel -> Linear Spectrogram
    # mel_spec shape: [1, 80, Time]
    linear_spec = inv_mel_transform(mel_spec)
    
    # Linear Spectrogram -> Waveform
    waveform = griffin_lim_transform(linear_spec)
    
    return waveform.cpu().detach().numpy()

def synthesize(text, ref_audio_path, output_path="output.wav"):
    model = load_model()
    if model is None: return
    
    # 1. Prepare Text
    print(f"Cloning voice from: {ref_audio_path}")
    print(f"Text to speak: {text}")
    
    text_seq = torch.tensor(text_to_sequence(text), dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # 2. Prepare Reference
    ref_mel = process_audio(ref_audio_path)
    if ref_mel is None: return
    
    # 3. Inference
    with torch.no_grad():
        # returns: mel_out, mel_final, gate_out, alignments
        _, mel_final, gate_out, alignments = model(text_seq, ref_mel=ref_mel)
    
    # 4. Vocode (Mel -> Wav)
    # Inverse Log-Mel (dB to Power)
    mel_final_linear = torch.exp(mel_final)

    # GriffinLim expects [Batch, Freq, Time]
    # Our mel_final is [1, Time, 80]. Need transpose to [1, 80, Time].
    mel_out = mel_final_linear.transpose(1, 2)
    
    wav = griffin_lim(mel_out)
    
    # Check for silence/convergence failure
    if torch.max(torch.abs(torch.from_numpy(wav))) < 0.01:
        print("\nWARNING: Generated audio is extremely quiet (near silence).")
        print("Reason: The model's attention mechanism likely hasn't converged yet.")
        print("Solution: Continue training for more epochs (usually 50-100+ needed for Tacotron).")
    
    # 5. Save
    sf.write(output_path, wav.squeeze(), 22050)
    print(f"Success! Audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Cloning Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to speak")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference audio (wav)")
    parser.add_argument("--out", type=str, default="output.wav", help="Output filename")
    
    args = parser.parse_args()
    
    synthesize(args.text, args.ref, args.out)
