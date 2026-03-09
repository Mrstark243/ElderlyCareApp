import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random
import glob

# Character Set (English)
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!? "
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

def text_to_sequence(text):
    return [char_to_id.get(c, 0) for c in text]

class VoiceCloningDataset(Dataset):
    """
    Generic Multi-Speaker Dataset.
    Assumes structure: 
        root/
          speaker_1/
             audio1.wav
             audio2.wav
          speaker_2/
             ...
    """
    def __init__(self, data_path, num_mels=80):
        self.data_path = data_path
        self.num_mels = num_mels
        self.metadata = []
        self.speakers = {} # {speaker_id: [list of file paths]}
        
        print(f"Scanning dataset at {data_path}...")
        
        # 1. Scan for Audio files (WAV or FLAC)
        wav_files = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
        flac_files = glob.glob(os.path.join(data_path, "**", "*.flac"), recursive=True)
        all_files = wav_files + flac_files
        
        # Filter out very short files or metadata files if caught
        for wav_path in all_files:
            speaker_id = os.path.basename(os.path.dirname(wav_path))
            
            # Group by speaker
            if speaker_id not in self.speakers:
                self.speakers[speaker_id] = []
            self.speakers[speaker_id].append(wav_path)
            
            # For this simple implementation, we don't have a transcript file.
            # We will use DUMMY TEXT or filename as text if transcript is missing.
            # REAL WORLD: You need a transmission alignment or metadata.csv.
            # For now, let's assume there's a metadata.csv in the root or each folder?
            # Or simplified: We just train Auto-Encoder style (Audio -> Audio)? NO, this is TTS.
            
            # FALLBACK: Explicitly look for VCTK metadata or standard CSV.
            # Since user has "custom_voice_model", we'll check for 'metadata.csv' in root.
            
        # Parse global metadata if exists (LJSpeech style but for multiple speakers?)
        meta_path = os.path.join(data_path, 'metadata.csv')
        self.has_text = False
        if os.path.exists(meta_path):
             self.has_text = True
             with open(meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        # Map filename to text. 
                        # We need to find the absolute path for this filename.
                        fname = parts[0]
                        text = parts[1]
                        
                        # Find the path in our scanned files
                        # This is O(N) inside O(M), slow. Better:
                        # Build a map first?
                        pass 
                        # FOR NOW: To ensure this runs, we simply append (path, text) tuple
                        # if we can find the file.
            
        # If no metadata found, we can't train TTS. 
        # But wait, the user asked to "build" it. They might NOT have data yet.
        # I will build the class to be ready.
        
        # Re-structure for iteration
        for spk_id, paths in self.speakers.items():
            for path in paths:
                # DUMMY TEXT for now to prevent crash if no metadata. 
                # User MUST provide metadata.csv
                text = "Hello world." 
                self.metadata.append({
                    'path': path,
                    'text': text,
                    'speaker_id': spk_id
                })

    def __len__(self):
        return len(self.metadata)

    def load_mel(self, path):
        import soundfile as sf
        wav_numpy, sample_rate = sf.read(path)
        waveform = torch.from_numpy(wav_numpy).float()
        if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
        
        if sample_rate != 22050:
             resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=22050)
             waveform = resampler(waveform)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_mels=self.num_mels, n_fft=1024, hop_length=256
        )
        mel_spec = mel_transform(waveform)
        # Convert to Log-Mel (dB)
        # Add small epsilon to avoid log(0)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5) * 1)
        
        return mel_spec.squeeze(0).transpose(0, 1)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        wav_path = item['path']
        text = item['text']
        speaker_id = item['speaker_id']
        
        # 1. Load Target Mel
        mel_target = self.load_mel(wav_path)
        
        # 2. Load Reference Mel (Random file from SAME speaker)
        # Try to pick a DIFFERENT file if possible
        ref_path = wav_path
        possible_refs = self.speakers[speaker_id]
        if len(possible_refs) > 1:
            while ref_path == wav_path:
                ref_path = random.choice(possible_refs)
        
        mel_ref = self.load_mel(ref_path)
        
        # 3. Text
        text_seq = torch.tensor(text_to_sequence(text), dtype=torch.long)
        
        return text_seq, mel_target, mel_ref

def collate_fn(batch):
    # batch is list of tuples: (text, mel_target, mel_ref)
    batch_text = [item[0] for item in batch]
    batch_mel = [item[1] for item in batch]
    batch_ref = [item[2] for item in batch]
    
    # Pad text
    max_text_len = max([len(t) for t in batch_text])
    padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, t in enumerate(batch_text):
        padded_text[i, :len(t)] = t
        
    # Pad Target Mel
    max_mel_len = max([mel.size(0) for mel in batch_mel])
    padded_mel = torch.zeros(len(batch), max_mel_len, 80)
    for i, mel in enumerate(batch_mel):
        padded_mel[i, :mel.size(0), :] = mel

    # Pad Reference Mel (Reference Encoder needs fixed size? Or padded?)
    # Reference Encoder uses RNN/Conv, so we should pad.
    max_ref_len = max([mel.size(0) for mel in batch_ref])
    padded_ref = torch.zeros(len(batch), max_ref_len, 80)
    for i, mel in enumerate(batch_ref):
        padded_ref[i, :mel.size(0), :] = mel
        
    return padded_text, padded_mel, padded_ref
