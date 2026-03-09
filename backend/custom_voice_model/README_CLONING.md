# Voice Cloning Setup Instructions

I have upgraded your model to a **High-Quality Zero-Shot Voice Cloning System** optimized for your **RTX 3050**.

## Features Added
1.  **Resume Capability**: If you stop training (Ctrl+C), just run `python train.py` again. It will auto-detect the latest checkpoint and resume exactly where it left off.
2.  **Memory Optimization**: Uses **Mixed Precision (AMP)** and **Gradient Accumulation** to train high-quality batches even on 4GB VRAM.
3.  **High Quality Audio**: Added a **Post-Net** (Convolutional Refinement Network) and Dual-Loss training to make the voice "almost perfect" and less blurry.

## 1. Download Dataset (Recommended: VCTK)
The **VCTK** dataset contains 109 English speakers, which is perfect for teaching the model how to clone voices.

1.  **Download VCTK**: Search for "VCTK dataset download" (approx 10GB).
    *   [University of Edinburgh Format](https://datashare.ed.ac.uk/handle/10283/3443)
    *   Any version (VCTK-Corpus or similar) works.

## 2. Setup Folder Structure
1.  Create a folder named `dataset` inside `backend/custom_voice_model`.
2.  Extract the audio files so the structure looks like this:
    ```
    backend/custom_voice_model/
        dataset/
            p225/
               p225_001.wav
               p225_002.wav
            p226/
               ...
    ```
    *Note: The script recursively finds `.wav` files. Different speakers MUST be in different folders for the model to learn "Identity".*

## 3. Train
Run the training script:
```bash
python train.py
```
*   **Time Estimate**: ~30-45 mins per epoch.
*   **Target**: Train for 100-200 epochs for best results.
*   **Stopping**: Press `Ctrl+C` safely anytime.

## 4. How to Use (Inference)
Once trained (or during training to test), run:
```bash
python inference.py --text "Hello, I am cloning your voice." --ref "dataset/p225/p225_001.wav" --out "result.wav"
```
The model will output the text spoken in the voice of the reference audio.
