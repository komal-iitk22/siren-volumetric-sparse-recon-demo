# SIREN Volumetric Sparse Reconstruction (Demo)

This project demonstrates an **end-to-end pipeline** for reconstructing large volumetric scalar fields from **sparse samples** using **Sinusoidal Representation Networks (SIREN)**. The implementation explores multiple sampling strategies — random, histogram-based, gradient-aware, hybrid, and adaptive — and evaluates reconstruction quality using **Peak Signal-to-Noise Ratio (PSNR)**.

## Why This Matters

- Shows how **implicit neural representations** can efficiently compress and reconstruct complex volumetric data.
- Provides a modular, clean implementation suitable for research, learning, or extension to real-world scientific datasets.
- Demonstrates my ability to bridge **research concepts with production-ready engineering**, with a focus on performance, clarity, and reproducibility.

## Features

- Multiple sampling methods (random, histogram, gradient-aware, hybrid, adaptive).
- Residual SIREN model implemented in PyTorch.
- End-to-end workflow: sampling → training → reconstruction → evaluation (PSNR).
- Support for both synthetic datasets and user-provided `.vti` volumes (kept outside the repo).

## Quickstart

### 1. Setup environment

```bash
python -m venv .venv
# Activate venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

2. Run with synthetic dataset

python -m src.main --dataset synthetic --method random_hist_based --sampling 5 --outdir outputs/

3. Run with your own .vti dataset

Place your .vti file in the data/ folder (this folder is git-ignored).

Run:

python -m src.main --dataset vti --vti_path data/YourVolume.vti --method hist_grad --sampling 5 --outdir outputs/

Outputs

outputs/*.vtp — sampled point cloud

outputs/recon_*.vti — reconstructed scalar field (adds recon array)

PSNR value printed in the console

Notes

This repository is a demo version created for professional review (e.g., job applications).
It does not include any proprietary or institutional datasets. Users can test with their own .vti files or the provided synthetic volume generator.

```
