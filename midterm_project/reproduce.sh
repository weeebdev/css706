#!/bin/bash

# reproduce.sh - Single command to reproduce all experiments
# Midterm Project: VAE vs DCGAN on CIFAR-10

set -e  # Exit on error

echo "=================================================="
echo "Starting Reproducible Experiment Pipeline"
echo "=================================================="
echo ""

# Check if running in correct environment
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

# Create necessary directories
echo "[1/6] Creating directories..."
mkdir -p results
mkdir -p checkpoints
mkdir -p figures
mkdir -p fid_real
mkdir -p fid_fake
echo "✓ Directories created"
echo ""

# Install dependencies
echo "[2/6] Installing dependencies..."
pip install --quiet torch torchvision tqdm scikit-learn matplotlib numpy jupyter pytorch-fid --break-system-packages
echo "✓ Dependencies installed"
echo ""

# Run the main notebook (convert to script and execute)
echo "[3/6] Running main training pipeline..."
jupyter nbconvert --to script notebook.ipynb --output train_script
python train_script.py
echo "✓ Training complete"
echo ""

# Calculate FID score
echo "[4/6] Calculating FID score..."
if [ -d "fid_real" ] && [ -d "fid_fake" ]; then
    if command -v pytorch-fid &> /dev/null; then
        FID_SCORE=$(python -m pytorch_fid fid_real fid_fake --device cuda 2>&1 | grep "FID:" | awk '{print $2}')
        echo "$FID_SCORE" > results/fid.txt
        echo "✓ FID Score: $FID_SCORE (saved to results/fid.txt)"
    else
        echo "Warning: pytorch-fid not found. Skipping FID calculation."
        echo "Install with: pip install pytorch-fid"
    fi
else
    echo "Warning: FID directories not found. Skipping FID calculation."
fi
echo ""

# Generate summary report
echo "[5/6] Generating summary..."
python << EOF
import json
import os

# Load results if available
if os.path.exists('results/summary.json'):
    with open('results/summary.json', 'r') as f:
        summary = json.load(f)
    
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print("\nVAE Metrics:")
    print(f"  Final ELBO: {summary['vae']['final_elbo']:.4f}")
    print(f"  Reconstruction Loss: {summary['vae']['final_recon_loss']:.4f}")
    print(f"  KL Divergence: {summary['vae']['final_kl_loss']:.4f}")
    print(f"  Linear Probe Accuracy: {summary['vae']['linear_probe_accuracy']:.4f}")
    
    print("\nGAN Metrics:")
    print(f"  Final D Loss: {summary['gan']['final_d_loss']:.4f}")
    print(f"  Final G Loss: {summary['gan']['final_g_loss']:.4f}")
    
    if os.path.exists('results/fid.txt'):
        with open('results/fid.txt', 'r') as f:
            fid = f.read().strip()
        print(f"  FID Score: {fid}")
    
    print("\nHyperparameters:")
    for key, value in summary['hyperparameters'].items():
        print(f"  {key}: {value}")
    print("="*60)
else:
    print("Summary file not found. Please check if training completed successfully.")
EOF
echo ""

# Verify deliverables
echo "[6/6] Verifying deliverables..."
REQUIRED_FILES=(
    "results/vae_recons.png"
    "results/gan_samples.png"
    "results/linear_probe.csv"
    "checkpoints/vae_model.pth"
    "checkpoints/gan_model.pth"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo ""
    echo "✓ All deliverables generated successfully!"
else
    echo ""
    echo "⚠ Warning: ${#MISSING_FILES[@]} file(s) missing"
fi

echo ""
echo "=================================================="
echo "Experiment Complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - results/       (output images and metrics)"
echo "  - checkpoints/   (trained model weights)"
echo "  - figures/       (training curves and analysis)"
echo ""
echo "Next steps:"
echo "  1. Review notebook.ipynb for detailed analysis"
echo "  2. Check results/ folder for generated images"
echo "  3. Use checkpoints/ for further experiments"
echo ""
