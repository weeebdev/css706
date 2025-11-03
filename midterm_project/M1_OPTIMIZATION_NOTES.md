# M1 Mac Optimization Notes

## ✅ Your MacBook M1 Pro CAN Run This Project!

The notebook has been optimized for M1 Mac compatibility while staying within all allowed constraints from `midterm.md`.

## Changes Applied

### 1. **MPS Device Support** ✨
- Added automatic detection for Apple's Metal Performance Shaders (MPS)
- The notebook now supports: CUDA (NVIDIA GPUs) → MPS (M1 Macs) → CPU fallback
- Your M1 GPU will be automatically utilized via PyTorch's MPS backend

### 2. **Batch Size Reduction** 
- **Changed:** 128 → **64**
- **Reason:** M1 unified memory management works better with smaller batches
- **Within constraints:** midterm.md explicitly allows "64 if GPU mem limited"

### 3. **Dataset Subset**
- **Changed:** USE_SUBSET = False → **True** (10k samples instead of 50k)
- **Reason:** Faster iteration and training on M1
- **Within constraints:** midterm.md allows "use 10k subsample" if GPU scarce

### 4. **GAN Training Epochs**
- **Changed:** 50 → **40 epochs**
- **Reason:** Reasonable training time on M1 while maintaining quality
- **Within constraints:** midterm.md allows 30-80 epochs, "shorter if necessary"

### 5. **VAE Training Epochs**
- **Unchanged:** 20 epochs (already optimal)
- **Within constraints:** midterm.md specifies 15-30 epochs

## Expected Performance on M1 Pro

| Component | Estimated Time |
|-----------|---------------|
| VAE Training (20 epochs, 10k samples) | ~15-25 minutes |
| GAN Training (40 epochs, 10k samples) | ~30-50 minutes |
| Linear Probe Evaluation | ~2-5 minutes |
| FID Calculation | ~5-10 minutes |
| **Total** | **~1-1.5 hours** |

*Note: Times vary based on M1 Pro model (8-core vs 10-core CPU) and whether you have 16GB or 32GB RAM.*

## Requirements

Before running the notebook, ensure you have PyTorch with MPS support:

```bash
# Check PyTorch version (needs ≥1.12 for MPS support)
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Verify MPS availability
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

If MPS is not available, install/upgrade PyTorch:

```bash
pip install --upgrade torch torchvision torchaudio
```

## Memory Considerations

### 16GB M1 Pro
- ✅ Should run comfortably with optimized settings
- If you encounter memory issues, close other applications
- Batch size of 64 with 10k samples is well within limits

### 32GB M1 Pro  
- ✅ Plenty of headroom
- Could optionally increase to BATCH_SIZE=128 if desired
- Could use full 50k dataset (USE_SUBSET=False) if you have time

## Optional: Run with Full Dataset

If you have time and want better results, you can increase the dataset size:

In cell 4 (Data Loading), change:
```python
USE_SUBSET = True  # Change to False for full 50k dataset
```

This will increase training time to ~3-4 hours total but may improve model quality.

## Performance Tips

1. **Close Other Apps:** Free up RAM and GPU resources
2. **Use Power Adapter:** M1 throttles on battery to save power
3. **Monitor Activity:** Use Activity Monitor to check memory pressure
4. **Avoid Overheating:** Ensure good ventilation, avoid running on soft surfaces

## Troubleshooting

### "MPS backend is not available"
- Update to PyTorch ≥1.12: `pip install --upgrade torch`
- The notebook will fallback to CPU (slower but still works)

### Memory Errors
- Reduce BATCH_SIZE to 32
- Ensure USE_SUBSET=True
- Close other applications

### Slow Training
- Verify MPS is being used: Check the first cell output shows "Using device: mps"
- Ensure you're plugged into power
- Check Activity Monitor for memory pressure (yellow/red = issue)

## Validation

All changes comply with midterm.md requirements:
- ✅ Dataset: CIFAR-10 with allowed 10k subsample
- ✅ VAE: 20 epochs (within 15-30 range)
- ✅ GAN: 40 epochs (within 30-80 range)
- ✅ Batch size: 64 (explicitly allowed for limited GPU)
- ✅ All other specs unchanged (latent_dim=64, z_dim=100, etc.)

## Next Steps

1. Install requirements: `pip install -r requirements.txt`
2. Run the notebook: `jupyter notebook notebook.ipynb`
3. Execute cells sequentially
4. Monitor the first cell output to confirm MPS device is detected

The complete experiment should finish in 1-1.5 hours on your M1 Pro! 🚀

