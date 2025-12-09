# 🚀 Quick Start Guide

## What You Have

Your complete midterm project implementation for **Convolutional VAE vs DCGAN on CIFAR-10**. Everything is ready to run!

## 📁 Files Overview

```
├── notebook.ipynb              ⭐ MAIN FILE - Complete implementation
├── reproduce.sh               ⭐ Run this for full pipeline
├── README.md                  📚 Detailed documentation
├── report_template.md         📝 4-page report template
├── presentation_template.md   📊 5-slide presentation template
└── requirements.txt           📦 Python dependencies
```

## ⚡ 3 Ways to Run

### Option 1: Full Pipeline (Recommended)
```bash
# Download all files to your machine
# Open terminal in the project directory

chmod +x reproduce.sh
./reproduce.sh
```
This will automatically:
- Install dependencies
- Train both models
- Generate all results
- Calculate metrics

**Time**: ~1-2 hours (depending on GPU)

### Option 2: Interactive Jupyter Notebook
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Then run cells one by one to:
- Understand each step
- Modify hyperparameters
- Experiment with variations

**Best for**: Learning and experimentation

### Option 3: Google Colab (No local GPU needed)
1. Upload `notebook.ipynb` to Google Colab
2. Change runtime to GPU (Runtime → Change runtime type → GPU)
3. Add this cell at the beginning:
```python
!pip install pytorch-fid
```
4. Run all cells

**Best for**: No local GPU or easy sharing

## 📊 Expected Results

After running, you'll have:

### Results Folder
```
results/
├── vae_recons.png          # VAE reconstructions
├── gan_samples.png         # GAN generated samples
├── linear_probe.csv        # Classification accuracy
├── fid.txt                 # FID score
└── summary.json           # All metrics
```

### Checkpoints Folder
```
checkpoints/
├── vae_model.pth          # Trained VAE weights
└── gan_model.pth          # Trained GAN weights
```

### Typical Performance
- **VAE Linear Probe**: 45-55% accuracy
- **GAN FID Score**: 80-120 (lower is better)
- **Training Time**: VAE ~20 min, GAN ~40 min (with GPU)

## 🎯 Next Steps After Running

### 1. Review Results (5 minutes)
```bash
# Look at generated images
open results/vae_recons.png
open results/gan_samples.png

# Check metrics
cat results/linear_probe.csv
cat results/fid.txt
cat results/summary.json
```

### 2. Write Your Report (2-3 hours)
- Open `report_template.md`
- Fill in [INSERT VALUE] placeholders with your results
- Add your generated images
- Write analysis based on your observations
- Convert to PDF:
```bash
# Using pandoc
pandoc report_template.md -o report.pdf

# Or copy to Google Docs / Word
```

### 3. Create Presentation (1 hour)
- Open `presentation_template.md`
- Insert your images and metrics
- Export to slides:
  - Import to Google Slides
  - Use Marp (markdown presentations)
  - Create in PowerPoint/Keynote

### 4. Submit Deliverables
```
your_submission/
├── code/
│   ├── notebook.ipynb
│   ├── reproduce.sh
│   ├── requirements.txt
│   └── README.md
├── results/
│   ├── vae_recons.png
│   ├── gan_samples.png
│   ├── linear_probe.csv
│   └── fid.txt
├── report.pdf              # ≤4 pages
└── slides.pdf              # ≤5 slides
```

## 🔧 Troubleshooting

### Problem: Out of Memory
**Solution**: Edit notebook, reduce batch size
```python
BATCH_SIZE = 64  # or 32
```

### Problem: Training Too Slow
**Solution**: Use subset or reduce epochs
```python
USE_SUBSET = True  # Use 10k samples
VAE_EPOCHS = 10
GAN_EPOCHS = 30
```

### Problem: FID Calculation Fails
**Solution**: Install pytorch-fid
```bash
pip install pytorch-fid
```

### Problem: No GPU Available
**Solution**: Code works on CPU (slower)
- Training will take 3-5x longer
- Consider using Google Colab with free GPU

## 📚 Understanding the Code

### VAE Section (Cells 3, 6, 8)
- Architecture: Conv encoder → latent → Conv decoder
- Loss: Reconstruction + β × KL divergence
- Evaluation: Reconstructions, interpolations, linear probe

### GAN Section (Cells 4, 7, 10)
- Architecture: Generator (z→image) + Discriminator (image→real/fake)
- Loss: Adversarial minimax game
- Evaluation: FID score, sample quality

### Key Hyperparameters
```python
# VAE
latent_dim = 64
lr = 1e-3
beta = 1.0
epochs = 20

# GAN
z_dim = 100
lr = 2e-4
epochs = 50
```

## 🎓 Grading Rubric Map

| Component | Points | Files |
|-----------|--------|-------|
| **Reproducibility** | 30 | reproduce.sh, README.md |
| **Core Experiments** | 30 | notebook.ipynb, results/ |
| **Analysis** | 25 | report.pdf (failure modes) |
| **Code Quality** | 10 | Clean, modular code |
| **Presentation** | 5 | report.pdf, slides.pdf |

All components are implemented and ready!

## 💡 Tips for Success

1. **Run Early**: Don't wait until the last day
2. **Check Results**: Verify images look reasonable
3. **Understand**: Read the analysis sections in templates
4. **Experiment**: Try β=4 for VAE, see what changes
5. **Document**: Take notes on what you observe

## 🆘 Need Help?

1. Check `README.md` for detailed docs
2. Read cell comments in notebook
3. Review error messages carefully
4. Search for specific errors online
5. Ask on course forum with error details

## ⏱️ Time Allocation

| Task | Time |
|------|------|
| Setup & Install | 15 min |
| Run Training | 1-2 hours |
| Review Results | 30 min |
| Write Report | 2-3 hours |
| Create Slides | 1 hour |
| **Total** | **5-7 hours** |

## 🎉 You're Ready!

Everything is set up for you. Just:
1. Run `./reproduce.sh`
2. Review your results
3. Fill in the templates
4. Submit!

Good luck with your project! 🚀

---

**Questions?** Check the README.md or notebook comments first!
