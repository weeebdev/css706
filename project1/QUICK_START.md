# Quick Start Guide - Sentiment Analysis Project

## ⚡ Getting Started in 5 Minutes

### 1. Install Dependencies
```bash
cd /root/phd/css706/project1
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter lab Sentiment_Analysis_Project.ipynb
```

### 3. Run All Cells Sequentially
Press `Shift + Enter` through each cell, or use `Run > Run All Cells`

---

## 📋 What to Expect

### Cell-by-Cell Guide

**Cells 1-2**: Setup and imports
- ✅ Expected: `Using device: cpu` or `cuda`

**Cell 3**: Load 20 Newsgroups dataset
- ✅ Expected: ~18K samples loaded
- ⏱️ Time: ~30 seconds

**Cell 4**: Preprocessing + Fixes
- ✅ Expected: "Data is ready for training - no overlaps, no empty texts!"
- 🔍 Watch for: 0 overlaps, 2000/499/498 samples

**Cells 5-8**: Tokenization and verification
- ✅ Expected: Vocab size ~5000, all variables defined

**Cells 9-10**: Model architecture overview
- 📖 Read: Understanding the models

**Cells 11-14**: Model definitions
- ✅ Expected: 4 models created successfully
- 📊 Parameters: ~80K-180K (RNN/LSTM/Transformer), ~110M (BERT)

**Cells 15-18**: Memory-efficient training setup
- ✅ Expected: Tiny models with 16-32 dims
- 💾 Memory: ~7-8 GB

**Cell 19**: Anti-overfitting documentation
- 📖 Read: Understanding regularization techniques

**Cell 20**: Training framework
- ✅ Expected: SentimentTrainer class created

**Cells 21**: Training overview
- 📖 Read: What to expect during training

**Cell 22**: Train RNN
- ⏱️ Time: ~2-5 minutes
- ✅ Expected: 70-85% train, 65-80% val, gap < 10%
- ⚠️ Watch for: Overfitting warnings if gap > 10%

**Cell 23**: Train LSTM
- ⏱️ Time: ~2-5 minutes
- ✅ Expected: Similar or slightly better than RNN

**Cell 24**: Train BERT (optional)
- ⏱️ Time: ~10-20 minutes
- 💾 Memory: High (skip if RAM < 10GB)

**Cell 25**: Train Custom Transformer
- ⏱️ Time: ~2-5 minutes
- ✅ Expected: Best custom model

**Cells 26-32**: Analysis and comparison
- 📊 Charts: Performance comparisons
- 📈 Metrics: Accuracy, F1, parameters

**Cell 33**: Final comprehensive report
- 📖 Read: Complete project summary

---

## ✅ Healthy Results Checklist

### During Training
- [ ] Train accuracy: 70-85% (NOT 100%)
- [ ] Validation accuracy: 65-80% (NOT 100%)
- [ ] Train-val gap: < 10%
- [ ] Early stopping triggers (patience 3)
- [ ] LR reduces when plateau

### Data Quality
- [ ] Zero overlaps between splits
- [ ] No empty texts
- [ ] ~2000/499/498 samples (train/val/test)

### Memory Usage
- [ ] Stays under 10GB
- [ ] Models load/unload cleanly
- [ ] No out-of-memory errors

---

## 🚨 Troubleshooting

### Problem: "RuntimeError: pack_padded_sequence"
**Solution**: Re-run Cell 4 to filter empty texts

### Problem: 100% train and val accuracy
**Solution**: Models are overfitting, check:
- Dropout is 0.6 (not 0.1)
- Weight decay is 1e-2 (not 1e-5)
- Early stopping patience is 3 (not 10)

### Problem: "CUDA out of memory"
**Solution**: 
- Use CPU: `device = torch.device('cpu')`
- Skip BERT (Cell 24)
- Reduce batch size in Cell 7

### Problem: Data overlaps detected
**Solution**: Re-run Cell 4, ensure output shows "0 overlaps"

### Problem: Cell takes too long
**Solution**: Reduce samples:
```python
max_train_samples = 1000  # Instead of 2000
max_val_samples = 250     # Instead of 500
max_test_samples = 250    # Instead of 500
```

---

## 📊 Performance Expectations

### Typical Results (Small Dataset)
| Model | Parameters | Train Acc | Val Acc | Gap |
|-------|-----------|-----------|---------|-----|
| RNN | ~80K | 75-80% | 70-75% | ~5% |
| LSTM | ~180K | 78-83% | 73-78% | ~5% |
| Custom Transformer | ~187K | 80-85% | 75-80% | ~5% |
| BERT | ~110M | 85-90% | 80-85% | ~5% |

### What's Healthy
- ✅ Gap < 10%: Good generalization
- ✅ Val similar to test: Proper evaluation
- ✅ Early stopping: Prevents overtraining

### What's Problematic
- ❌ 100% train/val: Memorization
- ❌ Gap > 20%: Severe overfitting
- ❌ Data overlaps: Data leakage

---

## 🎯 Project Goals

1. ✅ Compare RNN vs LSTM vs Transformer
2. ✅ Find optimal hyperparameters
3. ✅ Prevent overfitting
4. ✅ Handle small datasets properly
5. ✅ Work within 10GB RAM
6. ✅ Generate comprehensive report

---

## 📚 Additional Resources

- **FIXES_APPLIED.md**: Detailed technical fixes
- **README.md**: Project overview
- **Cell 19**: Anti-overfitting explanation
- **Cell 33**: Comprehensive report

---

## ⏱️ Total Estimated Time

- Setup: 5 minutes
- Data loading: 2 minutes
- RNN training: 3-5 minutes
- LSTM training: 3-5 minutes
- Custom Transformer: 3-5 minutes
- BERT (optional): 10-20 minutes
- Analysis: 2 minutes

**Total**: ~20-40 minutes (without BERT: ~15-20 minutes)

---

## 💡 Pro Tips

1. **Start small**: Run with reduced samples first to verify everything works
2. **Monitor memory**: Watch memory usage in Cell 16 and beyond
3. **Read explanations**: Cells 19 and 33 explain key concepts
4. **Check fixes**: Cell 4 output should show 0 overlaps
5. **Save checkpoints**: Models are not saved by default, add save logic if needed
6. **Experiment**: Try different hyperparameters after initial run

---

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ No runtime errors
- ✅ Data shows 0 overlaps
- ✅ Training shows realistic accuracy (70-85%, not 100%)
- ✅ Train-val gap stays < 10%
- ✅ Early stopping triggers naturally
- ✅ Comprehensive report generates at the end

**Happy Training! 🚀**


