# Fix BERT Memory Issue (10GB RAM Limit)

## 🚨 Problem: BERT Uses Too Much RAM

**BERT-base-uncased**: ~110M parameters → ~4-6GB RAM + training overhead → **Exceeds 10GB limit**

## ✅ Solution: Use DistilBERT Instead

**DistilBERT**: ~66M parameters (40% smaller) → ~2-3GB RAM → **Fits in 10GB!**
- 40% smaller than BERT
- 60% faster
- Retains 97% of BERT's performance
- Perfect for comparison purposes

---

## 🔧 Quick Fix

### Step 1: Change Cell 13 (BERT Model Definition)

**Find this line**:
```python
bert_model = BERTModel().to(device)
```

**Replace with**:
```python
# Use DistilBERT - 40% smaller, fits in 10GB RAM
bert_model = BERTModel(
    model_name='distilbert-base-uncased',  # Changed from 'bert-base-uncased'
    freeze_bert=True,                       # Freeze to save memory
    hidden_dim=128                          # Smaller classifier
).to(device)
```

### Step 2: Reduce Batch Size for BERT (Cell 7)

**Find**:
```python
train_loader_bert = DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
```

**Replace with**:
```python
# Smaller batch size for BERT to save memory
train_loader_bert = DataLoader(train_dataset_bert, batch_size=8, shuffle=True)
val_loader_bert = DataLoader(val_dataset_bert, batch_size=8, shuffle=False)
test_loader_bert = DataLoader(test_dataset_bert, batch_size=8, shuffle=False)
```

### Step 3: Update Cell 24 (BERT Training)

**Find**:
```python
bert_history = bert_trainer.train(
    train_loader_bert, 
    val_loader_bert, 
    num_epochs=3,
    learning_rate=2e-5,
    model_name="BERT"
)
```

**Replace with**:
```python
print("Training DistilBERT (40% smaller, memory-efficient)")
bert_history = bert_trainer.train(
    train_loader_bert, 
    val_loader_bert, 
    num_epochs=3,                    # Short epochs since frozen
    learning_rate=2e-5,              # Standard for transformers
    weight_decay=1e-4,               # Light regularization
    patience=2,                      # Early stopping
    model_name="DistilBERT"          # Updated name
)

print(f"\nDistilBERT Training Summary:")
print(f"Best validation accuracy: {max(bert_history['val_acc']):.2f}%")
print(f"Parameters: ~66M (vs 110M for BERT-base)")
print(f"Memory usage: ~2-3GB (vs 4-6GB for BERT-base)")
```

---

## 📊 Comparison: BERT vs DistilBERT

| Model | Parameters | RAM Usage | Speed | Performance |
|-------|-----------|-----------|-------|-------------|
| BERT-base | 110M | 4-6GB | 1x | 100% |
| **DistilBERT** | 66M | 2-3GB | 1.6x | 97% |
| Benefit | -40% | **-50%** | **+60%** | -3% |

**Verdict**: DistilBERT is perfect for your use case! ✅

---

## 🎯 Alternative Options (If Still Too Much)

### Option A: Use TinyBERT (Even Smaller)
```python
model_name='huawei-noah/TinyBERT_General_4L_312D'  # ~14M parameters
```

### Option B: Freeze All Layers (Only Train Classifier)
```python
bert_model = BERTModel(
    model_name='distilbert-base-uncased',
    freeze_bert=True,        # Freeze all BERT layers
    hidden_dim=64            # Very small classifier
)
```

### Option C: Use Gradient Checkpointing
```python
# In BERTModel.__init__ after loading model:
self.bert.gradient_checkpointing_enable()  # Trade compute for memory
```

### Option D: Skip BERT If Still Problems
```python
# In Cell 24, wrap in try-except:
try:
    # Train BERT
    bert_results = bert_trainer.train(...)
except RuntimeError as e:
    print(f"⚠️ Skipping BERT due to memory: {e}")
    print("Using RNN, LSTM, and Custom Transformer results only")
    bert_results = None
```

---

## 📝 Complete Cell 13 Replacement

Replace the entire Cell 13 with this memory-efficient version:

```python
# 3. Pre-trained Transformer Model (Memory-Efficient DistilBERT)
class BERTModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout=0.3, 
                 freeze_bert=True, hidden_dim=128):
        super(BERTModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_bert = freeze_bert
        
        # Load pre-trained model (DistilBERT by default)
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters to save memory
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"✓ Frozen {model_name} layers to save memory")
        
        # Get BERT output dimension
        bert_output_dim = self.bert.config.hidden_size
        
        # Smaller classification head to save memory
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize classifier weights
        self.init_classifier_weights()
    
    def init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

# Test DistilBERT model (memory-efficient)
print("Creating DistilBERT model (40% smaller than BERT-base)...")
bert_model = BERTModel(
    model_name='distilbert-base-uncased',
    freeze_bert=True,       # Freeze to save memory
    hidden_dim=128          # Smaller classifier
).to(device)

trainable_params = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in bert_model.parameters())

print(f"DistilBERT Model created")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} (classifier only)")
print(f"Frozen parameters: {total_params - trainable_params:,}")
print(f"Memory efficient: ~2-3GB vs 4-6GB for BERT-base")

# Test forward pass with BERT data
sample_batch_bert = next(iter(train_loader_bert))
sample_input_bert = sample_batch_bert['input_ids'].to(device)
sample_mask_bert = sample_batch_bert['attention_mask'].to(device)
sample_output_bert = bert_model(sample_input_bert, sample_mask_bert)
print(f"DistilBERT output shape: {sample_output_bert.shape}")
print("✅ DistilBERT ready for training!")
```

---

## 🔍 Memory Monitoring

Add this to check memory usage during training:

```python
import gc
import torch

def check_memory():
    """Check current memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    
    import psutil
    process = psutil.Process()
    print(f"RAM Usage: {process.memory_info().rss / 1e9:.2f}GB")

# Call before and after BERT training
check_memory()
```

---

## ✅ Expected Results

### After Fix:
```
✓ DistilBERT loads successfully
✓ Memory usage: 2-3GB (within 10GB limit)
✓ Training speed: 60% faster than BERT
✓ Accuracy: 80-88% (comparable to BERT)
✓ Perfect for comparison purposes
```

### Comparison Table:
```
RNN:               65-75% (~300K params)
LSTM:              70-80% (~300K params)
Custom Transformer: 75-85% (~400K params)
DistilBERT:        80-88% (~66M params, pre-trained)
```

---

## 📋 Verification Checklist

After applying fixes:
- [ ] Cell 13 uses `distilbert-base-uncased`
- [ ] `freeze_bert=True` is set
- [ ] Batch size reduced to 8 for BERT loaders
- [ ] Cell 13 runs without memory errors
- [ ] Training shows frozen parameters message
- [ ] DistilBERT trains within memory limit
- [ ] Results show DistilBERT performs well

---

## 🎓 Why This Works

1. **DistilBERT is distilled from BERT**:
   - Trained to mimic BERT's behavior
   - 40% fewer parameters
   - 97% of BERT's performance retained

2. **Freezing layers saves memory**:
   - No gradients for frozen layers
   - Only classifier trains
   - Much less memory overhead

3. **Smaller batch size**:
   - Fewer samples in memory at once
   - Slightly slower but much safer

4. **Perfect for comparison**:
   - Shows benefit of pre-training
   - Demonstrates transformer performance
   - Meets project requirements

---

## 💡 Bottom Line

**Use DistilBERT instead of BERT**:
- ✅ 40% smaller (66M vs 110M params)
- ✅ Fits in 10GB RAM easily
- ✅ 60% faster training
- ✅ 97% of BERT's performance
- ✅ Perfect for your comparison needs

**Apply the fixes above and BERT will work within your hardware limits!** 🚀

