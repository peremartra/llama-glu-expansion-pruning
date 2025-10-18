# Notebooks

This directory contains the Jupyter notebooks used for experimentation and analysis in the GLU pruning research.

## Index

### 00 - Neuron Selection Method Comparison
**File:** `00_Neuron_Selection_Method_Comparison.ipynb`

**Purpose:** Empirical validation of neuron importance metrics for GLU pruning

**What it does:**
- Compares three methods: MAW (Maximum Absolute Weight), VOW (Variance of Weights), PON (Product of L1 Norms)
- Tests Llama-3.2-1B at 10% pruning
- Evaluates on WikiText-2, BoolQ, and Lambada benchmarks

**Key Results:**
- **MAW:** WikiText PPL 17.49 (+51%), Lambada PPL 20.54 (+259%) ✅ **SELECTED**
- **VOW:** WikiText PPL 50.56 (+337%), Lambada PPL 532.36 (+9,207%) ❌ Catastrophic
- **PON:** WikiText PPL 72.52 (+527%), Lambada PPL 2032.80 (+35,440%) ❌ Catastrophic

**Conclusion:** MAW is the only viable method for GLU architectures. This validates our theoretical understanding that the gating mechanism requires magnitude-aware importance metrics.

**Runtime:** ~1 hour on Colab T4 GPU

---
