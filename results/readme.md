# Experimental Results - GLU Width Pruning

This directory contains evaluation results for **Llama-3.2-1B** base model with structured width pruning using the MAW (Maximum Absolute Weight) neuron selection method.

---

## Quick Summary Llama-3.2-1B

**Best Configuration:** 40% pruning (corresponding to 140% expansion ratio from the original paper)

### Benchmarks

| Pruning %       | MMLU            | GSM8K           | ARC-C           | HellaSwag       | BoolQ           | PiQA            | WikiText PPL      | Lambada PPL         |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ----------------- | ------------------- |
| **0%** baseline | 0.3111          | 0.0660          | 0.3626          | 0.6363          | 0.6343          | 0.7437          | 11.57             | 5.75                |
| 10%             | 0.2511 (-19.3%) | 0.0394 (-40.3%) | 0.3328 (-8.2%)  | 0.5791 (-9.0%)  | 0.626 (-1.3%)   | 0.7214 (-3.0%)  | 17.50 (+51.3%)    | 20.59 (+258.1%)     |
| 20%             | 0.2661 (-14.5%) | 0.0227 (-65.6%) | 0.3080 (-15.1%) | 0.5076 (-20.2%) | 0.6232 (-1.8%)  | 0.6850 (-7.9%)  | 25.05 (+116.5%)   | 33.07 (+475.1%)     |
| 30%             | 0.2610 (-16.1%) | 0.0159 (-75.9%) | 0.2637 (-27.3%) | 0.4382 (-31.1%) | 0.626 (-1.3%)   | 0.6643 (-10.7%) | 38.58 (+233.5%)   | 55.74 (+869.4%)     |
| **40%**         | 0.2689 (-13.6%) | 0.0205 (-68.9%) | 0.2509 (-30.8%) | 0.3737 (-41.3%) | 0.622 (-1.9%)   | 0.6235 (-16.2%) | 56.33 (+386.9%)   | 90.38 (+1471.8%)    |
| 50%             | 0.2606 (-16.2%) | 0.0167 (-74.7%) | 0.2474 (-31.8%) | 0.3251 (-48.9%) | 0.6141 (-3.2%)  | 0.6088 (-18.2%) | 117.04 (+911.5%)  | 428.3 (+7348.7%)    |
| 60%             | 0.2554 (-17.9%) | 0.0205 (-68.9%) | 0.2398 (-33.9%) | 0.2909 (-54.3%) | 0.5535 (-12.7%) | 0.5756 (-22.6%) | 322.95 (+2691.0%) | 2941.08 (+51049.2%) |

### Benchmarks That Improve With Pruning

| Pruning % | IFEval | MUSR | TruthfulQA-MC1 | TruthfulQA-MC2 | WinoGrande |
|-----------|--------|------|----------------|----------------|------------|
| **0%** (baseline) | 0.1035 | 0.3399 | 0.2338 | 0.3772 | 0.5991 |
| 10% | 0.1423 (**+37.5%**) | 0.3624 (**+6.6%**) | 0.2460 (**+5.2%**) | 0.4026 (**+6.7%**) | 0.6093 (**+1.7%**) |
| 20% | 0.1275 (**+23.2%**) | 0.3638 (**+7.0%**) | 0.2424 (**+3.7%**) | 0.4153 (**+10.1%**) | 0.5935 (-0.9%) |
| 30% | 0.1811 (**+75.0%**) | 0.3757 (**+10.5%**) | 0.2448 (**+4.7%**) | 0.4252 (**+12.7%**) | 0.5722 (-4.5%) |
| **40%** | 0.1516 (**+46.5%**) | 0.4286 (**+26.1%**) | 0.2485 (**+6.3%**) | 0.4298 (**+13.9%**) | 0.5706 (-4.8%) |
| 50% | 0.1534 (**+48.2%**) | 0.3743 (**+10.1%**) | 0.2460 (**+5.2%**) | 0.4314 (**+14.4%**) | 0.5312 (-11.3%) |
| 60% | 0.1368 (**+32.2%**) | 0.4087 (**+20.2%**) | 0.2375 (**+1.6%**) | 0.4661 (**+23.6%**) | 0.4870 (-18.7%) |

*Values in parentheses show percentage change from baseline. Bold percentages indicate improvement.*

---

## Key Observations

### 1. Instruction Following (IFEval) Dramatically Improves
- **IFEval shows substantial gains**, peaking at +75% at 30% pruning
- Performance remains elevated even at 60% pruning (+32.2%)
- Suggests that **the pruned model is less able to "digress" or "over-elaborate", which paradoxically helps in tasks that require following simple instructions to the letter.**
- This is a relevant finding for practical deployment where instruction-following is essential

### 2. Multi-Step Reasoning (MUSR) Also Benefits
- **MUSR improves up to +26.1%** at the optimal 40% pruning level
- Consistent improvements across all pruning levels (6-26%)
- Suggests improvements in structured reasoning tasks, though the mechanism remains unclear (note: GSM8K math reasoning degrades severely)
- Complements the IFEval findings on structured task execution

### 3. Truthfulness Shows Apparent Improvement (With Caveats)
- **TruthfulQA-MC2** improves progressively: +6.7% (10%) → +23.6% (60%)
- **TruthfulQA-MC1** shows modest gains: +1.6% to +6.3%
- **Critical context**: Baseline is near-random (23.38% vs ~25% chance). At 60% pruning, multiple benchmarks collapse to chance level (ARC-C: 23.98%, MMLU: 25.54%, WinoGrande: 48.70%)
- **Most likely mechanism**: Pruning flattens probability distributions, reducing the model's confidence in popular misconceptions that are over-represented in training data. This is bias reduction, not genuine knowledge improvement (note MMLU degrades -17.9% at 60%)
- **Practical benefit**: Pruning does reduce confident misinformation, but interpret cautiously at high pruning levels

### 4. Trade-offs: Knowledge vs Processing Capabilities
**What degrades:**
- **Perplexity metrics degrade severely** (WikiText +2691%, Lambada +51,000% at 60%)
- **GSM8K** (math reasoning): Highly fragile, -68.9% at 40% pruning
- **MMLU** (factual knowledge): Moderate degradation, -13.6% at 40%
- **Pattern recognition** (ARC, HellaSwag): Gradual degradation (-30% to -41% at 40%)

**What improves:**
- **Instruction following** (IFEval): Up to +75%
- **Multi-step reasoning** (MUSR): Up to +26%
- **Truthfulness** (TruthfulQA): Up to +24%

This reveals a trade-off: **width pruning sacrifices general computational capacity and knowledge-intensive tasks, but shows gains in tasks requiring literal instruction adherence**.

### 5. 140% Expansion (40% Pruning) as a Balanced Configuration
- Maintains **~87% of MMLU performance**
- Achieves **+46.5% improvement in instruction following** (IFEval)
- Delivers **+26.1% improvement in multi-step reasoning** (MUSR)
- Improves **truthfulness by 14%** (TruthfulQA-MC2)
- Balances capability preservation with improved instruction adherence

---

## Files in this Directory

- **`llama_3.2_1b_base_results.csv`**: Complete results table with all benchmarks and metrics
- **`llama_3.2_1b_base_metadata.json`**: Full experimental metadata including:
  - Hardware configuration (NVIDIA L4 GPU)
  - Exact benchmark configurations (few-shot settings)
  - Timestamp and experiment provenance
  - Pruning method details (MAW neuron selection)

---

## Experimental Details

- **Model Family:** Llama-3.2-1B (base, not instruct)
- **Pruning Method:** MAW (Maximum Absolute Weight) neuron selection
- **Architecture:** Structured width pruning in GLU-MLP layers (gate_proj + up_proj paired removal)
- **Hardware:** NVIDIA L4 GPU (24GB VRAM)
- **Evaluation Framework:** EleutherAI LM Evaluation Harness
- **Benchmarks Evaluated:** 13 tasks total
  - **Language Understanding:** WikiText (perplexity), Lambada-OpenAI
  - **Knowledge:** MMLU (5-shot), TruthfulQA-MC1/MC2
  - **Reasoning:** ARC-Challenge, HellaSwag, WinoGrande, PIQA, GSM8K (5-shot), MUSR
  - **Instruction Following:** IFEval, BoolQ

---

## Reproducibility

All results can be reproduced using the notebooks in the `notebooks/` directory:
- Pruning: See experiment notebooks with OptIFAIR library
- Evaluation: Using LM Evaluation Harness with exact configurations in metadata file

For detailed methodology, see the main repository README.
