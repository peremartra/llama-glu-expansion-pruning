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
| 40%         | 0.2689 (-13.6%) | 0.0205 (-68.9%) | 0.2509 (-30.8%) | 0.3737 (-41.3%) | 0.622 (-1.9%)   | 0.6235 (-16.2%) | 56.33 (+386.9%)   | 90.38 (+1471.8%)    |
| 50%             | 0.2606 (-16.2%) | 0.0167 (-74.7%) | 0.2474 (-31.8%) | 0.3251 (-48.9%) | 0.6141 (-3.2%)  | 0.6088 (-18.2%) | 117.04 (+911.5%)  | 428.3 (+7348.7%)    |
| 60%             | 0.2554 (-17.9%) | 0.0205 (-68.9%) | 0.2398 (-33.9%) | 0.2909 (-54.3%) | 0.5535 (-12.7%) | 0.5756 (-22.6%) | 322.95 (+2691.0%) | 2941.08 (+51049.2%) |

### Benchmarks That Improve With Pruning

| Pruning % | IFEval | MUSR | TruthfulQA-MC1 | TruthfulQA-MC2 | WinoGrande |
|-----------|--------|------|----------------|----------------|------------|
| **0%** (baseline) | 0.1035 | 0.3399 | 0.2338 | 0.3772 | 0.5991 |
| 10% | 0.1423 (**+37.5%**) | 0.3624 (**+6.6%**) | 0.2460 (**+5.2%**) | 0.4026 (**+6.7%**) | 0.6093 (**+1.7%**) |
| 20% | 0.1275 (**+23.2%**) | 0.3638 (**+7.0%**) | 0.2424 (**+3.7%**) | 0.4153 (**+10.1%**) | 0.5935 (-0.9%) |
| 30% | 0.1811 (**+75.0%**) | 0.3757 (**+10.5%**) | 0.2448 (**+4.7%**) | 0.4252 (**+12.7%**) | 0.5722 (-4.5%) |
| 40% | 0.1516 (**+46.5%**) | 0.4286 (**+26.1%**) | 0.2485 (**+6.3%**) | 0.4298 (**+13.9%**) | 0.5706 (-4.8%) |
| 50% | 0.1534 (**+48.2%**) | 0.3743 (**+10.1%**) | 0.2460 (**+5.2%**) | 0.4314 (**+14.4%**) | 0.5312 (-11.3%) |
| 60% | 0.1368 (**+32.2%**) | 0.4087 (**+20.2%**) | 0.2375 (**+1.6%**) | 0.4661 (**+23.6%**) | 0.4870 (-18.7%) |

*Values in parentheses show percentage change from baseline. Bold percentages indicate improvement.*

## Quick Summary Llama-3.2-3B
**Best Configuration:** 10% pruning

### Benchmarks
| Pruning %       | MMLU            | GSM8K           | ARC-C           | HellaSwag       | BoolQ           | PiQA            | WikiText PPL      | Lambada PPL         |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ----------------- | ------------------- |
| **0%** baseline | 0.5605          | 0.2684          | 0.4582          | 0.7357          | 0.7294          | 0.7748          | 9.26              | 3.95                |
| 10%      | 0.4333 (-22.7%) | 0.1418 (-47.2%) | 0.3959 (-13.6%) | 0.6853 (-6.9%)  | 0.5046 (-30.8%) | 0.7508 (-3.1%)  | 11.88 (+28.3%)    | 6.11 (+54.7%)       |
| 20%             | 0.2909 (-48.1%) | 0.0728 (-72.9%) | 0.3669 (-19.9%) | 0.6158 (-16.3%) | 0.3972 (-45.5%) | 0.7307 (-5.7%)  | 15.86 (+71.3%)    | 8.16 (+106.6%)      |
| 30%             | 0.2307 (-58.8%) | 0.0364 (-86.4%) | 0.3123 (-31.8%) | 0.5232 (-28.9%) | 0.4269 (-41.5%) | 0.6812 (-12.1%) | 23.35 (+152.2%)   | 14.72 (+272.7%)     |
| 40%             | 0.2587 (-53.9%) | 0.0182 (-93.2%) | 0.2654 (-42.1%) | 0.4145 (-43.7%) | 0.4208 (-42.3%) | 0.6474 (-16.4%) | 42.18 (+355.5%)   | 51.02 (+1191.6%)    |
| 50%             | 0.2555 (-54.4%) | 0.0136 (-94.9%) | 0.2381 (-48.0%) | 0.3399 (-53.8%) | 0.5119 (-29.8%) | 0.6045 (-22.0%) | 74.83 (+707.9%)   | 240.72 (+5994.2%)   |
| 60%             | 0.2589 (-53.8%) | 0.0227 (-91.5%) | 0.2150 (-53.1%) | 0.2959 (-59.8%) | 0.5034 (-31.0%) | 0.5539 (-28.5%) | 162.47 (+1654.0%) | 5960.46 (+150797.7%)|

### Benchmarks That Improve With Pruning
| Pruning % | IFEval | MUSR | TruthfulQA-MC1 | TruthfulQA-MC2 | WinoGrande |
|-----------|--------|------|----------------|----------------|------------|
| **0%** (baseline) | 0.0943 | 0.3638 | 0.2497 | 0.3919 | 0.6953 |
| 10% | 0.1312 (**+39.1%**) | 0.3730 (**+2.5%**) | 0.2203 (-11.8%) | 0.3767 (-3.9%) | 0.6748 (-2.9%) |
| 20% | 0.1220 (**+29.4%**) | 0.3439 (-5.5%) | 0.2387 (-4.4%) | 0.4302 (**+9.8%**) | 0.6385 (-8.2%) |
| 30% | 0.1534 (**+62.7%**) | 0.3373 (-7.3%) | 0.2607 (**+4.4%**) | 0.4390 (**+12.0%**) | 0.5927 (-14.8%) |
| 40% | 0.1645 (**+74.4%**) | 0.3558 (-2.2%) | 0.2448 (-2.0%) | 0.4484 (**+14.4%**) | 0.5572 (-19.9%) |
| 50% | 0.1627 (**+72.5%**) | 0.3545 (-2.6%) | 0.2472 (-1.0%) | 0.4391 (**+12.0%**) | 0.4886 (-29.7%) |
| 60% | 0.1331 (**+41.1%**) | 0.3598 (-1.1%) | 0.2387 (-4.4%) | 0.4574 (**+16.7%**) | 0.4815 (-30.8%) |

*Values in parentheses show percentage change from baseline. Bold percentages indicate improvement.*

---

## Key Observations

### 1. Instruction Following (IFEval) Improvements Are Consistent Across Model Sizes
**Llama-3.2-1B:**
- **IFEval shows substantial gains**, peaking at +75% at 30% pruning
- Performance remains elevated even at 60% pruning (+32.2%)

**Llama-3.2-3B:**
- IFEval improvements follow a similar pattern, peaking at **+74.4% at 40% pruning**
- Gains remain substantial across all pruning levels (29-74%)
- Baseline performance is lower (9.43% vs 10.35% in 1B), suggesting the phenomenon is independent of initial capability

**Analysis:**
The consistency of IFEval improvements across both model sizes (1B and 3B) strengthens the hypothesis that **width pruning reduces the model's tendency to elaborate beyond instructions**. This appears to be an architectural effect rather than a size-dependent artifact. The practical implication is notable: instruction-following reliability improves even as general capability degrades. Suggests that **the pruned model is less able to "digress" or "over-elaborate", which paradoxically helps in tasks that require following simple instructions to the letter.**

### 2. Multi-Step Reasoning (MUSR) Also Benefits but Shows Size-Dependent Behavior
**Llama-3.2-1B:**
- MUSR improves consistently: **+6.6% to +26.1%** across all pruning levels
- Peak improvement at 40% pruning aligns with the optimal configuration
- Suggests improvements in structured reasoning tasks, though the mechanism remains unclear (note: GSM8K math reasoning degrades severely)
  
**Llama-3.2-3B:**
- MUSR shows **marginal changes**: +2.5% at 10%, then negative or minimal changes
- Unlike 1B, no clear improvement trend emerges
- Baseline is slightly higher (36.38% vs 33.99% in 1B)

**Analysis:** 
The divergence in MUSR behavior between model sizes suggests this benchmark may be **more sensitive to model capacity than IFEval**. The 1B model's improvements could reflect a different computational bottleneck than in 3B. This warrants further investigation but suggests caution in generalizing reasoning improvements across model scales.


### 3. Truthfulness Shows Apparent Improvement 
**Llama-3.2-1B:**
- TruthfulQA-MC2 improves progressively: +6.7% (10%) to +23.6% (60%)
- TruthfulQA-MC1 shows modest gains: +1.6% to +6.3%

**Llama-3.2-3B:**
- TruthfulQA-MC2: Similar progressive improvement, **+9.8% (20%) to +16.7% (60%)**
- TruthfulQA-MC1: More volatile, with peak at 30% (+4.4%), but generally lower or negative at other levels
- Notably, 10% pruning shows **-11.8% degradation** in MC1, contrasting with 1B's improvement

**Analysis:** Both models start near chance levels (1B: 23.38%, 3B: 24.97% for MC1). **Pruning likely flattens probability distributions, reducing confident endorsement of common misconceptions.** This can be bias reduction, rather than improving genuine knowledge (evidenced by MMLU degradation in both models).

### 4. Knowledge and Reasoning Degradation Is More Severe in Larger Models

**Comparative degradation at 10% pruning:**

| Benchmark | 1B Degradation | 3B Degradation |
|-----------|----------------|----------------|
| MMLU | -19.3% | **-22.7%** |
| GSM8K | -40.3% | **-47.2%** |
| BoolQ | -1.3% | **-30.8%** |
| HellaSwag | **-9.0%** | -6.9% |
| ARC-C | -8.2% | **-13.6%** |

**Observation:** The 3B model shows **disproportionately larger degradation** in knowledge-intensive tasks (MMLU, BoolQ) and mathematical reasoning (GSM8K) at equivalent pruning levels. This suggests that:

1. **Larger models may rely more heavily on parameter redundancy** for factual recall
2. **The pruning threshold for acceptable degradation differs by model size** - 10% is more aggressive for 3B than for 1B
3. **Simple percentage-based pruning may not be optimal** across different model scales

The severe BoolQ degradation in 3B (-30.8% at 10%) versus minimal impact in 1B (-1.3%) is particularly noteworthy, as BoolQ is a simpler binary classification task.

### 5. Perplexity Degradation Follows Similar Exponential Patterns

**Both models show catastrophic perplexity increases:**
- **1B:** WikiText +2691%, Lambada +51,049% at 60%
- **3B:** WikiText +1654%, Lambada +150,798% at 60%

Despite different absolute magnitudes, the **exponential degradation pattern is consistent**, suggesting this is a fundamental property of width pruning in autoregressive language models. The 3B model's even more extreme Lambada degradation indicates that **next-token prediction quality collapses more severely in larger pruned models**.

### 6. Optimal Pruning Configuration Is Model-Size Dependent

**Llama-3.2-1B: 40% pruning (140% expansion)**
- Maintains ~87% of MMLU performance
- +46.5% IFEval improvement
- +26.1% MUSR improvement
- +13.9% TruthfulQA-MC2 improvement

**Llama-3.2-3B: 10% pruning is most conservative**
- Maintains ~77% of MMLU performance
- +39.1% IFEval improvement
- +2.5% MUSR improvement (marginal)
- -3.9% TruthfulQA-MC2 (slight degradation)

**Conclusion:** The data strongly suggests that **pruning aggressiveness should scale inversely with model size**. The 1B model tolerates 40% pruning while maintaining substantial capabilities, whereas the 3B model shows significant degradation beyond 10%. This could be due to:

1. Architectural differences in how capacity is distributed
2. Different roles of GLU expansion ratios across model scales
3. Potential over-parameterization in smaller models versus more efficient use in larger ones

### 7. Summary of Trade-offs Across Model Sizes

**Consistent improvements (both models):**
- Instruction following (IFEval): Robust gains regardless of size
- Truthfulness (TruthfulQA-MC2): Progressive improvements with pruning

**Size-dependent behaviors:**
- Multi-step reasoning (MUSR): Benefits 1B significantly, minimal effect on 3B
- Knowledge degradation: More severe and immediate in 3B
- Optimal pruning level: 40% for 1B, 10% for 3B

**Universal degradations:**
- Perplexity: Exponential collapse at high pruning rates (both models)
- Mathematical reasoning (GSM8K): Severe fragility regardless of size
- Pattern recognition: Gradual degradation proportional to pruning

**Research implication:** Width pruning reveals a fundamental trade-off in LLM architectures: **models can maintain instruction-following and reduce confident misinformation even as factual knowledge and computational capacity degrade**. However, the optimal balance point and degradation rates are **strongly model-size dependent**, suggesting the need for size-specific pruning strategies rather than universal heuristics.

---

## Files in this Directory

- **`llama_3.2_1b_base_results.csv` & `llama_3.2_3b_base_results.csv`**: Complete results table with all benchmarks and metrics
- **`llama_3.2_1b_base_metadata.json` & `llama_3.2_3b_base_metadata.json`**: Full experimental metadata including:
  - Hardware configuration (NVIDIA L4 GPU)
  - Exact benchmark configurations (few-shot settings)
  - Timestamp and experiment provenance
  - Pruning method details (MAW neuron selection)

---

## Experimental Details

- **Model Family:** Llama-3.2-1B & Llama-3.2-3B (base, not instruct)
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
