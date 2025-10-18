# GLU Pruning - Width Pruning Evaluation

## Project Overview
Evaluation of structured width pruning in GLU-MLP layers using expansion ratio modification.

**Focus:** Impact of reducing MLP expansion ratio on model capabilities.

---

## Models

### Base Models
- `Llama-3.2-1B`
- `Llama-3.2-3B`

### Instruct Models
- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B-Instruct`

**Total:** 4 models

---

## Benchmarks

### Base Models (10 benchmarks)

| Benchmark | Type | Config | Rationale |
|-----------|------|--------|-----------|
| **WikiText-2 PPL** | Perplexity | - | Fundamental language modeling capability |
| **MMLU** | Knowledge | 5-shot | Knowledge stored in weights (MLP-sensitive) |
| **ARC-Challenge** | Reasoning | 0-shot | Depth-sensitive reasoning |
| **HellaSwag** | Common Sense | 0-shot | Universal in pruning literature |
| **WinoGrande** | Common Sense | 0-shot | Standard suite (90%+ papers) |
| **PIQA** | Physical Reasoning | 0-shot | Universal, fundamental |
| **BoolQ** | QA | 0-shot | Non-monotonic behavior at high pruning |
| **Lambada** | Context | 0-shot | Resistant to pruning (in-context learning) |
| **TruthfulQA** | Truthfulness | MC1/MC2 | May improve post-pruning (reduces false knowledge) |
| **GSM8K** | Math Reasoning | 5-8 shot CoT | Extremely fragile stress test |

### Instruct Models (+1 additional)

| Benchmark | Type | Config | Rationale |
|-----------|------|--------|-----------|
| **IFEval** | Instruction Following | 0-shot | Core instruct capability |

**Total:** 11 benchmark evaluations per model

---

## Key Design Decisions

### Why the selected benchmarks?

1. **Dichotomy measurement:**
   - Knowledge in weights (MMLU, TruthfulQA) → MLP-sensitive
   - Algorithmic processing (Lambada) → MLP-resistant

2. **Critical additions from literature:**
   - WikiText-2 PPL: Most common metric (10+ papers)
   - WinoGrande + PIQA: Missing from original, universal in 2023-2025 papers
   - TruthfulQA: Can show improvement with width pruning

### Neuron Selection Method Comparison

Before conducting the main pruning experiments, we empirically validated three neuron importance metrics for GLU architectures:

- **MAW (Maximum Absolute Weight)** - Selected method ✅
- **VOW (Variance of Weights)** - Rejected due to catastrophic performance
- **PON (Product of Norms)** - Rejected due to catastrophic performance

**Key Finding:** At just 10% pruning on Llama-3.2-1B, VOW and PON caused perplexity increases of 9,000%+ on Lambada, while MAW showed acceptable 50% degradation. This validates our architectural understanding that GLU's gating mechanism requires magnitude-aware importance metrics.

See [Notebook 00 Neuron Selection Method](notebooks/00_Neuron_Selection_Method_Comparison.ipynb) for full experimental details.

---

## Execution Details

### Framework
- **EleutherAI LM Evaluation Harness** for reproducibility

### Estimated Time (Colab)
- **Per model:** ~4.5-5 hours
- **Total (4 models):** ~18-20 hours

### Resource Requirements
- GPU: T4/V100 level (Colab)
- RAM: ~15GB

---

## Expected Insights

### Width pruning hypothesis:
1. **Selective degradation:** Factual knowledge (MMLU) degrades
2. **Preserved:** Context processing (Lambada) survives
3. **Potential improvement:** Truthfulness (TruthfulQA) may increase
4. **Non-linear:** BoolQ may show U-curve behavior

### Paper narrative:
> "Width pruning in GLU-MLP layers selectively reduces memorized knowledge capacity while preserving algorithmic processing capabilities, potentially improving model truthfulness."

---

## Next Steps

- [ ] Setup LM Evaluation Harness
- [ ] Implement width pruning at different expansion ratios
- [ ] Run benchmark suite on all 4 models
- [ ] Analyze capability fragility hierarchy
- [ ] Compare base vs instruct degradation patterns
