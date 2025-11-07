# GLU Pruning - Llama-3.2 Width Pruning Evaluation

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

### Base Models (13 benchmarks)

| Benchmark | Type | Config | Rationale |
|----------|------|--------|-----------|
| WikiText-2 PPL | Perplexity | 0-shot | Fundamental language modeling capability |
| MMLU | Knowledge | 5-shot | Knowledge stored in weights (MLP-sensitive) |
| ARC-Challenge | Reasoning | 0-shot | Depth-sensitive reasoning |
| HellaSwag | Common Sense | 0-shot | Universal in pruning literature |
| WinoGrande | Common Sense | 0-shot | Standard suite (90%+ papers) |
| PIQA | Physical Reasoning | 0-shot | Universal, fundamental |
| BoolQ | QA | 0-shot | Non-monotonic behavior at high pruning |
| Lambada | Context | 0-shot | Language modeling stress test (context-dependent prediction) |
| TruthfulQA MC1 | Truthfulness | 0-shot | May improve post-pruning (single correct answer) |
| TruthfulQA MC2 | Truthfulness | 0-shot | May improve post-pruning (reduces false knowledge) |
| GSM8K | Math Reasoning | 5-shot | Extremely fragile stress test |
| IFEval | Instruction Following | 0-shot | Core instruct capability |
| MUSR | Multi-Step Reasoning | 0-shot | Complex compositional reasoning benchmark |


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
## Results Highlights - Llama-3.2-1B

### Key Findings

**1. Instruction Following Improves (+75%)**
- IFEval performance peaks at 30% pruning with +75% improvement over baseline
- Remains elevated (+46%) even at optimal 140% Expansion (40% pruning) configuration
- Mechanism unclear; may relate to reduced capacity for elaboration

**2. Validates 140% Expansion (40% Pruning) as Effective configuration**
- Maintains 87% of MMLU performance (-13.6%)
- Improves instruction following by +46.5% (IFEval)
- Improves multi-step reasoning by +26.1% (MUSR)
- Improves truthfulness metrics by +13.9% (see caveats in detailed [results](results/))

**3. Clear Performance Dichotomy**
- **Improves:** IFEval (+75%), MUSR (+26%), TruthfulQA (+24%)
- **Degrades:** GSM8K (-69%), Perplexity (+2,691%), HellaSwag (-41%)
- Reveals trade-off: pruning sacrifices computational fluency but improves performance on literal instruction-following tasks

📊 **[View complete results and analysis →](results/)**
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

### Validation Status (Llama-3.2-1B):
Selective degradation: MMLU degrades moderately (-14% at 40% pruning)
Preserved capabilities: Some metrics show resistance (BoolQ stable until 50%)
Truthfulness metrics improve: TruthfulQA-MC2 +14% at 40% (baseline near-random, see detailed [analysis](results/))
Non-monotonic patterns: IFEval peaks at 30% (+75%), remains elevated at 40% (+47%)
Unexpected gains: IFEval improves substantially, MUSR shows consistent gains (+26%)

### Paper narrative:
> "Width pruning in GLU-MLP layers selectively reduces memorized knowledge capacity while preserving algorithmic processing capabilities, potentially improving model truthfulness."

---

## Progress Tracker

- [x] **Llama-3.2-1B (Base)** - Completed ✅ 
  - 7 pruning levels (0-60%)
  - 13 benchmarks evaluated
  - [See detailed results](results/)
- [x] **Llama-3.2-3B (Base)** - Completed ✅
   - 7 pruning levels (0-60%)
   - 13 benchmarks evaluated
- [ ] Llama-3.2-1B-Instruct
- [ ] Llama-3.2-3B-Instruct
