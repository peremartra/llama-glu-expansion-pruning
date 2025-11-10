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
- **L2 Norm** - Rejected due to high degradation
- **Random** - Rejected due to catastrophic degradation

**Key Finding:** At just 10% pruning on Llama-3.2-1B, the `L2` and `Random` methods caused perplexity increases of over 500% on Lambada, while `MAW` showed acceptable degradation. This validates our architectural understanding that GLU's gating mechanism requires magnitude-aware importance metrics.

See [Notebook 00 Neuron Selection Method](notebooks/00_Neuron_Selection_Method_Comparison.ipynb) for full experimental details.

---
## Notebooks

The `/notebooks` directory contains all the Jupyter notebooks used to run the experiments and generate the analyses for this project. For a detailed breakdown of each notebook's purpose, methodology, and runtime, please see the dedicated readme file.

- 📓 **[View Notebook Descriptions →](notebooks/README.md)**

---
## Key Findings

This research reveals two core trade-offs introduced by structured width pruning of GLU-MLP layers.

### 1. Capability Trade-offs (Accuracy & Reasoning)

Pruning creates a clear dichotomy between two classes of model capabilities:

- **"Fragile" Capabilities (Degrade):** These are tasks that rely heavily on distributed knowledge stored in the MLP layers.
  - **Degradation:** Performance on benchmarks like **MMLU** (knowledge), **GSM8K** (math reasoning), and perplexity metrics (**WikiText**, **Lambada**) consistently degrades as pruning intensity increases.
  - **Most Fragile Task:** `gsm8k` is catastrophically affected, with performance collapsing even at moderate pruning levels.

- **"Robust" Capabilities (Improve):** These are tasks that appear to rely more on core algorithmic reasoning pathways that are refined, not eroded, by pruning.
  - **Improvement:** Performance on benchmarks like **IFEval** (instruction following), **MUSR** (multi-step reasoning), and **TruthfulQA** (truthfulness) is either stable or *improves significantly* with pruning.
  - **Peak Improvement:** `IFEval` performance on the 1B model peaks at a **+75% improvement** over baseline at 30% pruning.

This trade-off suggests that pruning acts as a form of regularization, sacrificing rote knowledge for enhanced performance on tasks requiring literal instruction adherence.

### 2. The Deployment Dilemma (Performance & Energy)

Pruning introduces a second trade-off related to inference performance, creating a dilemma for deployment:

- **The Win (Batch Throughput & Efficiency):** For offline or batch processing, pruning is highly beneficial.
  - **Throughput:** Batch throughput (tokens/sec) **improves** with more aggressive pruning, as the smaller model size allows for faster processing.
  - **Energy:** Energy efficiency (Joules/token) **improves significantly**, with up to a **~20% reduction** in energy consumption at high pruning levels.

- **The Cost (Interactive Latency):** For interactive, user-facing applications, pruning has a severe negative impact.
  - **Latency:** Time To First Token (TTFT) **worsens dramatically** with pruning, increasing by **+50-90%** at higher pruning levels.
  - **The Bottleneck:** This latency cost is isolated to the **prefill phase**. The token generation speed *after* the first token remains almost completely unaffected.

This dilemma means that the optimal pruning level depends entirely on the deployment scenario. Models intended for batch processing can be aggressively pruned to save costs, while models for interactive chatbots must remain largely unpruned to ensure a responsive user experience.

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
- [x] **Llama-3.2-1B-Instruct** - Completed ✅
  - 4 pruning levels (0-60%)
  - 7 benchmarks evaluated
- [x] **Llama-3.2-3B-Instruct** - Completed ✅
   - 4 pruning levels (0-60%)
   - 7 benchmarks evaluated
