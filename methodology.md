# Methodology

This section details the comprehensive methodology used to evaluate the effects of structured pruning on Large Language Models (LLMs), specifically focusing on the Llama 3.2 architecture. Our evaluation is twofold: assessing the impact on model capabilities using standard academic benchmarks and analyzing the effects on computational performance, including latency, throughput, and energy consumption.

## 1. Models and Pruning Technique

### 1.1. Base Models

We selected three variants of the Llama 3.2 model family for our experiments:

-   **Llama-3.2-1B:** A 1-billion parameter base model.
-   **Llama-3.2-3B:** A 3-billion parameter base model, to assess scalability.
-   **Llama-3.2-1B-Instruct:** A 1-billion parameter model fine-tuned for instruction-following, used to evaluate the impact on reasoning and alignment.

All models were loaded in `bfloat16` precision for both pruning and evaluation, utilizing the `optipfair` library for model manipulation.

### 1.2. Neuron Importance and Pruning Method

Our pruning strategy targets the Gated Linear Unit (GLU) MLP layers, which are critical components of modern transformer architectures. We employed **structured width pruning**, a method that removes entire neurons (columns) from the weight matrices (`gate_proj`, `up_proj`) and corresponding rows from the `down_proj` matrix.

The selection of which neurons to prune was based on an empirical comparison of three importance metrics:

1.  **Maximum Absolute Weight (MAW):** The importance of a neuron is determined by the sum of maximum absolute values of its weights in both `gate_proj` and `up_proj` matrices.
2.  **Vector-based Output Weighting (VOW):** Measures the L2 norm of the output weight vector associated with each neuron.
3.  **Product of Norms (PON):** Computes the product of the L2 norms of a neuron's input and output weight vectors.

An initial evaluation, documented in the `00_Neuron_Selection_Method_Comparison.ipynb` notebook, revealed that both VOW and PON led to catastrophic performance degradation immediately after pruning. In contrast, **MAW** preserved model capabilities most effectively. Consequently, MAW was selected as the sole importance metric for all subsequent experiments.

Models were pruned at incremental ratios from **10% to 60%** of the neurons in the GLU-MLP layers.

## 2. Evaluation Frameworks

### 2.1. Capability Evaluation

To measure the impact of pruning on model capabilities, we used the `lm-evaluation-harness`, a standardized framework for LLM evaluation. We curated two distinct sets of benchmarks:

-   **Base Model Benchmarks (`BENCHMARKS_BASE`):** A comprehensive suite of 13 tasks designed to test a broad range of capabilities, including:
    -   **Language Modeling:** `wikitext` (perplexity), `lambada_openai`.
    -   **Reading Comprehension:** `boolq`.
    -   **Reasoning and Commonsense:** `hellaswag`, `winogrande`, `piqa`.
    -   **Knowledge-Intensive:** `mmlu` (5-shot), `gsm8k` (5-shot), `arc_challenge` (0-shot).
    -   **Safety and Truthfulness:** `truthfulqa_mc1`, `truthfulqa_mc2`.
    -   **Instruction Following:** `ifeval`.
    -   **Multi-step Reasoning:** `leaderboard_musr`.
-   **Instruction-Following Benchmarks (`BENCHMARKS_INSTRUCT`):** A focused set of 7 tasks to evaluate the instruction-following and reasoning abilities of the `1B-Instruct` model:
    -   **Instruction Following:** `ifeval`.
    -   **Multi-step Reasoning:** `leaderboard_musr`.
    -   **Safety and Truthfulness:** `truthfulqa_mc1`, `truthfulqa_mc2`.
    -   **Language Understanding:** `lambada_openai`.
    -   **Knowledge:** `mmlu` (5-shot), `gsm8k` (5-shot).

The evaluation process, executed via the `run_robust_evaluation` function in `utils.py`, involved a single run for each model-pruning configuration using the EleutherAI LM Evaluation Harness with `batch_size="auto"` and appropriate few-shot configurations as specified above.

### 2.2. Performance and Energy Profiling

To quantify the computational benefits of pruning, we conducted a separate evaluation focused on performance and energy efficiency. This was orchestrated by the `run_carbon_profiling` function and measured the following metrics:

-   **Time to First Token (TTFT):** Latency before the model generates the first token, measured in milliseconds.
-   **Throughput:** The rate of token generation (tokens/second) during inference.
-   **Energy Consumption:** Total energy in kWh and efficiency in Joules per token, using the `codecarbon` library with 1-second measurement intervals and idle power correction.
-   **Memory Usage:** Model size and GPU memory allocation in GB.

This profiling was performed on a specific set of 6 generation workloads defined in `BENCHMARKS_CARBON`:

-   **Latency Measurement (batch=1):** `hellaswag` (100 prompts, 20 tokens), `mmlu` (100 prompts, 50 tokens), `ifeval` (30 prompts, 150 tokens)
-   **Throughput Measurement (batch=8):** Same tasks with increased batch size for throughput optimization

Each carbon profiling experiment was run three times with fixed seeds (`42`, `123`, `456`) to ensure statistical reliability. An idle power calibration was performed before experiments to enable net energy calculation. The results were aggregated using mean ± standard deviation, with outlier detection (>1.0 J/token) to ensure data quality.
