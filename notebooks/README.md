# Notebooks

This directory contains the Jupyter notebooks used for the experiments in this repository.

---

## 00 - Foundational Notebooks

These notebooks establish the baseline architectural details and validate the chosen pruning methodology.

### 00 - Expansion Rate Documentation
**File:** `00_Expansion_Rate.ipynb`
**Purpose:** To document and verify the exact architectural changes (expansion rates, parameter counts) resulting from pruning Llama-3.2 models.
**What it does:**
- Systematically applies predefined pruning percentages (10% to 60%) to 1B, 3B, and 1B-Instruct Llama-3.2 models.
- Uses `optipfair` to inspect the resulting model architecture after each pruning operation.
- Calculates the precise expansion ratio, total parameters, and parameter reduction for each configuration.
- Saves a comprehensive report to `expansion_rates.json` and a summary to `expansion_rates_summary.csv`.
**Conclusion:** This notebook serves as a foundational step, ensuring that all subsequent experiments are based on accurate and reproducible model architectures. It provides the ground-truth mapping between a desired pruning percentage and the resulting model geometry.
**Runtime:** ~30-45 minutes on a CPU (no GPU required).

### 00 - Neuron Selection Method Comparison
**File:** `00_Neuron_Selection_Method_Comparison.ipynb`
**Purpose:** To validate the choice of "Maximum Absolute Weight" (MAW) as the optimal neuron selection method for pruning GLU layers in Llama-style models.
**What it does:**
- Compares three different neuron selection methods: `MAW`, `L2`, and `Random`.
- Applies each method to prune the Llama-3.2-1B model at a fixed 40% pruning level.
- Evaluates the resulting models on a small, representative set of benchmarks (`wikitext`, `mmlu`, `gsm8k`).
- Measures the performance degradation of each method relative to the baseline model.
**Conclusion:** The results from this notebook confirm that the MAW method consistently outperforms L2 and Random selection, leading to the least performance degradation. This provides the empirical justification for using MAW in all subsequent experiments.
**Runtime:** ~30 minutes on a Google Colab T4 GPU.

---

## 02 - Capability Evaluation Notebooks

These notebooks run comprehensive benchmark suites to evaluate the impact of pruning on model capabilities (accuracy, reasoning, etc.).

### 02 - Evaluate Llama-3.2-1B (Base)
**File:** `02_Evaluate_1B.ipynb`
**Purpose:** To perform a comprehensive capability evaluation of the base Llama-3.2-1B model across multiple pruning levels.
**What it does:**
- Evaluates the baseline Llama-3.2-1B model and all its pruned versions (10% to 60%).
- Runs a broad suite of 13 general-purpose benchmarks (`BENCHMARKS_BASE`), including tasks for reasoning, knowledge, and language perplexity like `arc_challenge`, `mmlu`, `gsm8k`, and `wikitext`.
- Uses the same robust evaluation process as the other "02" notebooks, saving results to `results/llama_1b_results_latest.csv` and a complete JSON file.
- Analyzes the trade-offs between performance degradation and pruning intensity to identify the optimal configuration (the "star model").
**Conclusion:** This is the core evaluation notebook for the base 1B model, providing the primary data on how pruning affects its general knowledge and reasoning abilities, as opposed to specific instruction-following skills.
**Runtime:** ~4-5 hours on a Google Colab L4 GPU.

### 02 - Evaluate Llama-3.2-1B-Instruct
**File:** `02_Evaluate_1B_Instruct.ipynb`
**Purpose:** To evaluate the impact of GLU pruning on the instruction-following and reasoning capabilities of the Llama-3.2-1B-Instruct model.
**What it does:**
- Evaluates the baseline Llama-3.2-1B-Instruct model and several pruned versions (e.g., 10%, 40%, 60%).
- Runs a curated suite of 7 benchmarks (`BENCHMARKS_INSTRUCT`) tailored for instruction-following, including `ifeval`, `musr`, `mmlu`, and `gsm8k`.
- Uses the `run_robust_evaluation` utility, which leverages `lm-evaluation-harness` with multiple runs for statistical significance.
- Saves detailed evaluation results to `results/llama_1b_I_results_latest.csv` and `results/llama_1b_I_complete_results_latest.json`.
- Includes a decision matrix to determine which pruned models are candidates for uploading to Hugging Face Hub based on performance trade-offs.
**Conclusion:** This notebook tests the hypothesis that pruning can improve instruction-following (`ifeval`) at the cost of knowledge-based tasks (`mmlu`). It is a core component of the research, generating the primary capability metrics for the instruct-tuned model.
**Runtime:** ~4-5 hours on a Google Colab L4 GPU.

### 02 - Evaluate Llama-3.2-3B (Base)
**File:** `02_Evaluate_3B.ipynb`
**Purpose:** To evaluate the effects of GLU pruning on the larger Llama-3.2-3B base model and compare the results against the 1B model.
**What it does:**
- Evaluates the baseline Llama-3.2-3B model and its pruned versions (10% to 60%).
- Runs the same broad suite of 13 general-purpose benchmarks (`BENCHMARKS_BASE`) used for the 1B model, allowing for direct comparison.
- Follows the standard robust evaluation process, saving results to `results/llama_3b_results_latest.csv` and a complete JSON file.
- Identifies the "star model" for the 3B architecture (10% pruning) and analyzes performance degradation patterns to see how they scale with model size.
**Conclusion:** This notebook is crucial for understanding the scalability of the pruning effects. By comparing its results to the 1B model, the research can draw more general conclusions about how GLU pruning impacts Llama-style architectures of different sizes.
**Runtime:** ~4-5 hours on a Google Colab L4 GPU.

---

## 03 - Carbon & Performance Profiling Notebooks

These notebooks measure the real-world impact of pruning on inference speed and energy consumption.

### 03 - Evaluate Llama-3.2-1B (Carbon & Performance)
**File:** `03_Evaluate_1B_CARBON.ipynb`
**Purpose:** To measure the impact of GLU pruning on the inference performance and energy efficiency of the Llama-3.2-1B model.
**What it does:**
- Profiles the baseline Llama-3.2-1B model and its pruned versions (10% to 60%) on a set of generative workloads (`BENCHMARKS_CARBON`).
- Measures key efficiency metrics:
    - **Latency:** Time To First Token (TTFT).
    - **Throughput:** Tokens per second.
    - **Energy:** Joules per token, using the `codecarbon` library.
- Employs a robust methodology, including:
    - Calibrating and subtracting idle GPU power for accurate net energy measurement.
    - Performing multiple runs with fixed seeds for each model to ensure stable and reliable metrics.
- Saves the detailed, run-by-run performance and energy data to `results/llama_1b_carbon_complete_results_latest.json`.
**Conclusion:** This notebook provides the quantitative data to support claims about the efficiency gains from pruning. It moves beyond theoretical parameter reduction to measure real-world improvements in speed and energy consumption, which is a critical part of the research's contribution.
**Runtime:** ~1-2 hours on a Google Colab L4 GPU.

### 03 - Evaluate Llama-3.2-3B (Carbon & Performance)
**File:** `03_Evaluate_3B_CARBON.ipynb`
**Purpose:** To measure the inference performance and energy efficiency of the larger Llama-3.2-3B model and compare its efficiency gains with the 1B model.
**What it does:**
- Profiles the baseline Llama-3.2-3B model and its pruned versions (10% to 60%) on the same set of generative workloads (`BENCHMARKS_CARBON`) used for the 1B model.
- Measures Latency (TTFT), Throughput (tokens/sec), and Energy (Joules/token).
- Follows the same robust methodology as the 1B carbon notebook, including idle power calibration and multiple runs per model.
- Saves the comprehensive performance and energy results to `results/llama_3b_carbon_complete_results_latest.json`.
**Conclusion:** This notebook provides the data to analyze how the efficiency benefits of pruning scale with model size. By comparing these results to the 1B model's carbon metrics, the research can make broader claims about the relationship between model scale, pruning, and computational efficiency.
**Runtime:** ~1-2 hours on a Google Colab L4 GPU.

---

## 04 - Graphics & Analysis Notebooks

These notebooks are for post-processing and visualizing the results generated by the evaluation notebooks. They do not run experiments themselves.

### 04 - 1B Graphics and Analysis
**File:** `04_1B_Graphics.ipynb`
**Purpose:** To visualize and analyze the benchmark results for the Llama-3.2-1B model family, focusing on the trade-offs between different capabilities.
**What it does:**
- Loads the final, consolidated evaluation data from `results/llama_1b_complete_results_latest.json`.
- Preprocesses and normalizes the benchmark scores relative to the baseline model to allow for direct comparison across different metrics.
- Groups benchmarks into "Fragile" (knowledge-intensive) and "Robust" (reasoning-based) categories.
- Generates a series of plots to visualize the core hypothesis:
    - The degradation of fragile capabilities like `gsm8k` and perplexity.
    - The improvement of robust capabilities like `ifeval` and `musr`.
    - An aggregate plot that clearly shows the divergent paths of the two capability types as pruning increases.
**Conclusion:** This notebook is not for running experiments but for interpreting their results. It produces the key visualizations that support the paper's main arguments about the "capability trade-off" induced by GLU pruning.
**Runtime:** ~5 minutes on a CPU.

### 04 - 1B Graphics (Carbon & Performance)
**File:** `04_1B_Graphics_Carbon.ipynb`
**Purpose:** To visualize and analyze the inference performance and energy consumption results for the Llama-3.2-1B model family.
**What it does:**
- Loads the final carbon and performance data from `results/llama_1b_carbon_complete_results_latest.json`.
- Creates summary and detailed DataFrames to analyze the results.
- Generates a series of plots to visualize the "Deployment Dilemma":
    - The clear improvement in energy efficiency (Joules/Token).
    - The significant worsening of interactive latency (Time To First Token).
    - A dual-axis plot directly comparing the trade-off between batch throughput (which improves) and TTFT (which degrades).
    - Plots that break down the results by individual task and batch size to confirm the consistency of the observed trends.
**Conclusion:** This notebook produces the key visualizations that explain the practical consequences of pruning for model deployment. It clearly illustrates that while pruning makes models more energy-efficient and faster for batch processing, it comes at a direct cost to interactive user experience.
**Runtime:** ~1 minute on a CPU.

### 04 - 3B Graphics and Analysis
**File:** `04_3B_Graphics.ipynb`
**Purpose:** To visualize and analyze the benchmark results for the larger Llama-3.2-3B model family, comparing its capability trade-offs to the 1B model.
**What it does:**
- Loads the final evaluation data for the 3B model from `results/llama_3b_complete_results_latest.json`.
- Preprocesses and normalizes the benchmark scores relative to the 3B baseline model.
- Groups benchmarks into the same "Fragile" and "Robust" categories used for the 1B analysis.
- Generates a series of plots analogous to the 1B graphics notebook, visualizing:
    - The degradation of knowledge-intensive tasks.
    - The improvement or stability of reasoning-based tasks.
    - The aggregate "capability trade-off" plot for the 3B model.
**Conclusion:** This notebook is essential for understanding how the effects of GLU pruning scale with model size. By creating visualizations for the 3B model that are directly comparable to the 1B model's graphs, it allows the research to draw more general conclusions about the pruning methodology.
**Runtime:** ~5 minutes on a CPU.

### 04 - 3B Graphics (Carbon & Performance)
**File:** `04_3B_Graphics_Carbon.ipynb`
**Purpose:** To visualize and analyze the inference performance and energy consumption results for the larger Llama-3.2-3B model family.
**What it does:**
- Loads the final carbon and performance data for the 3B model from `results/llama_3b_carbon_complete_results_latest.json`.
- Creates summary and detailed DataFrames to analyze the results.
- Generates a series of plots to visualize the "Deployment Dilemma" for the 3B model, including:
    - The improvement in energy efficiency (Joules/Token).
    - The degradation of interactive latency (Time To First Token).
    - The dual-axis plot comparing batch throughput and TTFT.
    - Breakdowns by task and batch size.
**Conclusion:** This notebook produces the visualizations necessary to compare the deployment trade-offs of the 3B model with the 1B model, allowing the research to analyze how these efficiency metrics scale with model size.
**Runtime:** ~1 minute on a CPU.