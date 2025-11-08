# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating GLU (Gated Linear Unit) expansion ratio pruning in Llama-3.2 models. The project evaluates how reducing MLP layer expansion ratios affects model performance, exploring trade-offs between efficiency and capability using the OptiPFair pruning library.

## Technology Stack

- **Python** with PyTorch and Transformers
- **OptiPFair** - Custom pruning library for on-the-fly model modification
- **LM Evaluation Harness** - Standardized benchmarking framework
- **CodeCarbon** - Energy consumption tracking
- **Google Colab** - Primary development environment with GPU support

## Development Workflow

### Running Experiments

The project uses a Jupyter notebook-based workflow designed for Google Colab:

```python
# Typical notebook execution pattern
from utils import EXPERIMENT_CONFIG, run_evaluation, setup_environment
setup_environment()  # GPU detection, memory management
results = run_evaluation(model_name, benchmark_suite)
```

### Key Commands

No formal build system - notebooks are executed cell-by-cell. Key operations:

```python
# Model evaluation with checkpointing
python -c "from utils import run_evaluation; run_evaluation('meta-llama/Llama-3.2-1B', 'BENCHMARKS_BASE')"

# Carbon profiling
python -c "from utils import run_carbon_analysis; run_carbon_analysis(model_name, task)"

# Results consolidation  
python -c "from utils import consolidate_results; consolidate_results('results/')"
```

## Architecture Overview

### Core Configuration (`utils.py`)

- **EXPERIMENT_CONFIG**: Defines all model variants and pruning configurations
- **BENCHMARKS_BASE/CARBON**: Comprehensive evaluation suites (13 tasks including MMLU, HellaSwag, TruthfulQA)
- **Hardware abstraction**: Automatic GPU/CPU detection and memory management for Colab environments

### Model Management

- **On-the-fly pruning**: Uses OptiPFair MAW (Magnitude and Weight) method
- **No pre-pruned models**: Models are pruned during evaluation runtime
- **Expansion ratio mapping**: 4.0x (baseline) → 2.4x (optimal) → 1.6x (aggressive)
- **Multiple model variants**: 1B/3B base and instruct models

### Evaluation Framework

- **Checkpoint/resume system**: Handles Colab disconnections gracefully
- **LM Evaluation Harness integration**: Standardized benchmarking with `lm-eval`
- **Carbon profiling**: Energy measurement with idle power correction
- **Memory management**: Automatic cleanup between model evaluations

### Results Pipeline

- **Individual run storage**: `results/individual_files/`
- **Consolidated analysis**: CSV/JSON exports in `results/`
- **Performance tracking**: Task-specific degradation analysis
- **Energy efficiency**: Joules per token measurements

## Key Research Parameters

### Model Configurations
- **Llama-3.2-1B**: Base and Instruct variants
- **Llama-3.2-3B**: Base and Instruct variants
- **Pruning levels**: 0%, 10%, 20%, 30%, 40%, 50%, 60%

### Evaluation Tasks
- **Knowledge**: MMLU, ARC-Easy/Challenge, OpenBookQA
- **Reasoning**: HellaSwag, PIQA, Winogrande
- **Language**: TruthfulQA, GLUE tasks
- **Instruction Following**: IFEval (instruct models only)

### Architectural Patterns

1. **Configuration-driven experiments**: All parameters centralized in `utils.py`
2. **Fault-tolerant execution**: Built-in checkpoint/resume for long-running evaluations
3. **Modular utility design**: Reusable functions across different notebook experiments
4. **Resource-aware operation**: Automatic memory management for constrained environments

## Important Notes

- **No formal dependency management**: Install requirements manually in notebook cells
- **Colab-optimized**: Designed for Google Colab Pro GPU environments
- **OptiPFair dependency**: Requires author's custom pruning library
- **HuggingFace Hub integration**: Models loaded directly from hub, results can be pushed back