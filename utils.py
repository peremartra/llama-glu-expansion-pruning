"""
GLU Pruning Width Pruning Research - Utility Functions
========================================================

This module provides core utilities for the GLU expansion ratio pruning experiments
on Llama-3.2 models (1B and 3B variants).

Key Responsibilities:
--------------------
1. Experiment configuration management (models, pruning levels, HF repos)
2. Robust evaluation orchestration with checkpoint/resume support
3. LM Evaluation Harness integration with error handling
4. Model loading (HF Hub or on-the-fly pruning with OptiPFair)
5. GPU memory management and cleanup utilities
6. Results formatting and export to CSV/JSON

Target Environment:
------------------
- Google Colab (T4/V100 GPUs)
- Supports disconnection recovery via checkpoint system
- Designed for ~20 hours of total compute time across 4 models

Usage:
------
    from utils import run_robust_evaluation, load_or_create_model, EXPERIMENT_CONFIG
    
    model, tokenizer = load_or_create_model(config_entry)
    results = run_robust_evaluation(
        model, tokenizer, 
        tasks=BENCHMARKS_BASE, 
        checkpoint_path="./checkpoints/llama_1b_20pct.json"
    )

Author: Pere Martra
Repository: https://github.com/peremartra/glu-pruning
Paper: "Exploring GLU Expansion Ratios: Structured Pruning in Llama-3.2 Models"
"""

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG = [
    # -------------------------------------------------------------------------
    # Llama-3.2-1B Experiments (3 models)
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 20,
        "hf_repo_id": "peremartra/Llama-3.2-1B-pruned-20pct",
        "is_star": False,  # Recreate on-the-fly
        "expansion_rate": 220,  # Target expansion: 2048 → 6554
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 40,
        "hf_repo_id": "peremartra/Llama-3.2-1B-pruned-40pct",
        "is_star": True,  # ⭐ Star model (140% expansion - paper's optimal)
        "expansion_rate": 140,  # Target expansion: 2048 → 4916
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 60,
        "hf_repo_id": "peremartra/Llama-3.2-1B-pruned-60pct",
        "is_star": False,
        "expansion_rate": 60,  # Target expansion: 2048 → 3277
    },
    
    # -------------------------------------------------------------------------
    # Llama-3.2-3B Experiments (4 models)
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 10,
        "hf_repo_id": "peremartra/Llama-3.2-3B-pruned-10pct",
        "is_star": True,  # ⭐ Star model (140% expansion - paper's optimal)
        "expansion_rate": 140,  # Target expansion: 3072 → 7373
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 20,
        "hf_repo_id": "peremartra/Llama-3.2-3B-pruned-20pct",
        "is_star": False,
        "expansion_rate": 113,  # Target expansion: 3072 → 6554
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 40,
        "hf_repo_id": "peremartra/Llama-3.2-3B-pruned-40pct",
        "is_star": False,
        "expansion_rate": 60,  # Target expansion: 3072 → 4916
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 60,
        "hf_repo_id": "peremartra/Llama-3.2-3B-pruned-60pct",
        "is_star": False,
        "expansion_rate": 7,  # Target expansion: 3072 → 3277
    },
]


# =============================================================================
# BENCHMARK CONFIGURATIONS
# =============================================================================

# Base models benchmark suite (10 benchmarks)
# Default: 0-shot unless specified otherwise
BENCHMARKS_BASE = [
    {"name": "wikitext", "num_fewshot": 0},
    {"name": "boolq", "num_fewshot": 0},
    {"name": "lambada_openai", "num_fewshot": 0},
    {"name": "mmlu", "num_fewshot": 5},  # Standard is 5-shot for MMLU
    {"name": "arc_challenge", "num_fewshot": 0},
    {"name": "hellaswag", "num_fewshot": 0},
    {"name": "winogrande", "num_fewshot": 0},
    {"name": "piqa", "num_fewshot": 0},
    {"name": "truthfulqa_mc1", "num_fewshot": 0},
    {"name": "truthfulqa_mc2", "num_fewshot": 0},
    {"name": "gsm8k", "num_fewshot": 5},  # Chain-of-thought requires few-shot
]

# Instruct models additional benchmark (+1)
BENCHMARKS_INSTRUCT = BENCHMARKS_BASE + [
    {"name": "ifeval", "num_fewshot": 0},
]


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Device detection
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default checkpoint directory (Google Drive recommended for Colab)
DEFAULT_CHECKPOINT_DIR = "/content/drive/MyDrive/glu_pruning/checkpoints"

# Library versions for reproducibility (to be filled during development)
LIBRARY_VERSIONS = {
    "optipfair": None,  # Will be populated at runtime
    "transformers": None,
    "lm-eval": None,
    "torch": None,
}

# =============================================================================
# CORE EVALUATION FUNCTIONS
# =============================================================================

def model_evaluation(model_obj, tokenizer, tasks, limit=None):
    """
    Runs lm-eval on a model and tokenizer already in memory.
    
    Args:
        model_obj: PyTorch model object to evaluate
        tokenizer: Tokenizer object for the model
        tasks (list): List of task dicts with format:
                     [{"name": "wikitext", "num_fewshot": 0}, ...]
                     OR simple list of strings: ["wikitext", "boolq"]
        limit (int, optional): Number of samples per task for quick testing
        
    Returns:
        dict: Formatted results with metrics per task
        
    Raises:
        ImportError: If lm-eval is not installed
        Exception: If evaluation fails for all tasks
        
    Example:
        >>> results = model_evaluation(
        ...     model, tokenizer, 
        ...     tasks=BENCHMARKS_BASE,
        ...     limit=100  # Quick test
        ... )
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    
    # Extract model name for logging
    model_name = getattr(model_obj.config, '_name_or_path', 'unknown')
    limit_str = f"(limit={limit})" if limit else "(full dataset)"
    
    # Parse tasks to handle both dict and string formats
    task_names = []
    task_fewshot_map = {}
    
    for task in tasks:
        if isinstance(task, dict):
            task_name = task["name"]
            task_names.append(task_name)
            task_fewshot_map[task_name] = task["num_fewshot"]
        else:
            # Backward compatibility: simple string list
            task_names.append(task)
            task_fewshot_map[task] = 0
    
    print(f"\n{'='*70}")
    print(f"Starting lm-eval on model '{model_name}'")
    print(f"Tasks: {task_names} {limit_str}")
    print(f"Few-shot config: {task_fewshot_map}")
    print(f"{'='*70}\n")
    
    # Wrap model for lm-eval
    model_wrapper = HFLM(
        pretrained=model_obj,
        tokenizer=tokenizer,
        device=str(DEVICE)
    )
    
    # Run evaluation with per-task few-shot configuration
    # Note: lm-eval handles num_fewshot per task if tasks are configured properly
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=task_names,
        num_fewshot=None,  # Let task configs handle this
        limit=limit,
        device=str(DEVICE)
    )
    
    # Format results for clean display
    formatted_results = {}
    for task_name, res in results["results"].items():
        # Extract relevant metrics based on task type
        if 'perplexity,none' in res:
            # Perplexity tasks (wikitext, lambada)
            formatted_results[task_name] = {
                'perplexity': f"{res.get('perplexity,none', 0):.2f}",
                'word_perplexity': f"{res.get('word_perplexity,none', 0):.2f}",
                'bits_per_byte': f"{res.get('bits_per_byte,none', 0):.4f}"
            }
        elif 'acc,none' in res:
            # Accuracy tasks (boolq, arc, hellaswag, etc.)
            formatted_results[task_name] = {
                'accuracy': f"{res.get('acc,none', 0):.4f}",
                'acc_norm': f"{res.get('acc_norm,none', 0):.4f}" if 'acc_norm,none' in res else "N/A"
            }
        else:
            # Fallback: store all numeric metrics
            formatted_results[task_name] = {
                k: f"{v:.4f}" for k, v in res.items() 
                if isinstance(v, (int, float))
            }
    
    return formatted_results
