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
Repository: https://github.com/oopere/glu-pruning
Paper: "Exploring GLU Expansion Ratios: Structured Pruning in Llama-3.2 Models"
"""

try:
    import lm_eval
    import transformers
    import optipfair
    # Additional imports for carbon profiling
    import time
    import numpy as np
    from datetime import datetime
except ImportError as e:
    raise ImportError(
        f"Missing required library: {e.name}\n"
        "Install all dependencies with:\n"
        "  pip install optipfair lm-eval transformers torch langdetect"
    )

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG = [
    # -------------------------------------------------------------------------
    # Llama-3.2-1B Experiments (6 models)
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 10,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-10pct",
        "is_star": False,  # Recreate on-the-fly
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 20,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-20pct",
        "is_star": False,  # Recreate on-the-fly
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 30,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-30pct",
        "is_star": False,  
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 40,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-40pct",
        "is_star": True,  # ⭐ Star model (140% expansion - paper's optimal)
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 50,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-50pct",
        "is_star": False,  
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 60,
        "hf_repo_id": "oopere/Llama-3.2-1B-pruned-60pct",
        "is_star": False,
    },

    # -------------------------------------------------------------------------
    # Llama-3.2-1B Instruct Experiments (2 models)
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "pruning_pct": 10,
        "hf_repo_id": "oopere/Llama-3.2-1B-I-pruned-10pct",
        "is_star": False,  
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "pruning_pct": 40,
        "hf_repo_id": "oopere/Llama-3.2-1B-I-pruned-40pct",
        "is_star": True,  
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "pruning_pct": 60,
        "hf_repo_id": "oopere/Llama-3.2-1B-I-pruned-60pct",
        "is_star": False,
    },
    
    # -------------------------------------------------------------------------
    # Llama-3.2-3B Experiments (6 models)
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 10,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-10pct",
        "is_star": True,  # ⭐ Star model (140% expansion - paper's optimal)
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 20,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-20pct",
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 30,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-30pct",
        "is_star": False,
    },    
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 40,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-40pct",
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 50,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-50pct",
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 60,
        "hf_repo_id": "oopere/Llama-3.2-3B-pruned-60pct",
        "is_star": False,
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
    {"name": "ifeval", "num_fewshot": 0},
    {"name": "leaderboard_musr", "num_fewshot": 0},
]


# =============================================================================
# CARBON PROFILING CONFIGURATION
# =============================================================================

EXPERIMENT_CONFIG_CARBON = [
    # -------------------------------------------------------------------------
    # Llama-3.2-1B: Baseline + Star Model Only
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 0,
        "hf_repo_id": None,  # Baseline
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B",
        "pruning_pct": 40,
        "hf_repo_id": "peremartra/Llama-3.2-1B-pruned-40pct",
        "is_star": True,  # Star model (140% expansion)
    },
    
    # -------------------------------------------------------------------------
    # Llama-3.2-3B: Baseline + Star Model Only
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 0,
        "hf_repo_id": None,  # Baseline
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-3B",
        "pruning_pct": 10,
        "hf_repo_id": "peremartra/Llama-3.2-3B-pruned-10pct",
        "is_star": True,  # Star model
    },
    
    # -------------------------------------------------------------------------
    # Llama-3.2-1B-Instruct: Baseline + Star Model Only
    # -------------------------------------------------------------------------
    {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "pruning_pct": 0,
        "hf_repo_id": None,  # Baseline
        "is_star": False,
    },
    {
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "pruning_pct": 40,
        "hf_repo_id": "peremartra/Llama-3.2-1B-I-pruned-40pct",
        "is_star": True,  # Star model
    },
]

BENCHMARKS_CARBON = [
    {
        "name": "gsm8k_workload",
        "num_prompts": 100,
        "max_new_tokens": 100,
        "dataset": "gsm8k",
        "subset": "test",
        "description": "Math reasoning workload"
    },
    {
        "name": "mmlu_workload",
        "num_prompts": 100,
        "max_new_tokens": 50,
        "dataset": "mmlu",
        "subset": "test",
        "description": "Knowledge QA workload"
    },
]

# Instruct models additional benchmark (+1)
BENCHMARKS_INSTRUCT = [
    {"name": "leaderboard_musr", "num_fewshot": 0},  
    {"name": "truthfulqa_mc1", "num_fewshot": 0},
    {"name": "truthfulqa_mc2", "num_fewshot": 0},
    {"name": "lambada_openai", "num_fewshot": 0},
    {"name": "mmlu", "num_fewshot": 5},  # Standard is 5-shot for MMLU
    {"name": "gsm8k", "num_fewshot": 5},  # Chain-of-thought requires few-shot
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
    if len(set(task_fewshot_map.values())) == 1:
        # Todas las tareas usan el mismo num_fewshot
        fewshot_value = list(task_fewshot_map.values())[0]
    else:
        # Múltiples valores diferentes (no debería pasar, pero por si acaso)
        fewshot_value = 0
        
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=task_names,
        num_fewshot=fewshot_value,  # Let task configs handle this
        batch_size="auto",
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

# =============================================================================
# INTERNAL HELPER FUNCTIONS FOR CARBON PROFILING
# =============================================================================

def _get_checkpoint_dir(base_dir, model_size, mode="evaluation"):
    """
    Internal helper: Construct checkpoint directory based on mode.
    
    Args:
        base_dir: Base checkpoint directory
        model_size: "1b", "3b", "1b_instruct", etc.
        mode: "evaluation" (default) or "carbon"
    
    Returns:
        str: Full checkpoint directory path (created if doesn't exist)
    """
    import os
    
    if mode == "evaluation":
        subdir = model_size
    elif mode == "carbon":
        subdir = f"{model_size}_carbon"
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'evaluation' or 'carbon'")
    
    checkpoint_dir = os.path.join(base_dir, subdir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return checkpoint_dir


def _get_results_filename(model_size, mode="evaluation", version="latest"):
    """
    Internal helper: Construct results filename based on mode.
    
    Args:
        model_size: "1b", "3b", etc.
        mode: "evaluation" (default) or "carbon"
        version: "latest" or timestamp string
    
    Returns:
        str: Results filename
    """
    prefix = "carbon_" if mode == "carbon" else ""
    return f"{prefix}llama_{model_size}_results_{version}.csv"


def _load_workload_prompts(workload):
    """
    Internal helper: Load prompts from specified dataset.
    
    Args:
        workload (dict): Workload specification with keys:
            - dataset: "gsm8k", "mmlu", etc.
            - subset: "test", "train", etc.
            - num_prompts: Number of prompts to load
    
    Returns:
        list[str]: List of text prompts
    """
    from datasets import load_dataset
    
    dataset_name = workload["dataset"]
    num_prompts = workload["num_prompts"]
    subset = workload.get("subset", "test")
    
    try:
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=subset)
            prompts = [item["question"] for item in dataset.select(range(min(num_prompts, len(dataset))))]
        
        elif dataset_name == "mmlu":
            # Use a specific MMLU subset (e.g., "abstract_algebra") or aggregate
            dataset = load_dataset("cais/mmlu", "all", split=subset)
            prompts = [item["question"] for item in dataset.select(range(min(num_prompts, len(dataset))))]
        
        else:
            # Fallback: generic prompts
            print(f"⚠️ Dataset {dataset_name} not implemented, using generic prompts")
            prompts = [
                f"Solve this problem: What is {i} + {i+1}?" 
                for i in range(num_prompts)
            ]
        
        return prompts
    
    except Exception as e:
        print(f"⚠️ Error loading dataset {dataset_name}: {e}")
        print(f"   Using fallback generic prompts")
        # Fallback: generic prompts
        return [f"Sample prompt {i+1} for {dataset_name}" for i in range(num_prompts)]


def _measure_inference_performance(model, tokenizer, prompts, max_new_tokens, device="cuda"):
    """
    Measure inference performance with warm-up period.
    
    The first 5 prompts are used for GPU warm-up and excluded from metrics
    to avoid CUDA kernel compilation overhead.
    """
    WARMUP_PROMPTS = 5
    
    ttft_times = []
    total_tokens = 0
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - gen_start
        
        # Only count metrics after warm-up period
        if i >= WARMUP_PROMPTS:
            ttft_times.append(gen_time)
            total_tokens += outputs.shape[1]
    
    total_time = time.time() - start_time
    
    return {
        "avg_ttft_ms": float(np.mean(ttft_times) * 1000),
        "std_ttft_ms": float(np.std(ttft_times) * 1000),
        "min_ttft_ms": float(np.min(ttft_times) * 1000),
        "max_ttft_ms": float(np.max(ttft_times) * 1000),
        "throughput_tokens_per_sec": float(total_tokens / total_time),
        "total_time_sec": float(total_time),
        "total_tokens": int(total_tokens),
        "avg_tokens_per_prompt": float(total_tokens / len(ttft_times)),  # ← Usar len(ttft_times)
        "num_measured_prompts": len(ttft_times),  # ← Nuevo: explícito
        "num_warmup_prompts": WARMUP_PROMPTS      # ← Nuevo: documentar
    }


def _get_memory_stats(model, device="cuda"):
    """
    Internal helper: Get memory usage statistics.
    
    Args:
        model: PyTorch model
        device (str): Device placement
    
    Returns:
        dict: Memory statistics in GB
    """
    stats = {}
    
    if device == "cuda" and torch.cuda.is_available():
        stats["memory_allocated_gb"] = float(torch.cuda.memory_allocated() / (1024**3))
        stats["memory_reserved_gb"] = float(torch.cuda.memory_reserved() / (1024**3))
        stats["max_memory_allocated_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
    
    # Model size (works for both CPU and CUDA)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    stats["model_size_gb"] = float(model_size_bytes / (1024**3))
    
    return stats

def run_robust_evaluation(model, tokenizer, tasks, checkpoint_path, model_name=None):
    """
    Run evaluation with checkpoint/resume support for Colab disconnections.
    
    This function saves progress after each benchmark, allowing recovery from
    interruptions. Checkpoint files are stored as JSON with task completion status.
    
    Args:
        model: PyTorch model object to evaluate
        tokenizer: Tokenizer object for the model
        tasks (list): List of task dicts with format:
                     [{"name": "wikitext", "num_fewshot": 0}, ...]
        checkpoint_path (str): Path to checkpoint JSON file
                              (e.g., "/content/drive/MyDrive/glu_pruning/llama_1b_20pct.json")
        model_name (str, optional): Human-readable model name for logging
        
    Returns:
        dict: Complete results with all benchmark metrics
        
    Example:
        >>> results = run_robust_evaluation(
        ...     model, tokenizer,
        ...     tasks=BENCHMARKS_BASE,
        ...     checkpoint_path="/content/drive/MyDrive/checkpoints/model.json"
        ... )
        >>> # If interrupted, re-run the same command - it will resume
    """
    import json
    import os
    from datetime import datetime
    from pathlib import Path
    
    # Extract model name for metadata
    if model_name is None:
        model_name = getattr(model.config, '_name_or_path', 'unknown')
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Parse tasks to get task names
    task_names = [t["name"] if isinstance(t, dict) else t for t in tasks]
    
    # Load or create checkpoint
    if os.path.exists(checkpoint_path):
        print(f"📂 Found existing checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Validate checkpoint structure
        if "results" not in checkpoint or "pending_tasks" not in checkpoint:
            print("⚠️  Invalid checkpoint structure. Starting fresh.")
            checkpoint = _create_new_checkpoint(model_name, task_names)
        else:
            print(f"✅ Loaded checkpoint. Completed: {len(checkpoint['results'])}/{len(task_names)} tasks")
            print(f"   Pending: {checkpoint['pending_tasks']}")
            if checkpoint.get('failed_tasks'):
                print(f"   ⚠️  Previously failed: {checkpoint['failed_tasks']}")
    else:
        print(f"🆕 Creating new checkpoint: {checkpoint_path}")
        checkpoint = _create_new_checkpoint(model_name, task_names)
    
    # Identify tasks to run (pending + failed to retry)
    completed_tasks = set(checkpoint["results"].keys())
    tasks_to_run = [t for t in tasks if (t["name"] if isinstance(t, dict) else t) not in completed_tasks]
    
    if not tasks_to_run:
        print("🎉 All tasks already completed!")
        return checkpoint["results"]
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting evaluation: {len(tasks_to_run)} tasks remaining")
    print(f"{'='*70}\n")
    
    # Run each pending task
    for i, task in enumerate(tasks_to_run, 1):
        task_name = task["name"] if isinstance(task, dict) else task
        
        print(f"\n[{i}/{len(tasks_to_run)}] Evaluating: {task_name}")
        print(f"{'─'*70}")
        
        try:
            # Run evaluation for single task
            result = model_evaluation(
                model, tokenizer, 
                tasks=[task],
                limit=None
            )
            
            # Store result in checkpoint
            checkpoint["results"][task_name] = result[task_name]
            checkpoint["pending_tasks"].remove(task_name)
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Remove from failed tasks if it was there
            if task_name in checkpoint.get("failed_tasks", []):
                checkpoint["failed_tasks"].remove(task_name)
            
            # Save checkpoint after each task
            _save_checkpoint(checkpoint_path, checkpoint)
            
            print(f"✅ {task_name} completed and saved to checkpoint")
            print(f"   Results: {result[task_name]}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {task_name} FAILED: {error_msg}")
            
            # Track failed task but continue
            if "failed_tasks" not in checkpoint:
                checkpoint["failed_tasks"] = []
            if task_name not in checkpoint["failed_tasks"]:
                checkpoint["failed_tasks"].append(task_name)
            
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            _save_checkpoint(checkpoint_path, checkpoint)
            
            print(f"⚠️  Continuing with next task...")
            continue
    
    # Mark as completed if all tasks done
    if not checkpoint["pending_tasks"]:
        checkpoint["metadata"]["completed"] = True
        checkpoint["metadata"]["completed_at"] = datetime.now().isoformat()
        _save_checkpoint(checkpoint_path, checkpoint)
        
        print(f"\n{'='*70}")
        print("🎉 ALL TASKS COMPLETED!")
        if checkpoint.get("failed_tasks"):
            print(f"⚠️  Some tasks failed: {checkpoint['failed_tasks']}")
        print(f"{'='*70}\n")
    
    return checkpoint["results"]


def _create_new_checkpoint(model_name, task_names):
    """Create a new checkpoint structure."""
    from datetime import datetime
    return {
        "metadata": {
            "model_name": model_name,
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed": False
        },
        "results": {},
        "pending_tasks": task_names.copy(),
        "failed_tasks": []
    }


def _save_checkpoint(checkpoint_path, checkpoint):
    """Save checkpoint to file with error handling."""
    import json
    import shutil
    from pathlib import Path
    
    try:
        # Write to temporary file first (atomic write)
        temp_path = f"{checkpoint_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Move to final location
        shutil.move(temp_path, checkpoint_path)
        
        # Sync with Google Drive if path contains 'drive'
        if 'drive' in checkpoint_path.lower():
            try:
                # Force sync by touching the file
                Path(checkpoint_path).touch()
            except:
                pass  # Drive sync is best-effort
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint: {e}")
        # Don't crash the evaluation if checkpoint save fails


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_gpu_cache():
    """
    Clear GPU memory cache and run garbage collection.
    
    Essential for Colab environments to prevent OOM errors when
    switching between models or after pruning operations.
    """
    import gc
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    if torch.cuda.is_available():
        print(f"🧹 GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def get_model_stats(model):
    """
    Calculate model statistics: total parameters, trainable parameters, size.
    
    Args:
        model: PyTorch model object
        
    Returns:
        dict: Statistics including parameter counts and model size
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "size_mb": size_mb,
        "size_gb": size_mb / 1024
    }


def load_or_create_model(config_entry, device="auto"):
    """
    Load model from HF Hub (if star) or create via pruning (if not star).
    
    Args:
        config_entry (dict): Entry from EXPERIMENT_CONFIG
        device (str): Device placement ("auto", "cuda", "cpu")
        
    Returns:
        tuple: (model, tokenizer, stats_dict)
        
    Example:
        >>> config = EXPERIMENT_CONFIG[1]  # 1B-40% (star)
        >>> model, tokenizer, stats = load_or_create_model(config)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from optipfair import prune_model
    
    base_model_id = config_entry["base_model"]
    hf_repo_id = config_entry["hf_repo_id"]
    is_star = config_entry["is_star"]
    pruning_pct = config_entry["pruning_pct"]
    
    print(f"\n{'='*70}")
    print(f"Loading model: {hf_repo_id}")
    print(f"  Base: {base_model_id}")
    print(f"  Pruning: {pruning_pct}%")
    print(f"  Star model: {'⭐ Yes' if is_star else 'No (on-the-fly)'}")
    print(f"{'='*70}\n")
    
    # Load tokenizer (always from base model)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_star:
        # Try loading from HF Hub first
        try:
            print(f"📥 Attempting to load from HF Hub: {hf_repo_id}")
            model = AutoModelForCausalLM.from_pretrained(
                hf_repo_id,
                #torch_dtype=torch.float16, #L4
                torch_dtype=torch.bfloat16, #A100
                device_map=device
            )
            print(f"✅ Loaded from HF Hub")
            stats = {"source": "hf_hub", **get_model_stats(model)}
            return model, tokenizer, stats
            
        except Exception as e:
            print(f"⚠️  HF Hub load failed: {e}")
            print(f"   Falling back to on-the-fly pruning...")
    
    # Create via pruning (fallback or non-star)
    print(f"🔧 Creating model via on-the-fly pruning...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        #torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    print(f"✂️  Pruning with MAW method ({pruning_pct}%)...")
    pruned_model, prune_stats = prune_model(
        model=base_model,
        pruning_type="MLP_GLU",
        neuron_selection_method="MAW",
        pruning_percentage=pruning_pct,
        show_progress=True,
        return_stats=True
    )
    
    print(f"✅ Model created")
    print(f"   Original params: {prune_stats['original_parameters']:,}")
    print(f"   Pruned params: {prune_stats['pruned_parameters']:,}")
    print(f"   Reduction: {prune_stats['percentage_reduction']:.2f}%")
    
    stats = {
        "source": "on_the_fly_pruning",
        "pruning_stats": prune_stats,
        **get_model_stats(pruned_model)
    }
    
    return pruned_model, tokenizer, stats


def format_results_table(results_dict):
    """
    Format evaluation results as a pretty table for display.
    
    Args:
        results_dict (dict): Results from run_robust_evaluation()
        
    Returns:
        str: Formatted table string
    """
    import pandas as pd
    
    # Flatten nested metrics
    rows = []
    for task_name, metrics in results_dict.items():
        row = {"task": task_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_string(index=False)

# =============================================================================
# CARBON PROFILING MAIN FUNCTION
# =============================================================================

def run_carbon_profiling(
    model,
    tokenizer,
    workloads,
    checkpoint_path,
    model_name=None,
    device="cuda"
):
    """
    Run carbon and performance profiling on a model (parallel to run_robust_evaluation).
    
    Measures:
        - Energy consumption (CodeCarbon)
        - Throughput (tokens/second)
        - Latency (Time To First Token)
        - Memory footprint
    
    Uses the same checkpoint/resume system as run_robust_evaluation for reliability.
    
    Args:
        model: PyTorch model object
        tokenizer: Tokenizer for the model
        workloads (list): List of workload dicts from BENCHMARKS_CARBON
        checkpoint_path (str): Path to checkpoint JSON file
        model_name (str, optional): Human-readable model name
        device (str): Device placement ("cuda" or "cpu")
    
    Returns:
        dict: Complete profiling results with all metrics
    
    Example:
        >>> results = run_carbon_profiling(
        ...     model, tokenizer,
        ...     workloads=BENCHMARKS_CARBON,
        ...     checkpoint_path="/path/carbon_baseline.json",
        ...     model_name="Llama-3.2-1B-baseline"
        ... )
    """
    import json
    import os
    from codecarbon import EmissionsTracker
    
    # Extract model name
    if model_name is None:
        model_name = getattr(model.config, '_name_or_path', 'unknown')
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load or create checkpoint
    if os.path.exists(checkpoint_path):
        print(f"📂 Found existing checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        if "results" not in checkpoint:
            checkpoint["results"] = {}
        if "metadata" not in checkpoint:
            checkpoint["metadata"] = {
                "model_name": model_name,
                "started_at": datetime.now().isoformat(),
                "mode": "carbon_profiling"
            }
    else:
        print(f"🆕 Creating new checkpoint: {checkpoint_path}")
        checkpoint = {
            "metadata": {
                "model_name": model_name,
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "mode": "carbon_profiling"
            },
            "results": {},
            "completed_workloads": [],
            "failed_workloads": []
        }
    
    completed = checkpoint.get("completed_workloads", [])
    failed = checkpoint.get("failed_workloads", [])
    
    # Determine pending workloads
    pending = [w for w in workloads 
               if w["name"] not in completed and w["name"] not in failed]
    
    print(f"✅ Loaded checkpoint. Completed: {len(completed)}/{len(workloads)} workloads")
    if failed:
        print(f"⚠️ Previously failed: {failed}")
    
    if not pending:
        print("🎉 All workloads completed!")
        return checkpoint["results"]
    
    print(f"\n🚀 Starting profiling: {len(pending)} workloads remaining")
    print("="*70 + "\n")
    
    # Process each workload
    for i, workload in enumerate(pending, 1):
        workload_name = workload["name"]
        
        print(f"[{i}/{len(pending)}] Profiling: {workload_name}")
        print("-"*70)
        print("   Clearing GPU cache...")
        clear_gpu_cache()
        
        try:
            # Initialize CodeCarbon tracker
            tracker = EmissionsTracker(
                project_name=f"glu_pruning_{model_name}",
                measure_power_secs=1,
                save_to_file=False,  # We'll manage output ourselves
                log_level="warning"  # Reduce verbosity
            )
            tracker.start()
            
            # Load prompts
            print(f"   Loading {workload['num_prompts']} prompts from {workload['dataset']}...")
            prompts = _load_workload_prompts(workload)
            
            # Measure inference performance
            print(f"   Running inference...")
            perf_metrics = _measure_inference_performance(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=workload["max_new_tokens"],
                device=device
            )
            
            # Stop tracker and get emissions
            emissions: float = tracker.stop()
            
            # Get memory stats
            memory_stats = _get_memory_stats(model, device)
            
            # Consolidate results
            result = {
                **perf_metrics,
                **memory_stats,
                "energy_kwh": float(emissions) if emissions else 0.0,
                "num_prompts": len(prompts),
                "max_new_tokens": workload["max_new_tokens"],
                "workload_description": workload.get("description", "")
            }
            
            # Save to checkpoint
            checkpoint["results"][workload_name] = result
            checkpoint["completed_workloads"] = checkpoint.get("completed_workloads", []) + [workload_name]
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Persist checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Print summary
            print(f"✅ {workload_name} completed")
            print(f"   Energy: {result['energy_kwh']:.6f} kWh")
            print(f"   Throughput: {result['throughput_tokens_per_sec']:.2f} tok/s")
            print(f"   Avg TTFT: {result['avg_ttft_ms']:.2f} ms")
            print(f"   Memory: {result['model_size_gb']:.2f} GB\n")
            
        except Exception as e:
            print(f"❌ {workload_name} FAILED: {str(e)}")
            checkpoint["failed_workloads"] = checkpoint.get("failed_workloads", []) + [workload_name]
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Save checkpoint even on failure
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            print("⚠️ Continuing with next workload...\n")
            continue
    
    print("="*70)
    print("🎉 ALL WORKLOADS COMPLETED!")
    if checkpoint.get("failed_workloads"):
        print(f"⚠️ Some workloads failed: {checkpoint['failed_workloads']}")
    print("="*70 + "\n")
    
    return checkpoint["results"]
