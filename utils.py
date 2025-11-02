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
