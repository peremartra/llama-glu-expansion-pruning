### 5.3 Alternative Importance Metrics

During our investigation, we explored an alternative neuron importance 
calculation that incorporates the down_proj layer alongside gate_proj 
and up_proj. Preliminary experiments on Llama-3.2-1B at 40% pruning 
revealed interesting trade-offs:

The alternative method showed substantial improvements in perplexity-based 
tasks (Lambada: ~45% PPL reduction) and specific reasoning subtasks 
(MUSR Object Placements: +56%). However, it exhibited inconsistent 
behavior across different reasoning types, with significant degradation 
in other subtasks (MUSR Team Allocation: -29%). Overall performance 
across the benchmark suite remained comparable to our primary method, 
with no clear systematic advantage.

These mixed results suggest that down_proj inclusion may prioritize 
certain reasoning patterns over others, potentially reflecting different 
optimization objectives in the pruning process. We hypothesize that 
down_proj captures downstream task-specific information flow, while our 
primary method (gate_proj + up_proj) better preserves the fundamental 
gating mechanism's integrity. Further investigation with larger datasets 
and diverse architectures is needed to understand these trade-offs.

For this work, we prioritized the more stable and interpretable 
gate_proj + up_proj approach, which demonstrated consistent behavior 
across model sizes and tasks, enabling the identification of the 140% 
expansion ratio as an optimal pruning target.
