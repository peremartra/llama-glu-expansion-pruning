# Model Architectures
This directory contains architectural specifications for the pruned Llama-3.2 models used in this study. The data documents the relationship between pruning percentage and resulting MLP expansion ratios across different model sizes.

## Files

- **`expansion_rates.json`**: Complete architectural details for each pruned configuration, including parameter counts, layer dimensions, and OptIFAIR pruning statistics
- **`expansion_rates_summary.csv`**: Condensed view showing key metrics for each configuration

## Expansion Ratio Convention

In this study, expansion ratio refers to the multiplicative factor by which the MLP hidden dimension expands relative to the model's base hidden dimension. Following standard transformer literature, we express this as a multiplier (e.g., 4x) rather than percentage.

**Example:** Llama-3.2-1B has `hidden_size=2048` and `intermediate_size=8192`, giving an expansion ratio of 8192/2048 = **4x**.

## Pruning Configurations

### Llama-3.2-1B Models (Base & Instruct)

| Pruning % | Expansion Ratio | Intermediate Size | Parameters (M) | Param Reduction % |
|-----------|-----------------|-------------------|----------------|-------------------|
| 0% (baseline) | 4.0x | 8192 | 1235.81 | 0% |
| 10% | 3.6x | 7373 | 1155.30 | 6.51% |
| 20% | 3.2x | 6554 | 1074.79 | 13.03% |
| 30% | 2.8x | 5735 | 994.28 | 19.54% |
| 40% | 2.4x | 4916 | 913.77 | 26.06% |
| 50% | 2.0x | 4096 | 833.16 | 32.58% |
| 60% | 1.6x | 3277 | 752.65 | 39.10% |

**Architecture details:**
- `hidden_size`: 2048
- `num_hidden_layers`: 16
- Baseline `intermediate_size`: 8192

### Llama-3.2-3B Models (Base & Instruct)

| Pruning % | Expansion Ratio | Intermediate Size | Parameters (M) | Param Reduction % |
|-----------|-----------------|-------------------|----------------|-------------------|
| 0% (baseline) | 2.67x | 8192 | 3212.75 | 0% |
| 10% | 2.40x | 7373 | 3001.41 | 6.58% |
| 20% | 2.13x | 6554 | 2790.07 | 13.16% |
| 30% | 1.87x | 5735 | 2578.73 | 19.73% |
| 40% | 1.60x | 4916 | 2367.38 | 26.31% |
| 50% | 1.33x | 4096 | 2155.79 | 32.90% |
| 60% | 1.07x | 3277 | 1944.44 | 39.48% |

**Architecture details:**
- `hidden_size`: 3072
- `num_hidden_layers`: 28
- Baseline `intermediate_size`: 8192

## Key Observations

**Different baseline expansion ratios:**
- The 1B model starts with a higher expansion ratio (4x) than the 3B model (2.67x)
- This means achieving equivalent expansion ratios requires different pruning percentages across models
- Example: A 2.4x expansion ratio requires 40% pruning in 1B but only 10% pruning in 3B

**Non-linear parameter reduction:**
- Pruning percentage and parameter reduction are not proportional
- 10% pruning removes ~6.5% of parameters
- 60% pruning removes ~39% of parameters
- This is because MLP layers represent only a fraction of total model parameters

## Data Structure

Each entry in `expansion_rates.json` includes:

- Original and pruned architectural dimensions
- Total parameter counts before and after pruning
- Absolute and relative parameter reductions
- OptIFAIR verification metrics (exact expansion rates with floating-point precision)

## Usage Note

The expansion ratio serves as the primary independent variable in this research. When comparing results across models or referencing configurations, use the expansion ratio (e.g., "2.4x expansion") rather than pruning percentage, as models with different baseline architectures require different pruning amounts to achieve equivalent expansion ratios.

## Pruning Method

These configurations were generated using the OptIFAIR library with:
- **Neuron selection:** MAW (Maximum Absolute Weight)
- **Pruning type:** MLP_GLU (structured width pruning)
- **Target layers:** `gate_proj` and `up_proj` (pruned in pairs)
- **Downstream adjustment:** `down_proj` input dimension updated accordingly

All pruning is applied uniformly across transformer blocks.

[![Powered by OptIPFair](https://img.shields.io/badge/Powered%20by-OptIPFair-orange?style=flat&logo=github)](https://github.com/peremartra/optipfair)
