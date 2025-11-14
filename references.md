# References

## Learning both Weights and Connections for Efficient Neural Networks

This paper introduces a method to reduce the storage and computation required by neural networks without affecting their accuracy. The key idea is to learn which connections are important and prune the rest.

The proposed method consists of a three-step process:
1.  **Train Connectivity:** First, a conventional network is trained to learn which connections are important.
2.  **Prune Connections:** Connections with weights below a certain threshold are removed.
3.  **Retrain:** The network is retrained to fine-tune the weights of the remaining connections.

This method was evaluated on AlexNet and VGG-16 models. For AlexNet, the number of parameters was reduced by a factor of 9x (from 61 million to 6.7 million) with no loss of accuracy. For VGG-16, the parameters were reduced by 13x (from 138 million to 10.3 million) with no loss of accuracy.

This paper is relevant to our project as it provides a foundational understanding of network pruning and demonstrates its effectiveness in reducing model size while maintaining performance.

## LLM-Pruner: On the Structural Pruning of Large Language Models

This paper proposes a task-agnostic structural pruning method for large language models (LLMs) called LLM-Pruner. The goal is to compress LLMs while preserving their multi-task solving and language generation abilities, with minimal reliance on the original training data.

The method consists of three stages:
1.  **Discovery Stage:** Identifies groups of interdependent structures within the LLM.
2.  **Estimation Stage:** Estimates the contribution of each group to the model's performance and decides which groups to prune.
3.  **Recover Stage:** Uses low-rank approximation (LoRA) for fast post-training to alleviate performance degradation.

LLM-Pruner was validated on LLaMA, Vicuna, and ChatGLM. The results show that with a 20% parameter reduction, the pruned model maintains 94.97% of the original model's performance.

This paper is highly relevant to our project as it focuses on structured pruning of LLMs, which is the core of our research. The task-agnostic approach and the efficient recovery method using LoRA are particularly interesting.

## A Simple and Effective Pruning Approach for Large Language Models

This paper introduces Wanda (Pruning by Weights and activations), a simple and effective pruning method for large language models (LLMs). The method is designed to induce sparsity in pretrained LLMs without requiring retraining or weight updates.

Wanda prunes weights with the smallest magnitudes multiplied by the corresponding input activations, on a per-output basis. The key idea is that the importance of a weight is determined not only by its magnitude but also by the magnitude of its corresponding input activation.

The method was evaluated on LLaMA and LLaMA-2 models and significantly outperforms magnitude pruning. It also performs competitively against more complex methods that involve weight updates.

This paper is relevant to our project as it presents a novel pruning metric that considers both weights and activations. This could be an interesting alternative to the methods we are currently using.

## A Survey on Model Compression for Large Language Models

This paper provides a comprehensive survey of model compression techniques for large language models (LLMs). The authors cover a wide range of methods, including:

*   **Quantization:** Reducing the number of bits used to represent model parameters.
*   **Pruning:** Removing redundant parameters from the model. This is further divided into unstructured, structured, and semi-structured pruning.
*   **Knowledge Distillation:** Training a smaller model to mimic the behavior of a larger model.
*   **Low-Rank Factorization:** Decomposing large weight matrices into smaller matrices.

The paper also discusses the various metrics and benchmarks used to evaluate the performance of compressed LLMs.

This survey is highly relevant to our project as it provides a broad overview of the field of model compression, with a specific focus on pruning. It helps to contextualize our work and provides a valuable resource for understanding the different approaches to model compression.

## SliceGPT: Compress Large Language Models by Deleting Rows and Columns

This paper introduces SliceGPT, a post-training sparsification scheme that compresses large language models (LLMs) by deleting entire rows and columns from weight matrices. This is a form of structured pruning that reduces the embedding dimension of the network.

The key idea behind SliceGPT is the concept of "computational invariance" in transformer networks. The authors show that it is possible to apply orthogonal transformations to the weight matrices of a transformer without changing the model's output. This allows them to project the signal between blocks onto its principal components and then remove the least important components.

The method was evaluated on LLaMA-2, OPT, and Phi-2 models. The results show that SliceGPT can remove up to 25% of the model parameters while maintaining high performance. The sliced models also run faster and on fewer GPUs.

This paper is highly relevant to our project as it deals with structured pruning of LLMs. The idea of computational invariance and the method of slicing entire rows and columns are very interesting and could be applicable to our work on GLU-MLP layers.

## Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods

This paper investigates depth pruning as a method for compressing large language models (LLMs). The authors show that simple depth pruning, which involves removing entire layers or blocks, can be an effective way to compress LLMs, achieving comparable or even superior performance to width pruning methods.

The paper makes the following key contributions:
*   It demonstrates that depth pruning can significantly boost inference speeds, especially in memory-constrained environments where width pruning is less effective.
*   It provides a detailed comparison of retraining methods for pruned models, showing that continued pretraining on a large corpus is more effective than LoRA-based tuning, especially at high pruning ratios.
*   It introduces a simple yet effective method for depth pruning of LLMs by exploring various design factors.

This paper is highly relevant to our project as it provides a direct comparison between depth and width pruning, which is a central theme of our research. The findings on the effectiveness of different retraining methods are also very valuable.
