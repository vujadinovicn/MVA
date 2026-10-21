# Advanced Learning for Text and Graph Data (ALTEGRAD)

Course by Michalis Vazirgiannis. [Course page](https://www.master-mva.com/cours/cat-advanced-learning-for-text-and-graph-data-altegrad/)

The course is about machine learning and state-of-the-art AI methods for text and graph data, with a focus on applications. It covers text representations, attention and transformers, large language models and their fine-tuning and alignment, deep learning on graphs, and multimodal and generative models. 

## Topics

1. **Intro to NLP and attention.** Word embeddings, RNNs, LSTMs, GRUs, seq2seq, attention, graph-of-words.
2. **Deep learning for NLP.** CNNs and hierarchical attention networks for text, self-supervised pretraining.
3. **LLMs.** Transformers, positional encodings, pretraining, instruction tuning, LoRA.
4. **LLM advanced topics.** RLHF, PPO, DPO, retrieval-augmented generation, quantization, graph-augmented LLMs.
5. **Deep learning for graphs I.** Graph Laplacians, graph kernels, DeepWalk, node2vec, GCN, GraphSAGE, GAT.
6. **Deep learning for graphs II.** Expressivity of message passing (1-WL), graph pooling, equivariant GNNs.
7. **Multimodal LLMs and generative AI.** Agents, graph generation, molecules and proteins.

## Content of this folder

| Lab | Topic | Folder |
| --- | --- | --- |
| Lab 1 | Neural machine translation | [lab1-neural-machine-translation](lab1-neural-machine-translation) |
| Lab 2 | Pretraining and supervised fine-tuning | [lab2-pretraining-and-finetuning](lab2-pretraining-and-finetuning) |
| Lab 3 | Alignment with DPO and GRPO | [lab3-dpo-and-grpo](lab3-dpo-and-grpo) |
| Lab 4 | Graph analysis and graph kernels | [lab4-graph-analysis-and-kernels](lab4-graph-analysis-and-kernels) |
| Lab 5 | Distillation and RAG | [lab5-distillation-and-rag](lab5-distillation-and-rag) |
| Lab 6 | DeepWalk and GNNs | [lab6-deepwalk-and-gnn](lab6-deepwalk-and-gnn) |
| Lab 7 | GAT and variational graph autoencoders | [lab7-gat-and-vgae](lab7-gat-and-vgae) |

## Lab 1: Neural machine translation

Notebook: [lab1_neural_machine_translation.ipynb](lab1-neural-machine-translation/lab1_neural_machine_translation.ipynb)

We implement a seq2seq model for machine translation, with an RNN encoder, a decoder and a global attention mechanism, and train it. Then we analyze the limits of greedy decoding and the typical translation errors.

## Lab 2: Pretraining and supervised fine-tuning

Notebook: [lab2_pretraining_and_finetuning.ipynb](lab2-pretraining-and-finetuning/lab2_pretraining_and_finetuning.ipynb)

We fine-tune a small LLM (Qwen2.5-0.5B) on question answering. We set up the 4-bit quantization and the LoRA adapters ourselves, then train. We analyze what LoRA changes in the number of trainable parameters and the memory needed for training, and compare the model before and after fine-tuning.

## Lab 3: Alignment with DPO and GRPO

Notebook: [lab3_dpo_and_grpo.ipynb](lab3-dpo-and-grpo/lab3_dpo_and_grpo.ipynb)

We align a small LLM (Qwen2.5-0.5B-Instruct) on French data. First we train it with DPO, where the model learns to prefer chosen answers over rejected ones while staying close to a reference model. Then we train it with GRPO, a reinforcement learning alternative.

## Lab 4: Graph analysis and graph kernels

Code: [lab4-graph-analysis-and-kernels](lab4-graph-analysis-and-kernels), report: [report.pdf](lab4-graph-analysis-and-kernels/report.pdf)

We study a real collaboration network (CA-HepTh) and implement community detection with spectral clustering and modularity. Then we implement the graphlet kernel and use it to classify molecules (MUTAG). Finally we classify questions (TREC) represented as graphs of words, using several graph kernels from the GraKeL library.

## Lab 5: Distillation and RAG

Notebook: [lab5_distillation_and_rag.ipynb](lab5-distillation-and-rag/lab5_distillation_and_rag.ipynb)

We implement two techniques to make a small LLM useful for question answering. For distillation, we generate synthetic questions on Wikipedia with Qwen2.5-7B and train Qwen2.5-0.5B on its output distributions, to specialize it for RAG. For RAG, we build a retrieval pipeline from scratch (chunking, embeddings, similarity search, reranking) and use the retrieved text in the prompt.

## Lab 6: DeepWalk and GNNs

Code: [lab6-deepwalk-and-gnn](lab6-deepwalk-and-gnn), report: [report.pdf](lab6-deepwalk-and-gnn/report.pdf)

We implement DeepWalk and use the node embeddings for visualization and node classification. Then we implement a GNN and train it for node classification on the Karate club and Cora.

The two big data files (`web_sample.edgelist` and `cora.content`) are not in the repo.

## Lab 7: GAT and variational graph autoencoders

Code: [lab7-gat-and-vgae](lab7-gat-and-vgae), report: [report.pdf](lab7-gat-and-vgae/report.pdf)

We implement a graph attention network (GAT) and use it for node classification, then look at the learned attention scores. We also implement a variational graph autoencoder (VGAE) and use it for graph generation.
