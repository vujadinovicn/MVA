# Large Language Models for Code and Proof

Course by Marc Lelarge and Guillaume Baudart. [Course page](https://www.master-mva.com/cours/large-language-models-for-code-and-proof/)

The course is about the foundations of modern LLMs, with a focus on code generation. It covers the transformer architecture and efficient inference, tokenizers, pre-training, post-training and fine-tuning, retrieval-augmented generation, decoding and structured generation, formal verification of generated code, and agents. The evaluation is homework and a final exam.

## Topics by class

1. **Introduction.**
2. **Transformer architecture.** Attention, KV cache.
3. **Tokenizers.** For natural language and code.
4. **Pre-training, post-training and fine-tuning.** Scaling laws, LoRA.
5. **RAG.** Retrieval-augmented generation.
6. **Code generation.** Decoding strategies, structured generation, formal verification and theorem proving.
7. **Agents.**

## Content of this folder

| Homework | Topic | Folder |
| --- | --- | --- |
| Homework 1 | KV cache and LoRA | [llm-efficiency](llm-efficiency) |
| Homework 2 | FlashAttention in Triton | [flash-attention](flash-attention) |
| Homework 3 | Controlled generation | [controlled-generation](controlled-generation) |
| Homework 4 | Text adventure agent | [text-adventure-agent](text-adventure-agent) |

## Homework 1: LLM efficiency

Code: [llm-efficiency](llm-efficiency)

Two techniques to make an LLM cheaper, built on minGPT. We implement a KV cache, which stores the keys and values of the previous tokens so that only the new token is processed at each step, and benchmark the speedup. Then we implement LoRA, where the pre-trained weights are frozen and only low-rank adapters are trained.

## Homework 2: FlashAttention in Triton

Code: [flash-attention](flash-attention)

We implement FlashAttention-2 from scratch with Triton. We start with the online softmax, then a fused softmax-matmul kernel, and build the full attention kernel on top. We benchmark it against standard attention.

## Homework 3: Controlled generation

Code: [controlled-generation](controlled-generation)

Three ways to control the output of an LLM.
- **Structured generation.** We compile a regex into a token-level finite state machine and use it to mask the logits during generation, so the output always follows the format. It runs on a GPT-2 implemented in NumPy.
- **Meta-generation.** We generate several samples and select the best one, with best-of-n by log-probability and with Minimum Bayes Risk decoding (edit similarity and execution similarity). We evaluate on the MBPP coding benchmark.
- **Self-correction.** We build a tree-search pipeline where the LLM refines its code from the feedback of a verifier, to produce formally verified Rust code with Verus.

## Homework 4: Text adventure agent

Hugging Face Space: [vujadinovicn/text-adventure-agent](https://huggingface.co/spaces/vujadinovicn/text-adventure-agent), code: [text-adventure-agent](text-adventure-agent)

We build an LLM agent that plays text adventure games, using the ReAct pattern and tools exposed through MCP. We start from the baseline agent, play several games to see where it fails, and improve it.
- **State tracking.** The agent keeps track of the current location, its exits, the objects there and the actions already tried, and detects darkness. Locations and objects are extracted by parsing each observation.
- **Failed action detection.** An action counts as failed if the observation contains a known failure phrase or repeats an earlier observation. The agent does not retry a failed action at the same location.
- **Loop detection.** We detect repeated actions, cycles between locations or between directions, and repeated observations.
- **Recovery policy.** After a failed action, the agent samples a new action from untried candidates (movement, light, and verbs applied to the objects in the room).
