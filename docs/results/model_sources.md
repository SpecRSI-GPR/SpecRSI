# Backbone sources

The table lists official sources for all twenty system backbones in Figure 4B. The ten marked **Yes** also appear in Figure 2 and Figure 4A. Labels follow the paper; links identify the corresponding model releases or model cards.

## Model roles and comparison settings

The task backbone and the improver are distinct roles. In Figure 4, the task agent uses the listed backbone, while SpecRSI transfers the same frozen Qwen3.5-35B-A3B-trained Drafter, Scorer, and Policy across backbones. There is no further improver training on each transferred backbone. The reference implementation uses frozen BF16 text representations, rank-16 Drafter LoRA, and separate 256-dimensional Scorer/Policy branches.

| Comparison | Task backbones and tasks | Improver and permissions |
| --- | --- | --- |
| Figure 4A | Ten backbones, each on PyBaMM, AQM, FDTD, Reactor, and IRP (50 pairs) | Shared frozen Qwen-trained improver; matched scientific-task edit permissions, starting agents, contracts, seeds, and evaluation-call cap K |
| Figure 4B | Twenty system backbones, including the Qwen reference | Shared frozen Qwen-trained improver; D+H edits (prompts, tool workflows, and non-test data), frozen task-model weights, matched starts, contracts, splits, seeds, and search caps |
| Table 1B | Reference-backbone system comparison | Upper block: matched D/H/M permissions; lower block: D+H only for MetaRSI-H and SpecRSI-H. D/H/M denotes data, harness, and task-model updates. |

The ten-backbone membership is listed in [Figure 2 seed data](data/figure2_seeds.csv) and [Figure 4A seed data](data/figure4a_seeds.csv); the twenty-backbone list follows [Figure 4B seed data](data/figure4b_scores_seeds.csv). Figure 2 uses the same ten names, with `GPT 5.5`, `GLM 5.3`, and `DeepSeek V4 Pro` written without the hyphens used below. The links below document model releases; run-specific checkpoint, API, and task-partition settings are separate configuration fields.

## Official sources

| Model label | Figures 2 and 4A | Official source |
|---|:---:|---|
| MiniMax M3 | Yes | [MiniMax M3 release](https://www.minimax.io/blog/minimax-m3) |
| DeepSeek V4.1 Flash | — | [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| Amazon Nova Premier | — | [Amazon Nova Premier release](https://aws.amazon.com/about-aws/whats-new/2025/04/amazon-nova-premier-complex-tasks-model-distillation/) |
| Cohere Command A+ | — | [Command A+ release](https://cohere.com/blog/command-a-plus) |
| GPT-5.5 | Yes | [GPT-5.5 release](https://openai.com/index/introducing-gpt-5-5/) |
| Claude Sonnet 5-1M | Yes | [Claude Sonnet 5 documentation, including its 1M-token context window](https://platform.claude.com/docs/en/models/sonnet-5/overview) |
| Qwen3.5-35B-A3B* | — | [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Mistral Large 3 | — | [Mistral 3 release, including Mistral Large 3](https://mistral.ai/news/mistral-3/) |
| Llama 4 Maverick | — | [Llama 4 release, including Maverick](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| Qwen3.5-397B-A17B | — | [Qwen3.5-397B-A17B model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| Kimi K3 | Yes | [Kimi K3 technical blog](https://www.kimi.com/en/blog/kimi-k3) |
| GLM-5.3 | Yes | [GLM-5.3 documentation](https://docs.z.ai/guides/llm/glm-5.3) |
| Qwen3.8 Max | — | [Qwen3.8-Max documentation](https://www.alibabacloud.com/help/en/model-studio/qwen3-8-max) |
| DeepSeek V4-Pro | Yes | [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| Grok-4.7 | — | [Grok 4.7 release](https://x.ai/news/grok-4-7) |
| Gemini 3.5 Flash | Yes | [Gemini 3.5 Flash model card](https://deepmind.google/models/model-cards/gemini-3-5-flash/) |
| GPT-5.6 Sol | Yes | [GPT-5.6 release, including Sol](https://openai.com/index/gpt-5-6/) |
| Claude Opus 5 | Yes | [Claude Opus 5 documentation](https://platform.claude.com/docs/en/models/opus-5/overview) |
| Claude Opus 5.5 | — | [Claude Opus 5.5 release](https://www.anthropic.com/claude-opus-5-5) |
| GPT-6 Astra | Yes | [GPT-6 Astra release](https://openai.com/index/gpt-6-astra/) |

*Reference backbone. The released BF16 reference configuration records its publisher revision in the [Qwen3.5-35B-A3B model lock](../../configs/models/qwen35_35b_a3b_bf16.lock.json). The paper's `Sonnet 5-1M` label denotes Sonnet 5 with a 1M-token context window; the source documentation names the model Claude Sonnet 5.

Official sources checked on 28 October 2026.
