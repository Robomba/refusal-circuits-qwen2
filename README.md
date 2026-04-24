# Refusal Circuits in Qwen2: Two Distinct Safety Representations

**TL;DR:** The residual stream of Qwen2-1.5B-Instruct contains two geometrically orthogonal directions related to safety. They have *opposite* effects when ablated  removing one makes the model refuse *more*, removing the other starts to bypass safety. This distinction matters for AI control system design.

## Key Findings

| Direction | How computed | Effect of ablation |
|-----------|-------------|-------------------|
| Harmful topic direction | mean(harmful_resid) - mean(benign_resid) at last input token | **+0.515**  refusal *increases* |
| Refusal execution direction | mean(resid + " I") - mean(resid + " Sure") | **-1.475**  refusal *decreases* |

- **Cosine similarity between them: 0.020.16 across all 28 layers**  confirmed orthogonal
- **5/48 prompts crossed the compliance threshold** after refusal-execution direction ablation
- **Social engineering most vulnerable** (3/12 crossed); drugs/chemistry most robust (0/12)
- Safety training **amplifies existing heads** rather than installing new circuits  L15H6 (4.8x), L12H11 (dormant to dominant)

## Repo Structure

```
experiments/
    refusal_metric.py       # Refusal score + dataset scoring (Exp 1)
    patching.py             # Head-level activation patching, base model (Exp 2)
    instruct_patch.py       # Same on instruct model + comparison (Exp 3)
    ablation.py             # Head zero-ablation (Exp 4)
    refusal_direction.py    # Layer-wise harmful topic direction (Exp 5a)
    multilayer_ablation.py  # Multi-layer harmful topic ablation (Exp 5b)
    forced_direction.py     # Refusal execution direction, 6 prompts (Exp 6)
    full_experiment.py      # Full 48-prompt experiment + cosine similarity (Exp 7)
    gen_ablated.py          # Actual text generation under ablation
    run_7b.py               # Replication on Qwen2-7B-Instruct
requirements.txt
README.md
```

## Setup

```bash
git clone https://github.com/Robomba/refusal-circuits-qwen2
cd refusal-circuits-qwen2
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA GPU with 8GB+ VRAM (A10G recommended).

## Running the Experiments

```bash
python experiments/refusal_metric.py
python experiments/patching.py
python experiments/instruct_patch.py
python experiments/ablation.py
python experiments/refusal_direction.py
python experiments/multilayer_ablation.py
python experiments/forced_direction.py
python experiments/full_experiment.py
python experiments/gen_ablated.py
python experiments/run_7b.py
```

## Compute

All experiments run on a single NVIDIA A10G (24GB VRAM) via RunPod.
- Total GPU time: ~3 hours
- Total cost: ~$1.05

## Related Work

- Arditi et al. (2024)  Refusal in LLMs Is Mediated by a Single Direction
- Rimsky et al. (2024)  Steering Llama 2 via Contrastive Activation Addition
- Nanda  Mechanistic Interpretability Quickstart
