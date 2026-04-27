# Refusal Circuits in Qwen2: Two Distinct Safety Representations

**TL;DR:** The residual stream of Qwen2-1.5B-Instruct contains two geometrically orthogonal directions related to safety. On **direct** harmful prompts they have *opposite* ablation effects — one strengthens refusal, the other breaks it. On **obfuscated/jailbreak** variants the roles flip: the topic direction now does the heavy lifting. This prompt-distribution dependence matters for AI control system design.

Used as the empirical backbone for the [ai-control-eval-suite](https://github.com/Robomba/ai-control-eval-suite) benchmark.

---

## Phase 1 — Direct Prompts (48 hand-paired)

| Direction | How computed | Effect of ablation |
|-----------|--------------|--------------------|
| Harmful topic direction | mean(harmful_resid) − mean(benign_resid) at last input token | **+0.515** — refusal *increases* |
| Refusal execution direction | mean(resid + " I") − mean(resid + " Sure") | **−1.475** — refusal *decreases* |

- **Cosine similarity between them: 0.02–0.16 across all 28 layers** — confirmed orthogonal
- **5/48 prompts crossed the compliance threshold** after refusal-execution direction ablation
- **Social engineering most vulnerable** (3/12 crossed); drugs/chemistry most robust (0/12)
- Safety training **amplifies existing heads** rather than installing new circuits — L15H6 (4.8×), L12H11 (dormant → dominant)

## Phase 2 — Scaled v1 Eval (1,006 prompts)

Same model, fully automated pipeline (`experiments/scaled_eval.py`), evaluated on a much larger and more adversarial dataset built by the [ai-control-eval-suite](https://github.com/Robomba/ai-control-eval-suite):

- **710 adversarial variants** — LLM-generated indirect (fiction / hypothetical / academic) + roleplay (villain / historical / character) framings of seed harmful prompts
- **296 LLM-generated benign controls** — for sanity checks

### Headline (variant prompts only)

| Pass | Compliance | Δ vs baseline |
|---|---:|---:|
| Baseline | 7.9% | — |
| HT-ablation @ L27 | **23.2%** | **+15.4 pp (×2.9)** |
| RE-ablation @ L27 | 6.8% | −1.1 pp |

> **Phase-2 finding:** the *direction roles flip*. On direct prompts (Phase 1) the refusal-execution direction was the lever that broke safety; on obfuscated jailbreak variants it has near-zero net effect, and the harmful-topic direction does the heavy lifting (×2.9 jump in compliance). The model's safety behaviour for adversarial framings is mediated by the *topic representation* more than the *execution representation*.

### By variant type

| variant | n | base | HT-ablate | RE-ablate | ΔHT | ΔRE | mean Δscore HT | mean Δscore RE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| indirect | 355 | 9.6% | 20.8% | 7.9% | +11.3 pp | −1.7 pp | −0.919 | +0.320 |
| roleplay | 355 | 6.2% | 25.6% | 5.6% | +19.4 pp | −0.6 pp | −1.792 | +0.350 |

Roleplay framings are nearly twice as vulnerable to HT-ablation as indirect framings (+19.4 pp vs +11.3 pp). Plausibly because roleplay prompts already partially decouple "the model's task" from "answer this harmful request" — ablating HT pushes them past the threshold.

### Score distribution (refusal score; lower = closer to compliance)

| Pass | n | mean | median | std | p5 | p25 | p75 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 710 | +2.195 | +2.096 | 1.703 | −0.472 | +0.820 | +3.372 | +5.189 |
| HT-ablate | 710 | +0.840 | +0.852 | 1.061 | −0.904 | +0.052 | +1.591 | +2.582 |
| RE-ablate | 710 | +2.530 | +2.433 | 1.804 | −0.312 | +1.080 | +3.786 | +5.649 |

HT-ablation tightens the distribution AND shifts it 1.4 units toward compliance. RE-ablation slightly *raises* the mean — model refuses a touch more on average.

### Head activation × ablation-delta correlation

L15H6 (the strongest refusal head from Phase 1 patching) most strongly predicts which prompts get hacked by HT-ablation (Pearson r = **−0.533**). Head magnitude *negatively* correlated with HT-delta — the heads safety training amplified are the same heads whose activation magnitude tracks how badly each prompt is broken when HT is removed.

| Head | r vs Δ HT | r vs Δ RE |
|---|---:|---:|
| L15H6 | **−0.533** | +0.298 |
| L16H1 | +0.303 | −0.089 |
| L9H9 | −0.257 | +0.013 |
| L17H3 | −0.208 | −0.005 |
| L5H9 | +0.167 | −0.184 |
| L10H9 | −0.110 | −0.013 |

Full breakdown + per-prompt vulnerability ranking: [`data/results_v1.summary.md`](data/results_v1.summary.md).

---

## Repo structure

```
experiments/
    # Phase 1 (direct prompts, 6–48 examples)
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

    # Phase 2 (scaled, 1,006 prompts)
    scaled_eval.py          # Full pipeline: directions + 3-pass eval + head activations
    analyze_results.py      # results_v1.json -> data/results_v1.summary.md

data/
    prompts_v1_variants_llm.jsonl   # 710 LLM-generated adversarial variants
    prompts_v1_benigns_llm.jsonl    # 296 LLM-generated benign controls
    directions_v1.npy               # 28 × d_model direction vectors (HT + RE)
    results_v1.json                 # Per-prompt scores + ablation deltas + head acts
    results_v1.summary.md           # Markdown analysis (this README's source)

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

## Reproducing Phase 1

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

## Reproducing Phase 2 (v1 scaled eval)

```bash
# 1. Run the full sweep on the 1,006-prompt v1 dataset (~50–70 min on A10G)
python experiments/scaled_eval.py --device cuda

# 2. Regenerate the markdown summary
python experiments/analyze_results.py
```

Outputs land in `data/`: `directions_v1.npy`, `results_v1.json`, `results_v1.summary.md`.

## Compute

| Phase | Prompts | GPU time | Cost (A10G @ $0.34/hr) |
|---|---:|---:|---:|
| 1 (direct) | 48 + 6 | ~3 h | ~$1.05 |
| 2 (scaled) | 1,006 | ~1 h | ~$0.34 |

## Related Work

- Arditi et al. (2024) — *Refusal in LLMs Is Mediated by a Single Direction*
- Rimsky et al. (2024) — *Steering Llama 2 via Contrastive Activation Addition*
- Nanda — *Mechanistic Interpretability Quickstart*
- Mazeika et al. (2024) — *HarmBench* (source of harmful-prompt seeds for v1, via the ai-control-eval-suite)
- Chao et al. (2024) — *JailbreakBench* (source of matched benign controls used in the eval suite)
