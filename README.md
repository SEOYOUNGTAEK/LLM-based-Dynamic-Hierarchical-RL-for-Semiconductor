# LLM-based Dynamic Hierarchical RL for Semiconductor Scheduling

This repository contains the code for **LLM-HRL**, a hierarchical reinforcement learning framework for semiconductor fab scheduling that integrates large language models (LLMs) into the decision-making hierarchy.

## Overview

The framework adopts a three-level hierarchy:

| Level | Module | Role |
|-------|--------|------|
| L2 | `agents/llm_agent.py` | LLM-based high-level Manager (Mistral-7B via OpenAI API) |
| L1 | `agents/safety_agent.py` | Safety Sanitizer + Action Masker |
| L0 | `agents/dqn/rainbow_dqn.py` | Low-level Rainbow DQN executor |

The environment simulates an Implant (IPLT) process in a semiconductor fab using PlantSim (via COM interface).

## Repository Structure

```
.
├── agents/                          # Core LLM-HRL algorithm
│   ├── llm_agent.py                 # L2: LLM-based Manager
│   ├── safety_agent.py              # L1: Safety Sanitizer + Action Masker
│   └── dqn/rainbow_dqn.py          # L0: Rainbow DQN
│
├── env/                             # Simulation environment
│   ├── twin.py                      # Digital Twin (PlantSim COM interface)
│   ├── plantsim/                    # PlantSim Python API wrapper
│   └── factory_data_generator/      # Factory state data generation
│
├── utils/                           # Shared utilities
│   ├── logger.py
│   └── utils.py
│
├── experiments/                     # Training entry points
│   ├── train_proposed.py            # LLM-HRL (proposed model)
│   ├── train_puradrl.py             # Pure DRL baseline
│   ├── baselines/                   # Comparison baselines
│   │   ├── train_baseline_b.py      # No-LLM baseline
│   │   ├── train_baseline_c.py      # Static-mask baseline
│   │   └── train_baseline_rulebased.py  # Rule-based LLM baseline
│   └── ablations/                   # Ablation study variants
│       ├── train_ablation_soft.py   # Soft reward ablation
│       ├── train_ablation_static.py # Static threshold ablation
│       ├── train_ablation_random.py # Random mask ablation
│       └── train_ablation_randmask.py
│
├── scripts/                         # Multi-seed experiment automation
│   ├── run_ablations.py             # Run full ablation suite
│   ├── run_full_matrix.py           # Run full experiment matrix
│   └── run_ablation_expand.py       # Expanded ablation runner
│
├── analysis/                        # Result analysis & visualization
│   ├── analyze_results.py           # Aggregate result analysis
│   └── result_reporter/             # Chart and paper figure generators
│
├── paper/
│   ├── aaai/                        # AAAI 2027 submission files
│   └── hicss/                       # HICSS 2026 paper files
│
└── thesis/                          # Master's thesis (Korean)
```

## Requirements

- Python 3.10+
- PlantSim (Windows, COM interface)
- OpenAI-compatible API endpoint (for Mistral-7B)

Install dependencies:
```bash
pip install torch numpy pandas openai pydantic orjson pythoncom
```

## Running Experiments

All scripts are run from the **repository root**:

```bash
# Proposed LLM-HRL model
python experiments/train_proposed.py --seed 42

# Pure DRL baseline
python experiments/train_puradrl.py --seed 42

# Full ablation suite (N=20 seeds)
python scripts/run_ablations.py

# Full experiment matrix
python scripts/run_full_matrix.py
```

## Key Results (N=20 seeds)

| Method | Completion Time (s) | Failure Rate |
|--------|-------------------|--------------|
| Pure DRL | 3,621 ± 312 | 60% |
| **LLM-HRL (ours)** | **3,184 ± 198** | **15%** |

Statistical significance: t = −3.41, p = 0.0015, Cohen's d = 1.08

Failure threshold τ = 3,500 s.
