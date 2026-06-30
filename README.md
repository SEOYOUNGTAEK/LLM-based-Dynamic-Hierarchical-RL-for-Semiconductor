# LLM-Governed Action Constraints for Fault-Tolerant Reinforcement Learning

Official code for the AAAI 2027 submission:
> **LLM-Governed Action Constraints for Fault-Tolerant Reinforcement Learning**

We propose a hierarchical framework that integrates an LLM as a runtime constraint governor over a low-level RL agent for semiconductor fab scheduling. The LLM dynamically masks infeasible actions and penalizes fault-prone states, enabling fault-tolerant scheduling without manual rule engineering.

## Framework Overview

The system operates as a three-level hierarchy:

| Level | Module | Role |
|-------|--------|------|
| L2 | `agents/llm_agent.py` | LLM constraint governor (Mistral-7B) — dynamic action masking & fault detection |
| L1 | `agents/safety_agent.py` | Safety sanitizer — translates LLM output into executable masks |
| L0 | `agents/dqn/rainbow_dqn.py` | Rainbow DQN executor |

The environment is an Implant (IPLT) process in a semiconductor fab, interfaced via PlantSim digital twin.

## Repository Structure

```
.
├── agents/                          # Core algorithm
│   ├── llm_agent.py                 # L2: LLM constraint governor
│   ├── safety_agent.py              # L1: Safety sanitizer
│   └── dqn/rainbow_dqn.py          # L0: Rainbow DQN
│
├── env/                             # Simulation environment
│   ├── twin.py                      # Digital twin (PlantSim COM interface)
│   ├── plantsim/                    # PlantSim Python API wrapper
│   └── factory_data_generator/      # Factory state generation
│
├── utils/                           # Shared utilities
│
├── experiments/                     # Training entry points
│   ├── train_proposed.py            # Proposed model (LLM-governed)
│   ├── train_puradrl.py             # Pure DRL baseline
│   ├── baselines/                   # Comparison baselines
│   └── ablations/                   # Ablation study variants
│
├── scripts/                         # Multi-seed experiment automation
│
└── analysis/                        # Result analysis & visualization
```

## Requirements

- Python 3.10+
- PlantSim 16 (Windows, COM interface) — open `IPLT.spp` to launch the simulation
- OpenAI-compatible API endpoint serving Mistral-7B

```bash
pip install torch numpy pandas openai pydantic orjson pythoncom
```

## Running Experiments

All commands are run from the **repository root**:

```bash
# Proposed model
python experiments/train_proposed.py --seed 42

# Pure DRL baseline
python experiments/train_puradrl.py --seed 42

# Full ablation suite (N=20 seeds)
python scripts/run_ablations.py

# Full experiment matrix across all conditions
python scripts/run_full_matrix.py
```

## Key Results (N=20 seeds)

| Method | Mean Completion Time (s) | Fault Rate |
|--------|--------------------------|------------|
| Pure DRL | 3,621 ± 312 | 60% |
| **Proposed (ours)** | **3,184 ± 198** | **15%** |

Fault threshold τ = 3,500 s. Statistical test: t = −3.41, p = 0.0015, Cohen's d = 1.08.
