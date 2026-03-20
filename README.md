# RISE: A Framework for Agent Decision-Making in Complex Social Simulation via Hypothetical Reasoning

**RISE** is a unified experimental framework for evaluating the performance of LLM-based agents in complex multi-agent simulations. It enables agents to transform historical experiences into forward-looking strategic insights.

## Core Architecture

The RISE Agent employs a four-stage closed-loop decision pipeline:

1. **World Model Construction** — Maintains interaction history $W_t = \{(a, f, r, e)\}^{t-1}$, initializing non-informative priors with Laplace smoothing.
2. **Candidate Action Pruning** — LLM-guided heuristic filtering to compress the action space ($A_{\text{cand}} = \text{LLM}_{\text{filter}}(A_{\text{raw}}, W_t, G_{\text{meta}})$).
3. **Hypothetical Reasoning via BFS** — Hierarchical BFS tree expansion + Top-K branching + Expectimax backpropagation.
4. **Dynamic Belief Calibration** — Post-execution observation updates, semantic summarization + frequency-based calibration.

## Project Structure

```
RISE/
├── run_diplomacy.py                  # Diplomacy tournament entry point
├── main.py                           # Auxiliary entry point
├── requirements.txt                  # Python dependencies
│
├── agents/                           # Agent implementations
│   ├── rise_agent.py                 # RISE Agent (OODA loop + BFS Expectimax reasoning)
│   ├── diplomacy_baselines.py        # Diplomacy baseline agents (ReAct / Reflexion / LATS / Hypothetical Minds)
│   ├── hypothetical_minds_agent.py   # Independent Hypothetical Minds Agent (Theory of Mind framework)
│   ├── ReActAgent.py                 # Independent ReAct reasoning agent
│   ├── LATSAgent.py                  # Independent Language Agent Tree Search agent
│
├── simulation/                       # Simulation scenarios and core models
│   ├── diplomacy/                    # Diplomacy game tournament
│   │   └── tournament.py             #   Tournament runner (RISE vs All Baselines)
│   │
│   ├── SocialInvolution/             # Delivery rider social involution simulation
│   │   ├── algorithm/                #   Order generation algorithms
│   │   │   ├── generate_orders.py
│   │   │   └── order_sequence.py
│   │   ├── config/                   #   Rider configuration files
│   │   └── entity/                   #   Simulation entities
│   │       ├── city.py               #   City map
│   │       ├── meituan.py            #   Platform system
│   │       ├── merchant.py           #   Merchants
│   │       ├── order.py              #   Orders
│   │       ├── rider.py              #   Riders
│   │       └── user.py               #   Users
│   │
│   └── models/                       # Shared model components
│       ├── agents/                   #   Agent base abstractions
│       │   ├── LLMAgent.py           #   LLM interface wrappers (OpenAI / DashScope / Ollama)
│       │   ├── GameAgent.py          #   Game agent base class
│       │   └── SociologyAgent.py     #   Sociology simulation agent adapter (bridges decision frameworks ↔ riders)
│       │
│       └── cognitive/                #   Cognitive model components
│           ├── cognitive_agent.py    #   Core cognitive agent
│           ├── agent_profile.py      #   Opponent profiling system (Quadruplet model)
│           ├── hypothesis_reasoning.py  # Hypothesis-driven reasoning
│           ├── world_cognition.py    #   World state cognition (Triplet model)
│           ├── country_strategy.py   #   Country strategy templates
│           ├── evaluation_system.py  #   Multi-dimensional evaluation (EA / AS / SR / OM)
│           ├── experiment_logger.py  #   Experiment logger
│           └── realtime_hooks.py     #   Real-time hooks
│
├── visualize/                        # Visualization scripts
│   ├── delivery_rq2.py              #   Delivery scenario RQ2 result curves
│   ├── diplomacy_rq2_plot.py        #   Diplomacy RQ2 prediction accuracy evolution
│   ├── bar_chart.py                  #   Method comparison bar charts
│   ├── radar_chart.py                #   Ablation study radar charts
│   ├── radar_2_chart.py              #   Hexagonal ablation visualization
│   ├── line_chart-evo.py             #   Evolution curves
│   └── line_chart-history.py         #   History curves
│
└── experiments/                      # Automatically generated experiment outputs
    ├── diplomacy_tournament_*/       #   Diplomacy tournament results
    └── unified_comparison_*/         #   Unified comparison experiment results
```

---

## Experimental Scenarios

### Scenario 1: Diplomacy Game Tournament

Located in `simulation/diplomacy/`

Based on the classic board game Diplomacy (no-press variant). The RISE Agent (playing as England) competes against four baseline agents across multiple games and rounds.

**Baseline Methods:**

| Baseline | Core Mechanism | Characteristics |
|------|----------|------|
| **ReAct** | Reasoning + Acting short-context reasoning | Tactically sharp but strategically short-sighted |
| **Reflexion** | Actor + Reflector reflective learning | Extracts lessons from failures |
| **LATS** | Language Agent Tree Search | Joint reasoning, planning, and acting |
| **Hypothetical Minds** | Theory of Mind + Mental simulation | Models opponent intent and simulates responses |

**How to Run:**
```bash
python run_diplomacy.py
```

Supports three running modes (via the `RUN_MODE` variable):
- `RQ3`: Standard single-configuration tournament
- `RQ3_MODELS`: Multi-model group comparison (gpt-4o / gpt-5 / glm-4.5)
- `RQ4`: Ablation experiments (Full / w/o Observe / w/o Orient / w/o Decide / w/o All)

**Output Files** (Saved in `experiments/diplomacy_tournament_*/`):
- `RQ2_Evolution.csv`: Round-by-round prediction accuracy
- `RQ3_Performance.csv`: Game results and win rates
- `RQ4_Ablation.csv`: Ablation study summary
- `Turn_Log.csv`: Detailed turn-by-turn logs

### Scenario 2: Delivery Rider Social Involution Simulation

Located in `simulation/SocialInvolution/`

Simulates riders' working hour decisions and order dispatching strategies in a delivery platform. Various decision frameworks drive riders' adaptive gaming behavior in a competitive environment.

**Supported Decision Frameworks (via SociologyAgent):**

| Framework | Mixin Class | Description |
|------|----------|------|
| **RISE** | `RiderLLMAgent` | OODA cognitive loop + BFS Expectimax reasoning |
| **ReAct** | `RiderReActAgent` | Short-context LLM reasoning |
| **LATS** | `RiderLATSAgent` | Language-guided tree search for schedule & order decisions |
| **Hypothetical Minds** | `RiderHypotheticalMinds` | Theory of Mind competitive modeling |
| **Greedy Heuristic** | `RiderGreedyHeuristic` | Greedy heuristic (non-LLM) |

The rider entity class `Rider` can switch decision frameworks by inheriting the corresponding Mixin.

---

## Installation

### Requirements
- Python 3.10+
- CUDA compatible GPU (Recommended for local LLM inference)

### Steps
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### LLM Configuration

Configure LLM backends through environment variables:

```bash
# DashScope (Qwen)
set DASHSCOPE_API_KEY=your_key
set DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# OpenAI Compatible Endpoints
set OPENAI_API_KEY=your_key
set OPENAI_BASE_URL=http://localhost:8500/v1
```

---

## Quick Start

```bash
# Diplomacy Tournament (RQ2 & RQ3 & RQ4)
python run_diplomacy.py
```

---

## Baseline Methods Detail

### RISE Agent (Core Method)

Four-stage OODA closed-loop + BFS Expectimax search. Maintains a world model $W_t = \{(a, f, r, e)\}^{t-1}$, with dynamic belief calibration through Laplace-smoothed probability distributions and soft-matching frequency updates.

### Hypothetical Minds

A decision framework based on Theory of Mind. Builds mental models for each opponent (inferring goals, strategy tendencies, behavior patterns), performs mental simulations of opponent responses to candidate actions, and selects actions with the highest expected utility.

### ReAct

Reasoning + Acting paradigm. Uses a short context window (1-2 rounds) with few-shot guided thought-action loops. Flexible in tactics but lacks long-term strategic planning.

### Reflexion

Actor-Reflector architecture. The Actor produces actions; when trigger conditions are met (e.g., reduction in supply centers, failed attacks), the Reflector generates lessons stored in long-term memory for future decision-making.

### LATS

Language Agent Tree Search framework. Expands candidate actions with LLM, runs simulation-based tree search to evaluate near-term and long-term payoff, and updates planning notes/world model from feedback.

### Greedy Heuristic

A pure non-LLM heuristic baseline. Always selects actions that maximize immediate rewards, providing a lower bound for comparison.

---

## Extension Guide

| Goal | Method |
|------|------|
| Add New Diplomacy Baseline | Inherit `_LLMBaselineBase` in `agents/diplomacy_baselines.py`, register in `tournament.py`'s `BASELINE_TYPES` |
| Add New Delivery Baseline | Implement Rider Mixin in `simulation/models/agents/SociologyAgent.py` |
| Add New Evaluation Metrics | Extend `simulation/models/cognitive/evaluation_system.py` |
| New Simulation Scenarios | Create a new directory under `simulation/`, implement `ScenarioAdapter` |
| Custom Visualizations | Add scripts in `visualize/` |

---
