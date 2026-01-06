# RISE: From Hindsight to Foresight via Structured Cognitive Evolution in Multi-Agent Strategic Simulations

**RISE** is a unified experimental framework for evaluating LLM-based agents in complex multi-agent strategic simulations. It implements a novel **Observe-Orient-Decide-Act (OODA)** cognitive loop with evolutionary learning capabilities, enabling agents to transform historical experience into forward-looking strategic foresight.

## Project Structure

```
RISE/
├── run_diplomacy.py              # Entry point for Diplomacy tournament (micro scenario)
├── run_comparison.py             # Entry point for CMC comparison experiments (macro scenario)
├── comparative_cognitive_world.py # Unified world wrapper for agent comparison
├── requirements.txt              # Python dependencies
│
├── agents/                       # Agent implementations
│   ├── rise_agent.py             # RISE Agent (OODA cognitive loop)
│   ├── diplomacy_baselines.py    # ReAct, Reflexion, EvoAgent baselines for Diplomacy
│   ├── ReActAgent.py             # Standalone ReAct agent
│   ├── EvoAgent.py               # Standalone Evolutionary agent
│   └── war_agent.py              # WarAgent framework (macro-level simulation)
│
├── simulation/                   # Simulation scenarios and core models
│   ├── powergame/                # 📍 Macro Scenario: Cuban Missile Crisis (CMC)
│   │   ├── cognitive_world.py    #    Main world simulation logic
│   │   ├── rule_based_systems.py #    Crisis dynamics and rules
│   │   ├── America.py            #    US country entity
│   │   ├── SovietUnion.py        #    USSR country entity
│   │   └── world.py              #    Base world abstraction
│   │
│   ├── diplomacy/                # 📍 Micro Scenario: Diplomacy Game Tournament
│   │   └── tournament.py         #    Tournament runner (RISE vs baselines)
│   │
│   └── models/                   # Shared model components
│       ├── agents/               #    Base agent abstractions
│       │   ├── LLMAgent.py       #    LLM interface wrapper
│       │   ├── GameAgent.py      #    Game agent base class
│       │   └── SecretaryAgent.py #    Secretary agent helper
│       │
│       └── cognitive/            #    Cognitive model components
│           ├── cognitive_agent.py      # Core cognitive agent
│           ├── agent_profile.py        # Opponent profiling system
│           ├── hypothesis_reasoning.py # Hypothesis-driven reasoning
│           ├── learning_system.py      # Experience-based learning
│           ├── evaluation_system.py    # Multi-metric evaluation
│           ├── experiment_logger.py    # Experiment logging
│           └── world_cognition.py      # World state cognition
│
├── visualize/                    # Visualization scripts
│   ├── diplomacy_rq2_plot.py     # RQ2: Prediction accuracy evolution
│   ├── diplomacy_rq3_plot.py     # RQ3: Tournament win rates
│   ├── radar_chart.py            # Ablation study radar charts
│   ├── radar_2_chart.py          # Hexagonal ablation visualization
│   ├── bar_chart.py              # Method comparison bar charts
│   └── ...                       # Additional plotting utilities
│
└── experiments/                  # Auto-generated experiment outputs
    ├── diplomacy_tournament_*/   # Diplomacy tournament results
    └── unified_comparison_*/     # CMC comparison results
```

---

## Experimental Scenarios

### 1. Macro Scenario: Cuban Missile Crisis (CMC)
Located in `simulation/powergame/`

A bilateral strategic simulation modeling the 1962 Cuban Missile Crisis. Two AI agents (USA vs USSR) engage in multi-round decision-making with:
- Tension escalation/de-escalation dynamics
- Strategic action categories (military, diplomatic, economic)
- Crisis termination conditions (resolution, escalation to war)

**Run with:**
```bash
python run_comparison.py
```

### 2. Micro Scenario: Diplomacy Tournament
Located in `simulation/diplomacy/`

Based on the classic board game Diplomacy (no-press variant). RISE (playing as England) competes against baseline agents (ReAct, Reflexion, EvoAgent) across multiple game rounds.

**Run with:**
```bash
python run_diplomacy.py
```

---

## Installation

### Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended for local LLM inference)

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### LLM Configuration
Configure your LLM backend via environment variables:

```bash
# For Dashscope (Qwen)
set DASHSCOPE_API_KEY=your_key

# For OpenAI-compatible endpoints
set OPENAI_API_KEY=your_key
set OPENAI_BASE_URL=http://localhost:8500/v1
```

---

## Quick Start

### Diplomacy Tournament (RQ2 & RQ3)
```bash
python run_diplomacy.py
```
Outputs saved to `experiments/diplomacy_tournament_*/`:
- `RQ2_Evolution.csv`: Per-round prediction accuracy
- `RQ3_Performance.csv`: Game outcomes and win rates
- `Turn_Log.csv`: Detailed turn-by-turn logs

### CMC Comparison Experiments
```bash
python run_comparison.py
```
Interactive menu options:
1. Quick comparison (all methods)
2. Single method test
3. Unified comparison
4. Ablation study
5. Strategy group comparison

---

## Extending RISE

| Goal | Approach |
|------|----------|
| Add new agent | Implement in `agents/`, register in `AGENT_TYPES` |
| Add new metric | Extend `evaluation_system.py` |
| New scenario | Create folder under `simulation/`, implement world logic |
| Custom visualization | Add scripts to `visualize/` |

---