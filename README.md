# MAGES: Multi-Agent Game Evaluation & Cognitive Enhancement Platform

MAGES is a unified experimental framework for comparing multiple decision-making agent paradigms (Cognitive-Enhanced model, Chain-of-Thought, ReAct, Werewolf-Inspired, etc.) in structured geopolitical / strategic simulation scenarios. It supports:
- Batch comparative runs across agent reasoning styles
- Unified experiment directory organization with timestamped folders
- Cognitive model ablation studies (world model / hypothesis reasoning / full removal)
- Strategy group comparisons (same strategy on both sides)
- Multi-dimensional evaluation metrics (Event Alignment, Action Similarity, Strategy Rationality, Outcome Match + aggregated final score)

---
## 1. Core Directory Structure
```
MAGES/
├─ run_comparison.py              # Main interactive entrypoint (recommended)
├─ comparative_cognitive_world.py # Unified world wrapper for different agent types
├─ main.py                        # Example plotting script (not core)
├─ requirements.txt               # Python dependencies
│
├─ agents/                        # Custom country agents implementing reasoning paradigms
│   ├─ cot_agent.py               # Chain-of-Thought agent
│   ├─ react_agent.py             # ReAct agent
│   └─ werewolf_agent.py          # Werewolf-inspired agent
│
├─ models/                        # Generic or abstract layers (if used)
│   ├─ agents/                    # Base agent definitions
│   └─ cognitive/                 # Cognitive model components (profile, memory, reasoning, learning, evaluation, logging)
│       ├─ agent_profile.py
│       ├─ cognitive_agent.py
│       ├─ hypothesis_reasoning.py
│       ├─ learning_system.py
│       ├─ evaluation_system.py   # Metric computation
│       └─ experiment_logger.py   # Logging & persistence
│
├─ simulation/                    # Scenario implementations (e.g., PowerGameWorld, PrisonerDilemma)
│   └─ examples/PowerGameWorld/
│        └─ entity/               # World entities, rule systems, loggers, structured memory
│
├─ experiments/                   # Auto-generated experiment outputs
│   ├─ unified_comparison_*/      # A unified comparison run (timestamped)
│   │    ├─ *_test_*/             # Sub-folders per method (cot/react/werewolf/cognitive)
│   │    │    ├─ experiment_info.json
│   │    │    ├─ logs/
│   │    │    ├─ evaluation/
│   │    │    ├─ cognition_data/  # Only for cognitive model
│   │    │    └─ summary/
│   │    ├─ quick_comparison_results.json
│   │    ├─ unified_comparison_results.json
│   │    ├─ cuban_ablation_comparison_results.json (if ablation run)
│   │    └─ cognitive_strategy_groups_results.json (if strategy group run)
│   └─ cognitive_model_test_*/    # Standalone cognitive model runs
│
├─ visiualize/                    # (Typo: should be visualize) plotting scripts (radar/bar/strategy)
└─ figures/                       # Generated figures / exports
```

---
## 2. Environment & Installation
### 2.1 Python Version
Recommended: Python 3.10+ (for compatibility with torch, langchain, pandas, etc.).

### 2.2 Create Virtual Environment (Windows)
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2.3 Install Dependencies
```cmd
pip install -r requirements.txt
```
If you encounter regional network issues, configure a mirror (optional):
```cmd
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---
## 3. Environment Variables (LLM / API Backends)
`run_comparison.py` sets defaults using `os.environ.setdefault(...)`. Override them explicitly for security and portability:
```cmd
set DASHSCOPE_API_KEY=YOUR_KEY
set DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```
For OpenAI-compatible endpoints:
```cmd
set OPENAI_API_KEY=YOUR_OPENAI_KEY
set OPENAI_BASE_URL=https://api.openai.com/v1
```
(If using a proxy / compatible gateway, adjust `OPENAI_BASE_URL` accordingly.)

---
## 4. Quick Start
### 4.1 Interactive Menu (Recommended)
```cmd
python run_comparison.py
```
You will see:
```
1. Quick comparison (Cognitive + CoT + ReAct + Werewolf)
2. Single method test
3. Unified comparison (all results under one folder)
4. Cognitive model standalone test
5. Custom subset comparison
6. Cuban Missile Crisis ablation comparison
7. Cognitive model strategy-group comparison (same strategy both sides)
```
Enter the number and press Enter.

### 4.2 Output Locations
- All experiments: `experiments/`
- Per-method run: `*/logs`, `*/evaluation`, `*/summary`, plus `cognition_data` for cognitive
- Aggregated JSON: `quick_comparison_results.json`, `unified_comparison_results.json`, etc.

---
## 5. Experiment Modes
| Mode | Description | Key Output |
|------|-------------|------------|
| Quick Comparison | Run all four methods once | quick_comparison_results.json |
| Single Method | Run one chosen method | method-specific test_* folder |
| Unified Comparison | All methods, single parent folder | unified_comparison_results.json |
| Cognitive Standalone | Only cognitive world | cognitive_model_test_* |
| Custom Subset | Interactive subset selection | unified_comparison_* |
| Ablation (Cuban Crisis) | Remove world model / hypothesis reasoning / all | cuban_ablation_comparison_results.json |
| Strategy Groups | Both sides same strategy (flexible/hardline/concession/tit-for-tat) | cognitive_strategy_groups_results.json |

---
## 6. Evaluation Metrics
Implemented in `evaluation_system` (invoked via `experiment_logger.run_evaluation()`):
- EA (Event Alignment): Consistency between actions and evolving context
- AS (Action Similarity): Semantic / categorical proximity to expected or normative templates
- SR (Strategy Rationality): Internal logical coherence & strategic continuity
- OM (Outcome Match): Alignment of final state with desired / stable outcomes
- final_score: Weighted aggregate (customize via `run_final_evaluation(weights={...})`)

---
## 7. Core Architecture & Flow
1. `ComparativeCognitiveWorld` orchestrates agents (cognitive / cot / react / werewolf) in a shared turn-based environment.
   - `start_sim(max_steps=8)` loops `run_one_step()`
   - Each step: world snapshot → agent decisions → rule-based (or LLM) feedback → evaluation logging → learning → tension update → termination check
   - Final: report generation + evaluation
2. Custom Agents: Implement distinct reasoning paradigms under `agents/`.
3. Cognitive Model: Backed by `simulation.examples.PowerGameWorld.entity.cognitive_world` + modular enhancements in `models/cognitive/`.
4. Evaluation: `experiment_logger` aggregates per-round structured memory or logs and computes metrics.

---
## 8. Extending the Platform
| Goal | How to Proceed |
|------|----------------|
| Add a new reasoning method | Copy an existing agent file; implement decision logic; register in `AGENT_TYPES` + `_initialize_agents()` |
| Add a new metric | Extend `evaluation_system.py` and integrate into scoring aggregation |
| Change metric weights | Call `run_final_evaluation(weights={"ea":0.2, "as":0.3, ...})` |
| New scenario | Add a folder under `simulation/examples/YourWorld`; implement entity + rule systems |
| External visualization | Use data in `evaluation/` + `summary/` or build dashboards from logs |

---
## 9. FAQ
1. No cognitive folders generated? → Only created when running the cognitive model.
2. All scores zero / missing? → Check `experiments/.../logs/console_output.log`; confirm evaluation rounds were recorded and feedback generated.
3. Install failures? → Ensure correct Python version; clear pip cache; verify that `requirements.txt` is UTF-8.
4. How to shorten runs? → Lower `max_steps` when calling `start_sim()`.
5. Want deterministic runs? → (Future) add seeding; currently depends on model outputs and timing.
6. Encoding issues? → All repository text files should be UTF-8; re-save if needed.

---
## 10. Example Output Structure (Quick Comparison)
```
experiments/
└─ unified_comparison_0910_1530/
    ├─ cognitive_model_test_0910_1530/
    ├─ cot_test_0910_1531/
    ├─ react_test_0910_1532/
    ├─ werewolf_test_0910_1533/
    ├─ quick_comparison_results.json
    └─ unified_comparison_results.json
```

