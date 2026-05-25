# LLM Prompts

本文件收录 RISE 及所有 baseline agent 使用的全部 LLM 提示词，对应论文 §4.3 所承诺的公开内容。所有方法均遵循**零样本冷启动**协议，无离线自博弈、人类记录或少样本示例（ReAct 除外，其包含两条说明格式的 few-shot 示例）。

---

## 目录

1. [RISE Agent](#rise-agent)
   - [Stage 2 — Candidate Action Filter (`LLM_filter`)](#stage-2--candidate-action-filter-llm_filter)
   - [Stage 3 — Risk Assessment (`LLM_risk`)](#stage-3--risk-assessment-llm_risk)
   - [Stage 3 — Next-Action Actor (`LLM_actor`)](#stage-3--next-action-actor-llm_actor)
   - [Stage 3 — Leaf Utility Evaluator (`LLM_eval`)](#stage-3--leaf-utility-evaluator-llm_eval)
   - [Stage 4 — Experience Summarizer (`LLM_sum`)](#stage-4--experience-summarizer-llm_sum)
   - [Diplomacy — Concrete Order Translator](#diplomacy--concrete-order-translator)
   - [Fallback — Greedy Single-Call Decider](#fallback--greedy-single-call-decider)
2. [Baseline Agents](#baseline-agents)
   - [ReAct](#react)
   - [Reflexion — Actor](#reflexion--actor)
   - [Reflexion — Reflector](#reflexion--reflector)
   - [LATS — Expand](#lats--expand)
   - [LATS — Simulate](#lats--simulate)
   - [LATS — Post-Round Critic](#lats--post-round-critic)
   - [Hypothetical Minds — Decision](#hypothetical-minds--decision)

---

## RISE Agent

实现文件：`agents/rise_agent.py`

### Stage 2 — Candidate Action Filter (`LLM_filter`)

对应论文 Eq.(1)：$A_{\mathrm{cand}} = \mathrm{LLM}_{\text{filter}}(A_{\mathrm{raw}}, \mathcal{W}_t, G_{\mathrm{meta}})$

**User Prompt（模板）**

```
You are a strategic action filter.  Given the agent's world
model history, current state, and meta-goal, select a pruned
subset of promising candidate actions.  Eliminate actions that
lack execution feasibility or deviate from the current context.

Meta-Goal: {meta_goal}
World Model (recent interactions):
{world_summary}
Current State:
{state_desc}
Available Raw Actions: {actions}

Select only feasible, high-value actions.  Also provide a
one-sentence strategic guideline for the current situation.

Output JSON:
{"candidate_actions": ["action1", ...], "strategy": "one-sentence strategic guideline"}
```

---

### Stage 3 — Risk Assessment (`LLM_risk`)

对应论文 Eq.(3)：$\mathcal{L}_{\mathrm{safe}} = \mathrm{LLM}_{\text{risk}}(\mathcal{L}_d, G_{\mathrm{meta}})$

BFS 每层一次批量调用，对展开的所有候选状态打风险分（0~1），分数 ≥ `risk_threshold`（默认 0.7）的节点被剪枝。

**User Prompt（模板）**

```
You are a strategic risk assessor.  Evaluate the risk of each
hypothetical state against the meta-goal.

Meta-Goal: {meta_goal}
Strategy: {strategy}

States:
{states}

Output JSON: {"risks": [float, ...]}
Each value in [0, 1].  Array length MUST be {count}.
```

---

### Stage 3 — Next-Action Actor (`LLM_actor`)

对应论文 Eq.(4)：$a_{\mathrm{next}} = \mathrm{LLM}_{\text{actor}}(\mathcal{L}_{\mathrm{safe}}, A_{\mathrm{raw}})$

BFS 每层对存活节点批量分配下一步动作（仅在深度未到叶层时调用）。

**User Prompt（模板）**

```
For each state, select the best next action.

Meta-Goal: {meta_goal}
Strategy: {strategy}
Actions: {actions}

States:
{states}

Output JSON: {"actions": [str, ...]}
Array length MUST be {count}.
```

---

### Stage 3 — Leaf Utility Evaluator (`LLM_eval`)

对应论文 Eq.(5)：$U_{\mathrm{leaf}}(\ell) = \mathrm{LLM}_{\text{eval}}(\ell, G_{\mathrm{meta}}) \in [0, 1]$

到达搜索深度 $D$ 后对所有叶节点批量打效用分，用于 Expectimax 回传。

**User Prompt（模板）**

```
Evaluate each terminal state's alignment with the meta-goal.
Output a single utility score ∈ [0, 1] for each state where
1 = perfectly aligned and 0 = worst outcome.

Meta-Goal: {meta_goal}
Strategy: {strategy}

States:
{states}

Output JSON: {"utilities": [float, ...]}
Array length MUST be {count}.
```

---

### Stage 4 — Experience Summarizer (`LLM_sum`)

对应论文 Eq.(8)：$e_{\mathrm{obs}} = \mathrm{LLM}_{\text{sum}}(a^*, f_{\mathrm{obs}}, r_{\mathrm{obs}})$

执行动作后将本轮交互浓缩为一条语义经验 $e$，写入 Interaction Memory。

**User Prompt（模板）**

```
Summarise this interaction into a concise strategic lesson
(max 30 words):
Action: {action}
Feedback: {feedback}
Peer Reactions: {reactions}

Output JSON: {"experience": "..."}
```

---

### Diplomacy — Concrete Order Translator

将高层抽象动作（HOLD / MOVE / ATTACK 等）映射为 Diplomacy 引擎合法的逐单位指令。只在外交场景中调用。

**System Prompt**

```
Diplomacy(标准地图)要点：
- 每回合需要为每个可下单地点提交 1 条指令。
- 合法指令以引擎提供的 legal orders 为准；必须逐条原样选取。
- 常见指令：H(保持)、- (移动)、S(支援)、C(运输)。
- 目标优先级：争夺中立补给中心(SC)；保住本土；避免无谓对撞。
```

**User Prompt（模板）**

```
You are the tactical commander for {power} in Diplomacy.
You MUST ONLY select from the given legal orders.

Background:
{knowledge}
Goal: {goal}
Round={round} Phase={phase}
Intent: {action}
Strategy: {strategy}

My units: {my_units}
My SC: {my_centers}
Enemy: {enemy}

Legal orders:
{legal}

Pick ONE order per location.
Output JSON: {"orders_by_loc": {"LOC": "order"}, "rationale": "brief"}
```

---

### Fallback — Greedy Single-Call Decider

当假设推理被禁用（消融变体 `w/o Hypothetical Reasoning`）时退化为单次 LLM 调用。

**User Prompt（模板）**

```
Choose the single best action.
Strategy: {strategy}
Goal: {goal}
Actions: {actions}
State: {state}

Output JSON: {"action": "..."}
```

---

## Baseline Agents

实现文件：`agents/diplomacy_baselines.py`

---

### ReAct

对应论文 ReAct \[Yao et al., 2023\]。短上下文（最近 2 轮自身动作）+ 格式化 Thought/Action，战术敏锐但战略短视。

**System Prompt**

```
You are a Diplomacy baseline agent implementing ReAct (Reason+Act).
You are tactical, short-horizon, and only use the latest observation.
```

**User Prompt（含 2-shot 格式示例）**

```
Example 1
PlaintextObservation: Phase=S1902M Year=1902 MyUnits=F LON, A LVP MySC=LON, LVP
Thought: 1) Threat: Germany active. 2) Opportunity: secure tempo. 3) Choose defensive support.
Action: {"abstract_action":"SUPPORT_DEFEND", "daide":"(HLD)"}

Example 2
PlaintextObservation: Phase=F1903M Year=1903 MyUnits=F NTH MySC=LON,LVP,EDI
Thought: 1) Threat low. 2) Opportunity: expand. 3) Probe with move.
Action: {"abstract_action":"MOVE", "daide":"(MTO ...)"}

PlaintextObservation:
You are {name}.
Phase={phase} Year={year}
MyUnits({unit_count})={units}
MySC({sc_count})={centers}
Tension={tension}
LastTurnOrders={last_orders}
LastTurnFeedback={last_feedback}
RecentSelfActions={recent_actions}

Now follow the exact format:
Thought: <your concise tactical reasoning, max 5 lines>
Action: <a JSON object with keys abstract_action and daide>
abstract_action must be one of: [HOLD, MOVE, ATTACK, SUPPORT_ATTACK, SUPPORT_DEFEND, RETREAT].

Return a JSON object with keys: thought, abstract_action, daide.
Do not include extra keys.
```

---

### Reflexion — Actor

对应论文 Reflexion \[Shinn et al., 2023\]。使用长时反思记忆（最多保留最近 3 条教训）驱动动作选择。

**System Prompt**

```
You are a Diplomacy baseline agent implementing Reflexion (verbal RL).
You act, then learn from mistakes via short self-reflections.
[Lessons from the past]
{lessons}
```

**User Prompt（模板）**

```
Current situation for {name}:
Phase={phase} Year={year}
MyUnits({unit_count})={units}
MySC({sc_count})={centers}
Tension={tension}
LastTurnOrders={last_orders}
LastTurnFeedback={last_feedback}
Choose one abstract_action from: [HOLD, MOVE, ATTACK, SUPPORT_ATTACK, SUPPORT_DEFEND, RETREAT].
Return JSON: {"abstract_action":..., "rationale":...}
```

---

### Reflexion — Reflector

**触发条件**：SC 减少，或进攻类动作（ATTACK / SUPPORT_ATTACK / MOVE）未带来 gain 反馈。

**System Prompt**

```
You are a Reflector. Write one actionable lesson in one sentence.
```

**User Prompt（模板）**

```
You are {name}. Your last abstract_action was {last_action}.
Outcome feedback_label={feedback_label}, sc_delta={sc_delta}.
State summary: Phase={phase} Year={year} MyUnits={units} MySC={centers}
Other powers last orders: {last_orders}
Analyze why it failed (force, deception, timing) and output JSON: {"reflection": "..."}.
```

---

### LATS — Expand

对应论文 LATS \[Zhou et al., 2024\]。第一步：生成搜索根节点的候选动作集合。

**System Prompt**

```
You are a Diplomacy LATS planner.
Generate high-quality candidate actions for root expansion.
```

**User Prompt（模板）**

```
Power={name}
WorldModel={world_model_state}
Snapshot=Phase={phase} Year={year} MyUnits={units} MySC={centers}
Tension={tension}
LastOrders={last_orders}
LastFeedback={last_feedback}
PlannerNotes={planner_notes}

Action set: [HOLD, MOVE, ATTACK, SUPPORT_ATTACK, SUPPORT_DEFEND, RETREAT]
Return JSON with exactly this schema:
{"candidates": [{"action": "...", "thought": "short tactical intent", "prior": 0.0-1.0}]}.
Provide 3-5 diverse candidates only from action set.
```

---

### LATS — Simulate

第二步（每次 UCB 选节点后调用）：模拟一步后果并打价值/风险分，用于 MCTS 回传。回传公式：`reward = 0.55 × immediate + 0.45 × long_term − 0.35 × risk`。

**System Prompt**

```
You are a Diplomacy transition/value model for LATS.
Simulate one-step consequences and score utility.
```

**User Prompt（模板）**

```
Power={name}
CandidateAction={action}
Intent={thought}
WorldModel={world_model_state}
Snapshot=Phase={phase} Year={year} MyUnits={units} MySC={centers}
Tension={tension}
LastOrders={last_orders}
LastFeedback={last_feedback}

Return JSON with keys:
{"opponent_response": "...", "next_state_summary": "...",
 "immediate_value": 0.0-1.0, "long_term_value": 0.0-1.0, "risk": 0.0-1.0}.
```

---

### LATS — Post-Round Critic

每轮结算后更新 world model 状态和规划笔记（最多保留最近 6 条）。

**System Prompt**

```
You are a LATS critic. Produce concise planning improvements for next round.
```

**User Prompt（模板）**

```
Power={name}
LastAction={last_action}
Outcome={feedback_label}, sc_delta={sc_delta}
Snapshot=Phase={phase} Year={year} MyUnits={units} MySC={centers}
LastOrders={last_orders}
LastFeedback={last_feedback}
OldWorldModel={world_model_state}

Return JSON with keys:
{"planner_note": "one actionable lesson",
 "world_model_state": "updated concise strategic state"}.
```

---

### Hypothetical Minds — Decision

对应论文 Hypothetical Minds \[Cross et al., 2025\]。维护每个对手的 Theory-of-Mind 模型（目标 + 倾向 + 近期行动历史），对 2~3 个候选动作进行心智模拟后选出预期效用最高者。

**System Prompt**

```
You are a Diplomacy baseline agent implementing Hypothetical Minds.
You build Theory-of-Mind models of opponents and mentally simulate
their responses to your candidate actions before choosing.
```

**User Prompt（模板）**

```
You are {name}.
Phase={phase} Year={year}
MyUnits({unit_count})={units}
MySC({sc_count})={centers}
Tension={tension}
LastTurnOrders={last_orders}
LastTurnFeedback={last_feedback}

[Theory of Mind Models]
{tom_summary}

Step 1: Update your mental models of opponents (inferred goals and tendencies).
Step 2: Consider 2-3 candidate actions and mentally simulate opponent responses.
Step 3: Pick the action with the best expected outcome.

abstract_action must be one of: [HOLD, MOVE, ATTACK, SUPPORT_ATTACK, SUPPORT_DEFEND, RETREAT].
Return JSON:
{"tom_updates": [{"agent": "...", "inferred_goal": "...", "tendency": "..."}],
 "candidate_evaluations": [{"action": "...", "predicted_responses": {"country": "action"}, "expected_utility": 0.0}],
 "abstract_action": "...", "rationale": "..."}
```

> `tom_summary` 格式示例：
> ```
> France: recent=[MOVE, ATTACK], goal=expand, tendency=aggressive
> Germany: recent=[HOLD, HOLD], goal=unknown, tendency=defensive
> ```

---

## 说明

- 所有 JSON 输出均通过 `LLMAgent(json_format=True)` 强制要求，LLM 需严格返回合法 JSON。
- `{...}` 占位符在运行时由 Python 格式化替换为实际值。
- 花括号在实际 prompt 模板中使用 `{{...}}` 转义（Python f-string 规范），此处展示为字面含义。
- 提示词与论文描述的算法组件一一对应，修改提示词时请同步更新论文草稿。
