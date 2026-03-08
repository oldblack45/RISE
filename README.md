# RISE: From Hindsight to Foresight via Structured Cognitive Evolution in Multi-Agent Strategic Simulations

**RISE** 是一个用于评估基于 LLM 的智能体在复杂多智能体战略仿真中表现的统一实验框架。它实现了一种新颖的 **Observe-Orient-Decide-Act (OODA)** 认知循环与演化学习机制，使智能体能够将历史经验转化为前瞻性的战略洞察力。

## 核心架构

RISE Agent 采用四阶段闭环决策管道：

1. **World Model Construction（世界模型构建）** — 维护交互历史 $W_t = \{(a, f, r, e)\}^{t-1}$，以 Laplace 平滑初始化无信息先验
2. **Candidate Action Pruning（候选动作裁剪）** — LLM 引导的启发式过滤，压缩动作空间（$A_{\text{cand}} = \text{LLM}_{\text{filter}}(A_{\text{raw}}, W_t, G_{\text{meta}})$）
3. **Hypothetical Reasoning via BFS（假设推理）** — 分层 BFS 树展开 + Top-K 分支 + 期望最大值回传
4. **Dynamic Belief Calibration（动态信念校准）** — 执行后观测更新，语义摘要 + 频率校准

## 项目结构

```
RISE/
├── run_diplomacy.py                  # Diplomacy 锦标赛入口
├── main.py                           # 辅助入口
├── requirements.txt                  # Python 依赖
│
├── agents/                           # 智能体实现
│   ├── rise_agent.py                 # RISE Agent（OODA 认知循环 + BFS 期望最大值推理）
│   ├── diplomacy_baselines.py        # Diplomacy 基线 Agent（ReAct / Reflexion / EvoAgent / Hypothetical Minds）
│   ├── hypothetical_minds_agent.py   # 独立 Hypothetical Minds Agent（Theory of Mind 框架）
│   ├── ReActAgent.py                 # 独立 ReAct 推理 Agent
│   ├── EvoAgent.py                   # 独立进化策略 Agent
│
├── simulation/                       # 仿真场景与核心模型
│   ├── diplomacy/                    # Diplomacy 博弈锦标赛
│   │   └── tournament.py             #   锦标赛运行器（RISE vs 全部基线）
│   │
│   ├── SocialInvolution/             # 外卖骑手社会内卷仿真
│   │   ├── algorithm/                #   订单生成算法
│   │   │   ├── generate_orders.py
│   │   │   └── order_sequence.py
│   │   ├── config/                   #   骑手配置文件
│   │   └── entity/                   #   仿真实体
│   │       ├── city.py               #   城市地图
│   │       ├── meituan.py            #   平台系统
│   │       ├── merchant.py           #   商户
│   │       ├── order.py              #   订单
│   │       ├── rider.py              #   骑手
│   │       └── user.py               #   用户
│   │
│   └── models/                       # 共享模型组件
│       ├── agents/                   #   Agent 基础抽象
│       │   ├── LLMAgent.py           #   LLM 接口封装（OpenAI / DashScope / Ollama）
│       │   ├── GameAgent.py          #   游戏 Agent 基类
│       │   └── SociologyAgent.py     #   社会仿真 Agent 适配器（桥接多种决策框架 ↔ 骑手）
│       │
│       └── cognitive/                #   认知模型组件
│           ├── cognitive_agent.py    #   核心认知 Agent
│           ├── agent_profile.py      #   对手画像系统（四元组模型）
│           ├── hypothesis_reasoning.py  # 假设驱动推理
│           ├── world_cognition.py    #   世界状态认知（三元组模型）
│           ├── country_strategy.py   #   国家策略模板
│           ├── evaluation_system.py  #   多维度评估（EA / AS / SR / OM）
│           ├── experiment_logger.py  #   实验日志
│           └── realtime_hooks.py     #   实时钩子
│
├── visualize/                        # 可视化脚本
│   ├── delivery_rq2.py              #   外卖场景 RQ2 结果曲线
│   ├── diplomacy_rq2_plot.py        #   Diplomacy RQ2 预测精度演化
│   ├── bar_chart.py                  #   方法对比柱状图
│   ├── radar_chart.py                #   消融研究雷达图
│   ├── radar_2_chart.py              #   六边形消融可视化
│   ├── line_chart-evo.py             #   演化曲线
│   └── line_chart-history.py         #   历史曲线
│
└── experiments/                      # 自动生成的实验输出
    ├── diplomacy_tournament_*/       #   Diplomacy 锦标赛结果
    └── unified_comparison_*/         #   统一对比实验结果
```

---

## 实验场景

### 场景一：Diplomacy 博弈锦标赛

位于 `simulation/diplomacy/`

基于经典桌游 Diplomacy（无通信变体）。RISE Agent（扮演英国）与四种基线 Agent 在多局多轮博弈中竞争。

**基线方法：**

| 基线 | 核心机制 | 特点 |
|------|----------|------|
| **ReAct** | Reasoning + Acting 短上下文推理 | 战术敏锐但战略短视 |
| **Reflexion** | Actor + Reflector 反思学习 | 从失败中提取教训 |
| **EvoAgent** | 持续世界模型 + 叙事摘要 | 全局状态追踪 |
| **Hypothetical Minds** | Theory of Mind + 心智模拟 | 建模对手意图，模拟对手响应 |

**运行方式：**
```bash
python run_diplomacy.py
```

支持三种运行模式（通过修改 `RUN_MODE` 变量）：
- `RQ3`：单配置常规锦标赛
- `RQ3_MODELS`：多模型组对比（gpt-4o / gpt-5 / glm-4.5）
- `RQ4`：消融实验（Full / w/o Observe / w/o Orient / w/o Decide / w/o All）

**输出文件**（保存在 `experiments/diplomacy_tournament_*/`）：
- `RQ2_Evolution.csv`：逐轮预测精度
- `RQ3_Performance.csv`：对局结果与胜率
- `RQ4_Ablation.csv`：消融实验汇总
- `Turn_Log.csv`：逐回合详细日志

### 场景二：外卖骑手社会内卷仿真

位于 `simulation/SocialInvolution/`

模拟外卖平台中骑手的工时决策与接单策略，通过多种决策框架驱动骑手在竞争环境中的自适应博弈行为。

**支持的决策框架（通过 SociologyAgent 适配）：**

| 框架 | Mixin 类 | 说明 |
|------|----------|------|
| **RISE** | `RiderLLMAgent` | OODA 认知循环 + BFS 期望最大值推理 |
| **ReAct** | `RiderReActAgent` | 短上下文 LLM 推理 |
| **EvoAgent** | `RiderEvoAgent` | 进化策略种群 |
| **Hypothetical Minds** | `RiderHypotheticalMinds` | Theory of Mind 竞争建模 |
| **Greedy Heuristic** | `RiderGreedyHeuristic` | 贪婪启发式（无 LLM） |

骑手实体类 `Rider` 继承对应 Mixin 即可切换决策框架。

---

## 安装

### 环境要求
- Python 3.10+
- CUDA 兼容 GPU（推荐用于本地 LLM 推理）

### 安装步骤
```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### LLM 配置

通过环境变量配置 LLM 后端：

```bash
# DashScope (Qwen)
set DASHSCOPE_API_KEY=your_key
set DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# OpenAI 兼容端点
set OPENAI_API_KEY=your_key
set OPENAI_BASE_URL=http://localhost:8500/v1
```

---

## 快速开始

```bash
# Diplomacy 锦标赛（RQ2 & RQ3 & RQ4）
python run_diplomacy.py
```

---

## 基线方法详解

### RISE Agent（核心方法）

四阶段 OODA 闭环 + BFS 期望最大值搜索。维护世界模型 $W_t = \{(a, f, r, e)\}^{t-1}$，通过 Laplace 平滑概率分布和软匹配频率更新实现动态信念校准。

### Hypothetical Minds

基于 Theory of Mind 的决策框架。为每个对手构建心智模型（推断目标、策略倾向、行为模式），对候选动作进行对手响应的心智模拟，选择预期效用最高的动作。

### ReAct

Reasoning + Acting 范式。短上下文窗口（1-2 回合），Few-shot 引导的思考-行动循环。战术灵活但缺乏长期战略规划。

### Reflexion

Actor-Reflector 架构。Actor 产生动作，当触发条件命中（如 SC 减少、进攻失败）时，Reflector 生成教训存入长期记忆，用于后续决策参考。

### EvoAgent

进化策略框架。维护持续更新的世界模型叙事摘要，通过 Updater 更新全局状态、Planner 规划当轮动作。

### Greedy Heuristic

无 LLM 的纯启发式基线。始终选择最大化即时收益的行动，用于提供对比下界。

---

## 扩展指南

| 目标 | 方法 |
|------|------|
| 添加 Diplomacy 新基线 | 在 `agents/diplomacy_baselines.py` 中继承 `_LLMBaselineBase`，注册到 `tournament.py` 的 `BASELINE_TYPES` |
| 添加外卖场景新基线 | 在 `simulation/models/agents/SociologyAgent.py` 中实现 Rider Mixin |
| 添加新评估指标 | 扩展 `simulation/models/cognitive/evaluation_system.py` |
| 新仿真场景 | 在 `simulation/` 下创建新目录，实现 `ScenarioAdapter` |
| 自定义可视化 | 在 `visualize/` 中添加脚本 |

---