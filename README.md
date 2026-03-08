# RISE: From Hindsight to Foresight via Structured Cognitive Evolution in Multi-Agent Strategic Simulations

**RISE** 是一个用于评估基于 LLM 的智能体在复杂多智能体战略仿真中表现的统一实验框架。它实现了一种新颖的 **Observe-Orient-Decide-Act (OODA)** 认知循环与演化学习机制，使智能体能够将历史经验转化为前瞻性的战略洞察力。

## 核心架构

RISE Agent 采用四阶段闭环决策管道：

1. **World Model Construction（世界模型构建）** — 维护交互历史 W_t = {(a, f, r, e)}，以 Laplace 平滑初始化无信息先验
2. **Candidate Action Pruning（候选动作裁剪）** — LLM 引导的启发式过滤，压缩动作空间
3. **Hypothetical Reasoning via BFS（假设推理）** — 分层 BFS 树展开 + Top-K 分支 + 期望最大值回传
4. **Dynamic Belief Calibration（动态信念校准）** — 执行后观测更新，语义摘要 + 频率校准

## 项目结构

```
RISE/
├── run_diplomacy.py                  # Diplomacy 锦标赛入口（微观场景）
├── run_comparison.py                 # CMC 对比实验入口（宏观场景）
├── comparative_cognitive_world.py    # 多Agent方法统一对比框架
├── main.py                           # 辅助入口
├── requirements.txt                  # Python 依赖
│
├── agents/                           # 智能体实现
│   ├── rise_agent.py                 # RISE Agent（OODA 认知循环 + BFS 期望最大值推理）
│   ├── diplomacy_baselines.py        # Diplomacy 基线Agent（ReAct / Reflexion / EvoAgent）
│   ├── ReActAgent.py                 # 独立 ReAct 推理Agent
│   ├── EvoAgent.py                   # 独立演化Agent（Hypothetical Minds）
│   └── war_agent.py                  # WarAgent 框架（宏观博弈仿真）
│
├── simulation/                       # 仿真场景与核心模型
│   ├── diplomacy/                    # 📍 微观场景：Diplomacy 博弈锦标赛
│   │   └── tournament.py             #    锦标赛运行器（RISE vs 基线Agent）
│   │
│   ├── SocialInvolution/             # 📍 微观场景：外卖骑手平台仿真
│   │   ├── algorithm/                #    订单生成算法
│   │   │   ├── generate_orders.py
│   │   │   └── order_sequence.py
│   │   ├── config/                   #    骑手配置文件
│   │   └── entity/                   #    仿真实体
│   │       ├── city.py               #    城市
│   │       ├── meituan.py            #    平台
│   │       ├── merchant.py           #    商户
│   │       ├── order.py              #    订单
│   │       ├── rider.py              #    骑手
│   │       └── user.py               #    用户
│   │
│   └── models/                       # 共享模型组件
│       ├── agents/                   #    Agent 基础抽象
│       │   ├── LLMAgent.py           #    LLM 接口封装
│       │   ├── GameAgent.py          #    游戏Agent基类
│       │   ├── SociologyAgent.py     #    社会仿真Agent适配器（桥接 RISE ↔ 骑手）
│       │   └── SecretaryAgent.py     #    秘书Agent辅助类
│       │
│       └── cognitive/                #    认知模型组件
│           ├── cognitive_agent.py    #    核心认知Agent
│           ├── agent_profile.py      #    对手画像系统
│           ├── hypothesis_reasoning.py  # 假设驱动推理
│           ├── learning_system.py    #    经验学习系统
│           ├── evaluation_system.py  #    多维度评估
│           ├── experiment_logger.py  #    实验日志
│           ├── world_cognition.py    #    世界状态认知
│           ├── country_strategy.py   #    国家策略模板
│           ├── prompt_utils.py       #    提示词工具
│           └── realtime_hooks.py     #    实时钩子
│
├── visualize/                        # 可视化脚本
│   ├── diplomacy_rq2_plot.py         # RQ2：预测精度演化曲线
│   ├── delivery_rq2.py              # RQ2：外卖场景结果
│   ├── bar_chart.py                  # 方法对比柱状图
│   ├── radar_chart.py                # 消融研究雷达图
│   ├── radar_2_chart.py              # 六边形消融可视化
│   ├── line_chart-evo.py             # 演化曲线
│   └── line_chart-history.py         # 历史曲线
│
└── experiments/                      # 自动生成的实验输出
    ├── diplomacy_tournament_*/       # Diplomacy 锦标赛结果
    └── unified_comparison_*/         # CMC 对比实验结果
```

---

## 实验场景

### 1. 微观场景：Diplomacy 博弈锦标赛

位于 `simulation/diplomacy/`

基于经典桌游 Diplomacy（无通信变体）。RISE Agent（扮演英国）与多种基线Agent（ReAct、Reflexion、EvoAgent）在多局多轮博弈中竞争。

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

### 2. 微观场景：外卖骑手社会内卷仿真

位于 `simulation/SocialInvolution/`

模拟外卖平台中骑手的工时决策与接单策略，通过 RISE Agent 驱动骑手在竞争环境中的自适应博弈行为。

### 3. 宏观场景：Agent 方法对比实验（CMC 等）

通过 `comparative_cognitive_world.py` 提供统一的多Agent方法对比框架，支持多种认知策略的评估。

**运行方式：**
```bash
python run_comparison.py
```

交互式菜单：
1. 快速对比测试（全部方法）
2. 单方法测试
3. 统一对比测试
4. 认知模型独立测试
5. 消融对比
6. 四策略组对比

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

# 多Agent方法对比实验
python run_comparison.py
```

---

## 扩展 RISE

| 目标 | 方法 |
|------|------|
| 添加新Agent | 在 `agents/` 中实现，在对应场景中注册 |
| 添加新评估指标 | 扩展 `evaluation_system.py` |
| 新仿真场景 | 在 `simulation/` 下创建新目录，实现世界逻辑 |
| 自定义可视化 | 在 `visualize/` 中添加脚本 |

---