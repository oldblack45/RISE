"""
支持多种Agent方法的认知世界比较框架
复用现有的模拟系统，支持Hypothetical Minds、Werewolf、WarAgent等方法的比较评估
"""

from datetime import datetime
import time
import os
import json
from typing import Dict, List, Optional
from pathlib import Path

# 导入现有的模拟系统组件
from simulation.models.cognitive.experiment_logger import init_logger
from simulation.examples.PowerGameWorld.entity.rule_based_systems import (
    RuleBasedAttributeAdjuster, RuleBasedScoreCalculator,
    WorldFeedbackSystem, StructuredWorldMemory
)

# 导入我们新创建的Agent类
from agents.EvoAgent import HypotheticalMindsAgent
from agents.ReActAgent import WerewolfCountryAgent
from agents.war_agent import (
    WarCountryAgent, Board, create_america_profile, create_soviet_profile
)

# 导入原有的认知Agent作为基准
try:
    from simulation.examples.PowerGameWorld.entity.cognitive_world import CognitiveCountryAgent
    ORIGINAL_AGENT_AVAILABLE = True
except ImportError:
    ORIGINAL_AGENT_AVAILABLE = False
    CognitiveCountryAgent = None
    print("Warning: 原始CognitiveCountryAgent不可用，将跳过该方法")


class ComparativeCognitiveWorld:
    """支持多种Agent方法的比较实验世界"""

    # 支持的Agent类型
    AGENT_TYPES = {
        "cognitive": "认知增强Agent (MAGES)",
        "hm": "Hypothetical Minds (心智理论框架)",
        "werewolf": "Werewolf-Inspired Agent",
        "war": "WarAgent Framework (宏观历史模拟)"
    }

    def __init__(self, agent_type: str = "cognitive", use_rule_based: bool = True,
                 experiment_name: Optional[str] = None, base_dir: str = "experiments"):
        """
        初始化比较实验世界

        Args:
            agent_type: Agent类型 ("cognitive", "cot", "react", "werewolf", "war")
            use_rule_based: 是否使用规则式系统
            experiment_name: 实验名称
            base_dir: 实验基础目录
        """
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(f"不支持的Agent类型: {agent_type}. 支持的类型: {list(self.AGENT_TYPES.keys())}")

        if agent_type == "cognitive" and not ORIGINAL_AGENT_AVAILABLE:
            raise ValueError("原始认知Agent不可用，请选择其他类型")

        self.agent_type = agent_type
        self.use_rule_based = use_rule_based

        # 实验配置
        if not experiment_name:
            date_str = datetime.now().strftime("%m%d_%H%M")
            experiment_name = f"{agent_type}_comparison_{date_str}"
        self.experiment_name = experiment_name

        # 初始化实验日志系统
        self.experiment_logger = init_logger(
            experiment_name=experiment_name,
            base_dir=base_dir
        )

        # WarAgent特有组件
        if agent_type == "war":
            self.board = Board()
            self._initialize_war_board()
        else:
            self.board = None

        # 博弈环境初始化
        self.step = 1
        self.exit_game = False
        self.world_memory = None
        self.current_tension = 50.0
        self.last_scores = (None, None, None, None)

        # 初始化Agent
        self._initialize_agents()

        # 初始化世界系统
        self._initialize_world_systems()

        # 学习统计
        self.learning_stats = {"america": {}, "soviet": {}}

    def _initialize_war_board(self):
        """初始化WarAgent的看板系统（古巴导弹危机背景）"""
        self.board.add_hostility("美国", "苏联")
        self.board.add_treaty("联合国宪章")
        self.board.add_treaty("日内瓦公约")
        self.board.add_international_event("1962年10月，美国发现苏联在古巴部署导弹")
        self.board.add_international_event("肯尼迪总统宣布对古巴进行海上封锁")
        self.board.add_international_event("国际社会高度关注，担心核战争爆发")

    def _initialize_agents(self):
        """初始化Agent"""
        other_countries = ["国家A", "国家B"]
        game_attributes = {"military_strength": 80, "economic_power": 75, "diplomatic_influence": 70}

        if self.agent_type == "cognitive" and ORIGINAL_AGENT_AVAILABLE:
            self.america = CognitiveCountryAgent(
                "国家A", ["国家B"], game_attributes.copy(), self.experiment_logger
            )
            self.soviet_union = CognitiveCountryAgent(
                "国家B", ["国家A"], game_attributes.copy(), self.experiment_logger
            )
        elif self.agent_type == "hm":
            self.america = HypotheticalMindsAgent(
                "国家A", ["国家B"], game_attributes.copy(), self.experiment_logger
            )
            self.soviet_union = HypotheticalMindsAgent(
                "国家B", ["国家A"], game_attributes.copy(), self.experiment_logger
            )
        elif self.agent_type == "werewolf":
            self.america = WerewolfCountryAgent(
                "国家A", ["国家B"], game_attributes.copy(), self.experiment_logger
            )
            self.soviet_union = WerewolfCountryAgent(
                "国家B", ["国家A"], game_attributes.copy(), self.experiment_logger
            )
        elif self.agent_type == "war":
            america_profile = create_america_profile()
            soviet_profile = create_soviet_profile()

            if not hasattr(self, 'board') or self.board is None:
                self.board = Board()
                self._initialize_war_board()

            self.america = WarCountryAgent(
                "美国", america_profile, ["苏联"], game_attributes.copy(), self.experiment_logger
            )
            self.soviet_union = WarCountryAgent(
                "苏联", soviet_profile, ["美国"], game_attributes.copy(), self.experiment_logger
            )

            self.america.set_board(self.board)
            self.soviet_union.set_board(self.board)

            self.experiment_logger.log_print("初始化WarAgent框架 - 古巴导弹危机场景", level="INFO")

    def _initialize_world_systems(self):
        """初始化世界系统"""
        if self.use_rule_based:
            self.score_calculator = RuleBasedScoreCalculator()
            self.attribute_adjuster = RuleBasedAttributeAdjuster()
            self.feedback_system = WorldFeedbackSystem()
            self.structured_memory = StructuredWorldMemory()
            self.experiment_logger.log_print("初始化规则式世界系统", level="INFO")

    def run_one_step(self):
        """执行一步仿真"""
        self.experiment_logger.log_print(f"=== Step {self.step} ===", level="INFO")

        world_info = self._get_world_information()

        # WarAgent需要传入board参数
        if self.agent_type == "war":
            america_decision = self.america.cognitive_game_decide(world_info, self.board)
            soviet_decision = self.soviet_union.cognitive_game_decide(world_info, self.board)
        else:
            america_decision = self.america.cognitive_game_decide(world_info)
            soviet_decision = self.soviet_union.cognitive_game_decide(world_info)

        # 生成世界反馈
        america_feedback = ""
        soviet_feedback = ""
        if self.use_rule_based:
            america_feedback = self._generate_rule_based_feedback(america_decision["action"])
            soviet_feedback = self._generate_rule_based_feedback(soviet_decision["action"])

        america_decision["world_feedback"] = america_feedback
        soviet_decision["world_feedback"] = soviet_feedback

        self._record_for_evaluation(america_decision, soviet_decision)

        # WarAgent不需要学习系统
        if self.agent_type != "war":
            self._agent_learning(america_decision, soviet_decision)

        self._update_world_state(america_decision, soviet_decision)
        self._check_game_end()
        self._print_step_info(america_decision, soviet_decision)

        self.step += 1

    def _get_world_information(self) -> str:
        """获取当前世界信息"""
        tension_desc = "低" if self.current_tension < 30 else ("中" if self.current_tension < 70 else "高")

        world_info = f"""当前世界形势 (第{self.step}轮):
- 地区紧张程度: {tension_desc} ({self.current_tension:.1f})
- 国际关注度: {"高" if self.step > 5 else "中"}
- 经济影响: {"显著" if self.current_tension > 60 else "轻微"}"""

        if self.step > 1:
            america_last_action = self.america.action[-1] if self.america.action else "无"
            soviet_last_action = self.soviet_union.action[-1] if self.soviet_union.action else "无"
            world_info += f"\n- 上轮行动: 国家A={america_last_action}, 国家B={soviet_last_action}"

        return world_info.strip()

    def _generate_rule_based_feedback(self, action: str) -> str:
        """生成规则式反馈"""
        feedback_templates = {
            "外交谈判": "短期效果: 国际社会表示欢迎，紧张局势有所缓解; 长期影响: 持续对话建立互信",
            "和平协议": "短期效果: 和平协议签署引起国际社会广泛赞誉; 长期影响: 和平红利逐渐释放",
            "军事演习": "短期效果: 军事演习引起对手警觉; 长期影响: 可能触发军备竞赛",
            "经济制裁": "短期效果: 经济制裁措施开始实施; 长期影响: 制裁效应逐渐显现",
            "撤回行动": "短期效果: 撤回行动被视为缓和信号; 长期影响: 善意举措促进对话",
            "区域封锁": "短期效果: 区域封锁严重影响贸易流通; 长期影响: 导致经济损失扩大",
            "武器部署": "短期效果: 武器部署引发强烈反应; 长期影响: 军备竞赛升级",
            "最后通牒": "短期效果: 最后通牒引发严重危机; 长期影响: 极端紧张导致冲突升级",
            "情报侦察": "短期效果: 情报活动引起外交抗议; 长期影响: 持续情报收集提供支持"
        }
        return feedback_templates.get(action, "短期效果: 行动产生了一定影响; 长期影响: 各方正在观察")

    def _record_for_evaluation(self, america_decision: Dict, soviet_decision: Dict):
        """记录到evaluation系统"""
        self.experiment_logger.log_evaluation_round(
            round_num=self.step * 2,
            actor="国家A",
            declaration=america_decision["declaration"],
            action=america_decision["action"],
            world_feedback=america_decision["world_feedback"],
            timestamp=f"t+{(self.step - 1) * 10}"
        )

        self.experiment_logger.log_evaluation_round(
            round_num=self.step * 2 - 1,
            actor="国家B",
            declaration=soviet_decision["declaration"],
            action=soviet_decision["action"],
            world_feedback=soviet_decision["world_feedback"],
            timestamp=f"t+{(self.step - 1) * 10 + 5}"
        )

    def _agent_learning(self, america_decision: Dict, soviet_decision: Dict):
        """Agent学习过程"""
        try:
            self.america.learn_from_interaction(
                america_decision["action"],
                america_decision["world_feedback"],
                {"国家B": soviet_decision["action"]},
                self.world_memory
            )

            self.soviet_union.learn_from_interaction(
                soviet_decision["action"],
                soviet_decision["world_feedback"],
                {"国家A": america_decision["action"]},
                self.world_memory
            )
        except Exception as e:
            self.experiment_logger.log_print(f"Agent学习过程出错: {e}", level="WARNING")

    def _update_world_state(self, america_decision: Dict, soviet_decision: Dict):
        """更新世界状态"""
        tension_changes = {
            "外交谈判": -5, "和平协议": -10, "撤回行动": -8,
            "情报侦察": 2, "公开声明": 1,
            "军事演习": 5, "经济制裁": 6, "区域封锁": 8,
            "武器部署": 10, "最后通牒": 12, "宣战": 15, "核打击": 20
        }

        america_change = tension_changes.get(america_decision["action"], 0)
        soviet_change = tension_changes.get(soviet_decision["action"], 0)

        self.current_tension += (america_change + soviet_change)
        self.current_tension = max(0.0, min(100.0, self.current_tension))

        if self.use_rule_based and hasattr(self, 'structured_memory'):
            world_feedback = f"紧张度: {self.current_tension:.1f}/100"
            self.structured_memory.add_round_memory(
                round_num=self.step,
                america_action=america_decision.get("action", "未知行动"),
                soviet_action=soviet_decision.get("action", "未知行动"),
                america_declaration=america_decision.get("declaration", ""),
                soviet_declaration=soviet_decision.get("declaration", ""),
                world_feedback=world_feedback
            )

    def _check_game_end(self):
        """检查游戏结束条件"""
        if self.current_tension >= 95:
            self.exit_game = True
            self.experiment_logger.log_print("紧张程度达到极限，博弈结束", level="WARNING")
        elif self.current_tension <= 5:
            self.exit_game = True
            self.experiment_logger.log_print("局势完全缓解，博弈结束", level="INFO")
        elif self.step >= 15:
            self.exit_game = True
            self.experiment_logger.log_print("达到最大轮数，博弈结束", level="INFO")

    def _print_step_info(self, america_decision: Dict, soviet_decision: Dict):
        """打印轮次信息"""
        self.experiment_logger.log_print(
            f"国家A: {america_decision['action']} - {america_decision['declaration'][:50]}...",
            level="INFO"
        )
        self.experiment_logger.log_print(
            f"国家B: {soviet_decision['action']} - {soviet_decision['declaration'][:50]}...",
            level="INFO"
        )
        self.experiment_logger.log_print(f"当前紧张度: {self.current_tension:.1f}", level="INFO")
        self.experiment_logger.log_print("=" * 80, level="INFO")

    def start_sim(self, max_steps: int = 8):
        """开始仿真"""
        method_name = self.AGENT_TYPES[self.agent_type]
        system_type = "规则式" if self.use_rule_based else "LLM"

        self.experiment_logger.log_print(
            f"开始 {method_name} 博弈仿真 ({system_type}系统)...",
            level="INFO"
        )

        self.step = 1
        self.exit_game = False
        self.current_tension = 50.0

        for i in range(max_steps):
            self.run_one_step()
            if self.exit_game:
                self.experiment_logger.log_print(f"博弈在第{self.step-1}步结束", level="INFO")
                break

        self._generate_final_report()
        summary_file = self.experiment_logger.finalize_experiment()
        self.experiment_logger.log_print(f"实验日志已保存到: {summary_file}", level="INFO")

        return summary_file

    def run_final_evaluation(self, weights: Optional[Dict[str, float]] = None):
        """运行最终评测"""
        self.experiment_logger.log_print("开始运行最终评测...", level="INFO")

        try:
            if hasattr(self, 'structured_memory') and self.structured_memory:
                result = self.experiment_logger.run_evaluation(
                    structured_memory_data=self.structured_memory.memory_data,
                    weights=weights
                )
            else:
                result = self.experiment_logger.run_evaluation(weights=weights)

            if result:
                self.experiment_logger.log_print(
                    f"评测完成! 最终得分: {result.final_score:.3f}",
                    level="INFO"
                )
                self._save_evaluation_result(result)
                return result
            else:
                self.experiment_logger.log_print("评测失败", level="WARNING")
                return None

        except Exception as e:
            self.experiment_logger.log_print(f"评测过程出错: {e}", level="ERROR")
            return None

    def _save_evaluation_result(self, result):
        """保存评测结果"""
        eval_result = {
            "method": self.agent_type,
            "agent_type_description": self.AGENT_TYPES[self.agent_type],
            "scores": {
                "ea_score": float(result.ea_score),
                "as_score": float(result.as_score),
                "sr_score": float(result.sr_score),
                "om_score": float(result.om_score),
                "final_score": float(result.final_score)
            },
            "experiment_info": {
                "use_rule_based": self.use_rule_based,
                "total_steps": self.step - 1,
                "final_tension": self.current_tension
            }
        }

        eval_dir = Path(self.experiment_logger.experiment_dir) / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        eval_file = eval_dir / "evaluation_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)

        self.experiment_logger.log_print(f"评测结果已保存到: {eval_file}", level="INFO")

    def _generate_final_report(self):
        """生成最终报告"""
        self.experiment_logger.log_print("="*60, level="INFO")
        self.experiment_logger.log_print(f"{self.AGENT_TYPES[self.agent_type]} 最终报告", level="INFO")
        self.experiment_logger.log_print("="*60, level="INFO")

        america_stats = self.america.get_cognition_statistics()
        soviet_stats = self.soviet_union.get_cognition_statistics()

        self.experiment_logger.log_print(f"国家A认知统计: {america_stats}", level="INFO")
        self.experiment_logger.log_print(f"国家B认知统计: {soviet_stats}", level="INFO")

        self.america.export_cognition_report()
        self.soviet_union.export_cognition_report()

        final_summary = {
            "method": self.agent_type,
            "method_description": self.AGENT_TYPES[self.agent_type],
            "game_completed": True,
            "total_steps": self.step - 1,
            "final_tension": self.current_tension,
            "use_rule_based": self.use_rule_based,
            "america_stats": america_stats,
            "soviet_stats": soviet_stats
        }

        self.experiment_logger.save_experiment_summary(final_summary)


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 单个方法测试")
    print("2. WarAgent vs MAGES 对比测试")

    choice = input("请选择 (1/2, 默认2): ").strip() or "2"

    if choice == "1":
        available_methods = list(ComparativeCognitiveWorld.AGENT_TYPES.keys())
        if not ORIGINAL_AGENT_AVAILABLE:
            available_methods.remove("cognitive")

        print(f"可用的方法: {available_methods}")
        method = input(f"请选择方法 (默认war): ").strip() or "war"

        if method not in available_methods:
            print(f"无效方法，使用默认的war")
            method = "war"

        world = ComparativeCognitiveWorld(
            agent_type=method,
            use_rule_based=True,
            experiment_name=f"{method}_test_{int(time.time())}"
        )

        world.start_sim(max_steps=8)
        world.run_final_evaluation()
    else:
        # WarAgent vs MAGES对比
        print("\n开始 WarAgent vs MAGES 对比实验...")

        methods_to_test = ["war"]
        if ORIGINAL_AGENT_AVAILABLE:
            methods_to_test.append("cognitive")

        for method in methods_to_test:
            print(f"\n{'='*60}")
            print(f"测试方法: {ComparativeCognitiveWorld.AGENT_TYPES[method]}")
            print(f"{'='*60}")

            world = ComparativeCognitiveWorld(
                agent_type=method,
                use_rule_based=True,
                experiment_name=f"{method}_comparison_{int(time.time())}"
            )

            world.start_sim(max_steps=8)
            result = world.run_final_evaluation()

            if result:
                print(f"\n{method} 最终得分: {result.final_score:.3f}")
