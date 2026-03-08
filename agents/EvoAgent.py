"""
EvoAgent Implementation
基于进化算法的Agent决策方法

核心机制：
1. 策略种群（Strategy Population）：维护多个候选策略
2. 适应度评估（Fitness Evaluation）：基于历史效果评估策略质量
3. 选择（Selection）：选择高适应度策略
4. 变异（Mutation）：通过LLM生成策略变体
5. 交叉（Crossover）：融合优势策略特征
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class Strategy:
    """策略个体类"""
    
    def __init__(self, strategy_id: int, description: str, 
                 action_preferences: Dict[str, float], generation: int = 0):
        self.strategy_id = strategy_id
        self.description = description  # 策略的自然语言描述
        self.action_preferences = action_preferences  # 行动偏好权重
        self.fitness = 0.5  # 适应度分数 [0, 1]
        self.generation = generation  # 所属代数
        self.usage_count = 0  # 使用次数
        self.success_count = 0  # 成功次数
        self.history: List[Dict] = []  # 使用历史
    
    def update_fitness(self, reward: float):
        """更新适应度"""
        self.usage_count += 1
        if reward > 0:
            self.success_count += 1
        # 指数移动平均更新适应度
        alpha = 0.3
        self.fitness = alpha * reward + (1 - alpha) * self.fitness
        self.fitness = max(0.0, min(1.0, self.fitness))
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'description': self.description,
            'action_preferences': self.action_preferences,
            'fitness': self.fitness,
            'generation': self.generation,
            'usage_count': self.usage_count,
            'success_rate': self.get_success_rate()
        }


class EvolutionEngine:
    """进化引擎 - EvoAgent的核心"""
    
    def __init__(self, agent_name: str, llm_agent: LLMAgent,
                 population_size: int = 5, mutation_rate: float = 0.3):
        self.agent_name = agent_name
        self.llm_agent = llm_agent
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        self.population: List[Strategy] = []
        self.strategy_counter = 0
        self.generation = 0
        self.evolution_history: List[Dict] = []
        
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]
    
    def initialize_population(self):
        """初始化策略种群"""
        initial_strategies = [
            ("保守防御", {"外交谈判": 0.3, "和平协议": 0.3, "撤回行动": 0.2, "情报侦察": 0.2}),
            ("积极进攻", {"军事演习": 0.3, "武器部署": 0.3, "区域封锁": 0.2, "最后通牒": 0.2}),
            ("均衡策略", {"外交谈判": 0.2, "军事演习": 0.2, "情报侦察": 0.2, "经济制裁": 0.2, "和平协议": 0.2}),
            ("威慑策略", {"武器部署": 0.4, "军事演习": 0.3, "外交谈判": 0.3}),
            ("合作主导", {"和平协议": 0.4, "外交谈判": 0.4, "撤回行动": 0.2})
        ]
        
        for desc, prefs in initial_strategies[:self.population_size]:
            self.strategy_counter += 1
            strategy = Strategy(
                strategy_id=self.strategy_counter,
                description=desc,
                action_preferences=prefs,
                generation=0
            )
            self.population.append(strategy)
    
    def select_strategy(self, world_info: str) -> Strategy:
        """选择策略 - 使用轮盘赌选择"""
        if not self.population:
            self.initialize_population()
        
        # 计算选择概率（基于适应度）
        total_fitness = sum(s.fitness for s in self.population)
        if total_fitness == 0:
            return random.choice(self.population)
        
        # 轮盘赌选择
        r = random.random() * total_fitness
        cumulative = 0
        for strategy in self.population:
            cumulative += strategy.fitness
            if cumulative >= r:
                return strategy
        
        return self.population[-1]
    
    def mutate_strategy(self, parent: Strategy, world_info: str) -> Optional[Strategy]:
        """变异策略 - 使用LLM生成变体"""
        
        prompt = f"""基于以下父策略，生成一个变异后的新策略：

父策略描述：{parent.description}
父策略行动偏好：{json.dumps(parent.action_preferences, ensure_ascii=False)}
父策略适应度：{parent.fitness:.2f}

当前局势信息：
{world_info[:300]}

可选行动：{', '.join(self.available_actions)}

请生成一个改进的策略变体，返回JSON格式：
{{
    "description": "新策略的描述",
    "action_preferences": {{"行动1": 权重, "行动2": 权重, ...}},
    "mutation_reason": "变异原因"
}}

注意：权重之和应为1.0
"""
        
        try:
            response = self.llm_agent.get_response(prompt, flag_debug_print=False)
            
            if isinstance(response, dict):
                self.strategy_counter += 1
                new_strategy = Strategy(
                    strategy_id=self.strategy_counter,
                    description=response.get('description', f'{parent.description}_变异'),
                    action_preferences=response.get('action_preferences', parent.action_preferences.copy()),
                    generation=self.generation
                )
                # 继承部分父代适应度
                new_strategy.fitness = parent.fitness * 0.8
                return new_strategy
        except Exception as e:
            print(f"[EvoAgent] 变异失败: {e}")
        
        return None
    
    def crossover_strategies(self, parent1: Strategy, parent2: Strategy) -> Optional[Strategy]:
        """交叉策略 - 融合两个策略"""
        
        prompt = f"""融合以下两个策略，生成一个新的混合策略：

策略1：{parent1.description}
行动偏好1：{json.dumps(parent1.action_preferences, ensure_ascii=False)}
适应度1：{parent1.fitness:.2f}

策略2：{parent2.description}
行动偏好2：{json.dumps(parent2.action_preferences, ensure_ascii=False)}
适应度2：{parent2.fitness:.2f}

可选行动：{', '.join(self.available_actions)}

请生成融合后的策略，返回JSON格式：
{{
    "description": "融合策略的描述",
    "action_preferences": {{"行动1": 权重, "行动2": 权重, ...}},
    "crossover_insight": "融合思路"
}}

注意：权重之和应为1.0
"""
        
        try:
            response = self.llm_agent.get_response(prompt, flag_debug_print=False)
            
            if isinstance(response, dict):
                self.strategy_counter += 1
                new_strategy = Strategy(
                    strategy_id=self.strategy_counter,
                    description=response.get('description', '融合策略'),
                    action_preferences=response.get('action_preferences', {}),
                    generation=self.generation
                )
                # 继承父代平均适应度
                new_strategy.fitness = (parent1.fitness + parent2.fitness) / 2
                return new_strategy
        except Exception as e:
            print(f"[EvoAgent] 交叉失败: {e}")
        
        return None
    
    def evolve_population(self, world_info: str):
        """进化种群 - 一轮进化"""
        self.generation += 1
        
        # 按适应度排序
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        # 精英保留：保留前50%
        elite_count = max(1, self.population_size // 2)
        new_population = self.population[:elite_count]
        
        # 生成新个体填充种群
        while len(new_population) < self.population_size:
            if random.random() < self.mutation_rate and self.population:
                # 变异
                parent = random.choice(self.population[:elite_count])
                child = self.mutate_strategy(parent, world_info)
                if child:
                    new_population.append(child)
            elif len(self.population) >= 2:
                # 交叉
                parent1, parent2 = random.sample(self.population[:elite_count], 2)
                child = self.crossover_strategies(parent1, parent2)
                if child:
                    new_population.append(child)
        
        self.population = new_population[:self.population_size]
        
        # 记录进化历史
        self.evolution_history.append({
            'generation': self.generation,
            'avg_fitness': sum(s.fitness for s in self.population) / len(self.population),
            'best_fitness': max(s.fitness for s in self.population),
            'population_size': len(self.population)
        })
    
    def update_strategy_fitness(self, strategy: Strategy, reward: float):
        """更新策略适应度"""
        strategy.update_fitness(reward)
    
    def get_best_strategy(self) -> Optional[Strategy]:
        """获取最佳策略"""
        if not self.population:
            return None
        return max(self.population, key=lambda s: s.fitness)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取进化统计"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'total_strategies_created': self.strategy_counter,
            'avg_fitness': sum(s.fitness for s in self.population) / max(1, len(self.population)),
            'best_strategy': self.get_best_strategy().to_dict() if self.population else None,
            'evolution_history_len': len(self.evolution_history)
        }


class EvoAgent(LLMAgent):
    """EvoAgent - 基于进化算法的国家智能体"""
    
    def __init__(self, country_name: str, other_countries: List[str],
                 game_attributes: Dict[str, int], experiment_logger: ExperimentLogger,
                 population_size: int = 5, evolution_interval: int = 3):
        
        super().__init__(
            agent_name=f"Evo_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-max',
            online_track=False,
            json_format=True
        )
        
        self.country_name = country_name
        self.other_countries = other_countries
        self.game_attributes = game_attributes.copy()
        self.experiment_logger = experiment_logger
        
        # 核心：进化引擎
        self.evolution_engine = EvolutionEngine(
            country_name, self, 
            population_size=population_size
        )
        
        # 决策记录
        self.action_history = []
        self.declaration_history = []
        self.round_counter = 0
        self.evolution_interval = evolution_interval
        self.current_strategy: Optional[Strategy] = None
        self.last_reward = 0.0
        
        # 行动空间
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]
    
    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """基于进化策略的决策流程"""
        
        self.round_counter += 1
        
        # 初始化种群（如果需要）
        if not self.evolution_engine.population:
            self.evolution_engine.initialize_population()
        
        # 定期进化
        if self.round_counter > 1 and self.round_counter % self.evolution_interval == 0:
            self.evolution_engine.evolve_population(world_info)
        
        # 选择策略
        self.current_strategy = self.evolution_engine.select_strategy(world_info)
        
        # 基于策略选择行动
        action, reasoning = self._execute_strategy(world_info, self.current_strategy)
        
        # 生成声明
        declaration = self._generate_declaration(action, reasoning, world_info)
        
        # 记录
        self.action_history.append(action)
        self.declaration_history.append(declaration)
        
        return {
            'action': action,
            'declaration': declaration,
            'reasoning_result': {
                'evolution_stats': self.evolution_engine.get_statistics(),
                'current_strategy': self.current_strategy.to_dict() if self.current_strategy else None,
                'reasoning': reasoning,
                'method': 'evo_agent',
                'final_satisfaction_score': self.current_strategy.fitness if self.current_strategy else 0.5,
                'reasoning_depth': self.evolution_engine.generation
            },
            'satisfaction_score': self.current_strategy.fitness if self.current_strategy else 0.5,
            'reasoning_depth': self.evolution_engine.generation
        }
    
    def _execute_strategy(self, world_info: str, strategy: Strategy) -> Tuple[str, str]:
        """执行策略选择行动"""
        
        prompt = f"""基于以下策略和当前局势，选择最合适的行动：

策略描述：{strategy.description}
策略偏好：{json.dumps(strategy.action_preferences, ensure_ascii=False)}
策略适应度：{strategy.fitness:.2f}

当前局势：
{world_info[:500]}

可选行动：{', '.join(self.available_actions)}

历史行动：{', '.join(self.action_history[-3:]) if self.action_history else '无'}

请选择行动并说明理由，返回JSON格式：
{{
    "chosen_action": "从可选行动中选择",
    "reasoning": "选择理由",
    "confidence": 0.0-1.0的置信度
}}
"""
        
        try:
            response = self.get_response(prompt, flag_debug_print=False)
            
            if isinstance(response, dict):
                action = response.get('chosen_action', '外交谈判')
                reasoning = response.get('reasoning', '基于当前策略判断')
                
                if action not in self.available_actions:
                    # 根据策略偏好选择
                    action = self._sample_from_preferences(strategy.action_preferences)
                
                return action, reasoning
        except Exception as e:
            self.experiment_logger.log_print(f"EvoAgent执行策略失败: {e}", level="WARNING")
        
        # 备选：根据偏好采样
        action = self._sample_from_preferences(strategy.action_preferences)
        return action, '基于策略偏好选择'
    
    def _sample_from_preferences(self, preferences: Dict[str, float]) -> str:
        """根据偏好权重采样行动"""
        if not preferences:
            return random.choice(self.available_actions)
        
        actions = list(preferences.keys())
        weights = list(preferences.values())
        total = sum(weights)
        
        if total == 0:
            return random.choice(actions) if actions else self.available_actions[0]
        
        r = random.random() * total
        cumulative = 0
        for action, weight in zip(actions, weights):
            cumulative += weight
            if cumulative >= r:
                return action
        
        return actions[-1] if actions else self.available_actions[0]
    
    def _generate_declaration(self, action: str, reasoning: str, world_info: str) -> str:
        """生成行动声明"""
        
        prompt = f"""为以下行动生成简短的外交声明：

行动：{action}
理由：{reasoning}

要求：简洁有力，1-2句话。返回JSON格式：
{{"declaration": "声明内容"}}
"""
        
        try:
            response = self.get_response(prompt, flag_debug_print=False)
            
            if isinstance(response, dict):
                return response.get('declaration', f'{self.country_name}决定{action}')
            return f'{self.country_name}决定{action}'
        except Exception as e:
            self.experiment_logger.log_print(f"EvoAgent生成声明失败: {e}", level="WARNING")
            return f'{self.country_name}决定{action}，以维护国家利益。'
    
    def learn_from_interaction(self, own_action: str, world_feedback: str,
                               other_actions: Dict[str, str], world_memory: Any = None):
        """从交互中学习 - 更新策略适应度"""
        
        # 计算奖励
        reward = self._calculate_reward(world_feedback)
        self.last_reward = reward
        
        # 更新当前策略的适应度
        if self.current_strategy:
            self.evolution_engine.update_strategy_fitness(self.current_strategy, reward)
            self.current_strategy.history.append({
                'action': own_action,
                'feedback': world_feedback[:100],
                'reward': reward
            })
    
    def _calculate_reward(self, feedback: str) -> float:
        """计算奖励信号"""
        reward = 0.5  # 基础奖励
        
        feedback_lower = feedback.lower()
        
        # 正面反馈
        positive_keywords = ['成功', '缓和', '合作', '支持', '胜利', '优势', '收益', '稳定']
        for kw in positive_keywords:
            if kw in feedback_lower:
                reward += 0.1
        
        # 负面反馈
        negative_keywords = ['失败', '紧张', '对抗', '损失', '危机', '风险', '威胁', '冲突']
        for kw in negative_keywords:
            if kw in feedback_lower:
                reward -= 0.1
        
        return max(0.0, min(1.0, reward))
    
    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取认知统计信息"""
        return {
            "framework": "EvoAgent",
            "total_decisions": len(self.action_history),
            "evolution_stats": self.evolution_engine.get_statistics(),
            "current_generation": self.evolution_engine.generation,
            "action_distribution": self._get_action_distribution(),
            "avg_reward": sum(s.fitness for s in self.evolution_engine.population) / max(1, len(self.evolution_engine.population))
        }
    
    def _get_action_distribution(self) -> Dict[str, int]:
        """获取行动分布"""
        from collections import Counter
        return dict(Counter(self.action_history))
    
    def export_cognition_report(self):
        """导出认知报告"""
        report = {
            "agent_name": self.country_name,
            "framework": "EvoAgent",
            "evolution_engine": self.evolution_engine.get_statistics(),
            "population": [s.to_dict() for s in self.evolution_engine.population],
            "evolution_history": self.evolution_engine.evolution_history[-10:],
            "decision_history": {
                "actions": self.action_history[-20:],
                "declarations": self.declaration_history[-10:]
            },
            "statistics": self.get_cognition_statistics()
        }
        
        try:
            from pathlib import Path
            report_path = Path(f"{self.country_name}_evo_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.experiment_logger.log_print(f"导出EvoAgent报告失败: {e}", level="WARNING")
        
        return report
    
    @property
    def action(self):
        """属性访问器"""
        return self.action_history
    
    @property
    def declaration(self):
        """属性访问器"""
        return self.declaration_history
    
    def get_best_strategy(self) -> Optional[Dict]:
        """获取当前最佳策略"""
        best = self.evolution_engine.get_best_strategy()
        return best.to_dict() if best else None
