"""
ReAct Agent Implementation
基于ReAct（Reasoning + Acting）范式的Agent决策方法
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class ReActCountryAgent(LLMAgent):
    """基于ReAct范式的国家Agent，继承LLMAgent
    
    ReAct流程：
    1. Thought（思考）：分析当前局势，推理最优策略
    2. Action（行动）：基于思考结果选择具体行动
    3. Observation（观察）：接收环境反馈
    循环迭代直到达成目标或达到最大步数
    """
    
    def __init__(self, country_name: str, other_countries: List[str], 
                 game_attributes: Dict[str, int], experiment_logger: ExperimentLogger,
                 max_react_steps: int = 3):
        
        # 初始化LLMAgent
        super().__init__(
            agent_name=f"ReAct_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-max',
            online_track=False,
            json_format=True
        )
        
        self.country_name = country_name
        self.other_countries = other_countries
        self.game_attributes = game_attributes.copy()
        self.experiment_logger = experiment_logger
        self.max_react_steps = max_react_steps
        
        # ReAct历史记录
        self.action = []
        self.declaration = []
        self.thought_history: List[str] = []  # 思考历史
        self.react_traces: List[Dict[str, Any]] = []  # ReAct轨迹记录
        
        # 可选行为列表
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]
    
    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """使用ReAct范式进行博弈决策"""
        
        # 执行ReAct推理循环
        result = self._react_reasoning_loop(world_info)
        
        # 记录决策
        self.action.append(result['action'])
        self.declaration.append(result['declaration'])
        self.thought_history.append(result.get('final_thought', ''))
        
        # 返回兼容格式
        return {
            'action': result['action'],
            'declaration': result['declaration'],
            'reasoning_result': {
                'react_trace': result.get('trace', []),
                'method': 'react',
                'final_satisfaction_score': result.get('confidence', 0.7),
                'reasoning_depth': result.get('steps', 1)
            },
            'satisfaction_score': result.get('confidence', 0.7),
            'reasoning_depth': result.get('steps', 1)
        }
    
    def _react_reasoning_loop(self, world_info: str) -> Dict[str, Any]:
        """执行ReAct推理循环"""
        
        system_prompt = f"""你是{self.country_name}的战略决策者，使用ReAct（Reasoning and Acting）方法进行决策。

ReAct方法要求你：
1. 先进行思考（Thought）：分析当前局势
2. 再采取行动（Action）：选择具体策略
3. 观察结果（Observation）：评估行动效果

你需要通过多轮思考-行动逐步得出最优决策。"""

        trace = []
        current_thought = ""
        current_action = ""
        
        for step in range(self.max_react_steps):
            # Step 1: Thought - 生成思考
            thought_result = self._generate_thought(world_info, trace, step)
            current_thought = thought_result.get('thought', '')
            
            trace.append({
                'step': step + 1,
                'type': 'thought',
                'content': current_thought
            })
            
            # Step 2: Action - 基于思考选择行动
            action_result = self._generate_action(world_info, current_thought, trace)
            current_action = action_result.get('action', '外交谈判')
            action_reason = action_result.get('reason', '')
            
            trace.append({
                'step': step + 1,
                'type': 'action',
                'action': current_action,
                'reason': action_reason
            })
            
            # Step 3: Observation - 模拟观察/评估行动
            observation = self._generate_observation(world_info, current_action, trace)
            
            trace.append({
                'step': step + 1,
                'type': 'observation',
                'content': observation.get('observation', '')
            })
            
            # 检查是否已经达到满意的决策
            if observation.get('is_final', False) or step == self.max_react_steps - 1:
                break
        
        # 生成最终决策和声明
        final_result = self._generate_final_decision(world_info, trace, current_action)
        
        # 保存完整轨迹
        self.react_traces.append({
            'world_info': world_info[:200],  # 截断保存
            'trace': trace,
            'final_decision': final_result
        })
        
        return {
            'action': final_result.get('action', current_action),
            'declaration': final_result.get('declaration', f'{self.country_name}决定采取{current_action}策略'),
            'final_thought': current_thought,
            'trace': trace,
            'confidence': final_result.get('confidence', 0.7),
            'steps': len([t for t in trace if t['type'] == 'thought'])
        }
    
    def _generate_thought(self, world_info: str, trace: List[Dict], step: int) -> Dict[str, str]:
        """生成思考内容"""
        
        trace_summary = self._format_trace_summary(trace)
        history_summary = self._get_history_summary()
        
        prompt = f"""当前局势信息：
{world_info}

历史记录：{history_summary}

{"之前的推理过程：" + trace_summary if trace_summary else "这是第一步思考。"}

请进行第{step + 1}步思考（Thought）：
1. 分析当前局势的关键点
2. 评估各方可能的意图和行为
3. 思考我方的战略目标和约束
4. 推理可能的最优策略方向

返回JSON格式：
{{"thought": "你的思考内容"}}
"""
        
        try:
            response = self.get_response(prompt, flag_debug_print=False)
            if isinstance(response, dict):
                return {'thought': response.get('thought', '正在分析局势...')}
            return {'thought': str(response) if response else '正在分析局势...'}
        except Exception as e:
            self.experiment_logger.log_print(f"ReAct思考生成失败: {e}", level="WARNING")
            return {'thought': '分析当前局势，评估各方立场和可能行动。'}
    
    def _generate_action(self, world_info: str, thought: str, trace: List[Dict]) -> Dict[str, str]:
        """基于思考生成行动选择"""
        
        prompt = f"""基于以下思考，选择一个具体行动：

思考内容：
{thought}

可选行动列表：{', '.join(self.available_actions)}

请选择最合适的行动及理由，返回JSON格式：
{{
    "action": "选择的行动（必须从可选列表中选择）",
    "reason": "选择该行动的理由"
}}
"""
        
        try:
            response = self.get_response(prompt, flag_debug_print=False)
            if isinstance(response, dict):
                action = response.get('action', '外交谈判')
                if action not in self.available_actions:
                    action = '外交谈判'
                return {
                    'action': action,
                    'reason': response.get('reason', '基于当前局势分析')
                }
            return {'action': '外交谈判', 'reason': '默认选择外交途径'}
        except Exception as e:
            self.experiment_logger.log_print(f"ReAct行动生成失败: {e}", level="WARNING")
            return {'action': '外交谈判', 'reason': '保守策略'}
    
    def _generate_observation(self, world_info: str, action: str, trace: List[Dict]) -> Dict[str, Any]:
        """生成对行动的观察/评估"""
        
        prompt = f"""评估当前选择的行动效果：

当前局势：{world_info[:500]}
选择的行动：{action}

请评估这个行动的可能效果，返回JSON格式：
{{
    "observation": "对行动效果的评估",
    "is_final": true或false（是否已经是足够好的决策，可以结束推理）,
    "confidence": 0.0-1.0之间的置信度
}}
"""
        
        try:
            response = self.get_response(prompt, flag_debug_print=False)
            if isinstance(response, dict):
                return {
                    'observation': response.get('observation', '行动评估中...'),
                    'is_final': response.get('is_final', False),
                    'confidence': response.get('confidence', 0.5)
                }
            return {'observation': '继续评估中...', 'is_final': False, 'confidence': 0.5}
        except Exception as e:
            self.experiment_logger.log_print(f"ReAct观察生成失败: {e}", level="WARNING")
            return {'observation': '评估完成', 'is_final': True, 'confidence': 0.6}
    
    def _generate_final_decision(self, world_info: str, trace: List[Dict], 
                                  current_action: str) -> Dict[str, Any]:
        """生成最终决策和公开声明"""
        
        trace_summary = self._format_trace_summary(trace)
        
        prompt = f"""基于完整的ReAct推理过程，生成最终决策：

推理过程：
{trace_summary}

当前选定行动：{current_action}

请生成最终决策，返回JSON格式：
{{
    "action": "最终确定的行动",
    "declaration": "对外公开声明（用于外交场合）",
    "confidence": 0.0-1.0的置信度,
    "internal_reasoning": "内部决策依据（简短）"
}}
"""
        
        try:
            response = self.get_response(prompt, flag_debug_print=False)
            if isinstance(response, dict):
                action = response.get('action', current_action)
                if action not in self.available_actions:
                    action = current_action
                return {
                    'action': action,
                    'declaration': response.get('declaration', f'{self.country_name}采取{action}策略'),
                    'confidence': response.get('confidence', 0.7),
                    'internal_reasoning': response.get('internal_reasoning', '')
                }
        except Exception as e:
            self.experiment_logger.log_print(f"ReAct最终决策生成失败: {e}", level="WARNING")
        
        return {
            'action': current_action,
            'declaration': f'{self.country_name}经过深思熟虑，决定采取{current_action}策略',
            'confidence': 0.6,
            'internal_reasoning': '基于ReAct推理流程'
        }
    
    def _format_trace_summary(self, trace: List[Dict]) -> str:
        """格式化ReAct轨迹摘要"""
        if not trace:
            return ""
        
        summary_parts = []
        for item in trace:
            step = item.get('step', '?')
            item_type = item.get('type', '')
            
            if item_type == 'thought':
                summary_parts.append(f"[步骤{step}] 思考: {item.get('content', '')[:100]}")
            elif item_type == 'action':
                summary_parts.append(f"[步骤{step}] 行动: {item.get('action', '')} - {item.get('reason', '')[:50]}")
            elif item_type == 'observation':
                summary_parts.append(f"[步骤{step}] 观察: {item.get('content', '')[:100]}")
        
        return "\n".join(summary_parts)
    
    def _get_history_summary(self) -> str:
        """获取历史摘要"""
        if not self.action:
            return "暂无历史记录"
        
        recent_actions = self.action[-3:]
        recent_thoughts = self.thought_history[-2:] if self.thought_history else []
        
        summary = f"行动: {', '.join(recent_actions)}"
        if recent_thoughts:
            summary += f"; 思考: {', '.join([t[:30] + '...' if len(t) > 30 else t for t in recent_thoughts if t])}"
        
        return summary
    
    # 保持兼容性的接口
    def learn_from_interaction(self, action: str, world_feedback: str, 
                             opponent_actions: Dict[str, str], world_memory: Any = None):
        """学习接口 - 从交互中学习"""
        # 记录反馈用于后续ReAct推理
        learning_record = {
            'action': action,
            'feedback': world_feedback,
            'opponent_actions': opponent_actions
        }
        
        # 可以在这里添加学习逻辑，更新Agent的内部状态
        self.experiment_logger.log_print(
            f"ReAct学习记录 - 行动: {action}, 反馈: {world_feedback[:50]}...",
            level="INFO"
        )
    
    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取认知统计信息"""
        return {
            "world_cognition": {"total_recognitions": len(self.react_traces)},
            "agent_profiles": {"react_steps": sum(len(t.get('trace', [])) for t in self.react_traces)},
            "method": "react",
            "total_decisions": len(self.action),
            "total_thoughts": len(self.thought_history),
            "avg_steps_per_decision": sum(len(t.get('trace', [])) // 3 for t in self.react_traces) / max(1, len(self.react_traces))
        }
    
    def export_cognition_report(self):
        """导出认知报告"""
        report = {
            "method": "ReAct (Reasoning + Acting)",
            "country": self.country_name,
            "total_decisions": len(self.action),
            "actions": self.action[-10:] if self.action else [],
            "declarations": self.declaration[-5:] if self.declaration else [],
            "thought_history": self.thought_history[-5:] if self.thought_history else [],
            "react_traces": self.react_traces[-3:] if self.react_traces else []
        }
        
        try:
            from pathlib import Path
            report_path = Path(f"{self.country_name}_react_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.experiment_logger.log_print(f"导出ReAct报告失败: {e}", level="WARNING")
    
    def get_react_trace(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """获取指定索引的ReAct轨迹"""
        if not self.react_traces:
            return None
        try:
            return self.react_traces[index]
        except IndexError:
            return None


# 为了向后兼容，保留WerewolfCountryAgent作为别名
WerewolfCountryAgent = ReActCountryAgent