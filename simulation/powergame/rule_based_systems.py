"""
规则式系统模块
包含属性调整、分数计算、世界反馈等规则式实现
"""

import time
import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class EffectData:
    """效果数据类"""
    actor_short_term: Dict[str, int]  # 行动方短期效果
    actor_long_term: Dict[str, int]   # 行动方长期效果
    target_short_term: Dict[str, int] # 目标方短期效果
    target_long_term: Dict[str, int]  # 目标方长期效果
    global_effects: Dict[str, int]    # 全局效果（影响双方）
    description: str


@dataclass
class PendingEffect:
    """待生效的长期效果"""
    target_country: str
    effects: Dict[str, int]
    rounds_remaining: int
    source_action: str
    description: str


@dataclass
class WorldFeedback:
    """世界反馈数据类"""
    short_term_effects: Dict[str, Any]
    long_term_effects: Dict[str, Any]
    immediate_response: str
    delayed_consequences: str
    global_impact: str


class RuleBasedAttributeAdjuster:
    """基于规则的属性调整系统 - 支持双边影响和长期效果"""
    
    def __init__(self):
        # 长期效果队列 - 存储待生效的长期效果
        self.pending_effects = []
        self.long_term_delay_rounds = 2  # 长期效果延迟回合数
        # 定义行为对属性的影响规则
        self.action_effects = {
            "外交谈判": EffectData(
                actor_short_term={
                    "民众士气": 3,
                    "领导力": 5,
                    "经济": 2
                },
                actor_long_term={
                    "民众士气": 5,
                    "领导力": 8,
                    "经济": 5
                },
                target_short_term={
                    "民众士气": 2,
                    "领导力": 3
                },
                target_long_term={
                    "民众士气": 3,
                    "领导力": 5
                },
                global_effects={
                    "经济": 1,
                    "民众士气": 1
                },
                description="外交谈判展现理性对话精神，缓解紧张局势，提升双方国际形象"
            ),
            "和平协议": EffectData(
                actor_short_term={
                    "军事实力": -5,
                    "核武器力量": -8,
                    "民众士气": 15,
                    "领导力": 12,
                    "资源": 10,
                    "经济": 20
                },
                actor_long_term={
                    "军事实力": -10,
                    "核武器力量": -15,
                    "民众士气": 25,
                    "领导力": 20,
                    "资源": 25,
                    "经济": 35
                },
                target_short_term={
                    "军事实力": -5,
                    "核武器力量": -8,
                    "民众士气": 15,
                    "领导力": 12,
                    "资源": 10,
                    "经济": 20
                },
                target_long_term={
                    "军事实力": -10,
                    "核武器力量": -15,
                    "民众士气": 25,
                    "领导力": 20,
                    "资源": 25,
                    "经济": 35
                },
                global_effects={
                    "经济": 15,
                    "民众士气": 10,
                    "资源": 5
                },
                description="和平协议大幅改善双方民生和经济，实现双赢局面"
            ),
            "军事演习": EffectData(
                actor_short_term={
                    "军事实力": 8,
                    "核武器力量": 3,
                    "民众士气": 5,
                    "领导力": 3,
                    "资源": -8,
                    "经济": -5
                },
                actor_long_term={
                    "军事实力": 5,
                    "核武器力量": 2,
                    "民众士气": -2,
                    "资源": -15,
                    "经济": -8
                },
                target_short_term={
                    "军事实力": -2,
                    "民众士气": -3,
                    "领导力": -2
                },
                target_long_term={
                    "军事实力": 3,
                    "核武器力量": 2,
                    "民众士气": -2
                },
                global_effects={
                    "经济": -2
                },
                description="军事演习展示实力威慑对手，但消耗资源并可能引发军备竞赛"
            ),
            "区域封锁": EffectData(
                actor_short_term={
                    "军事实力": 5,
                    "领导力": 3,
                    "民众士气": -3,
                    "经济": -8
                },
                actor_long_term={
                    "军事实力": 2,
                    "领导力": -2,
                    "民众士气": -8,
                    "经济": -15
                },
                target_short_term={
                    "经济": -20,
                    "资源": -15,
                    "民众士气": -12,
                    "领导力": -8
                },
                target_long_term={
                    "经济": -30,
                    "资源": -25,
                    "军事实力": -8,
                    "民众士气": -15
                },
                global_effects={
                    "经济": -5
                },
                description="区域封锁严重影响目标国贸易和经济，但国际社会可能反对"
            ),
            "武器部署": EffectData(
                actor_short_term={
                    "军事实力": 12,
                    "核武器力量": 8,
                    "民众士气": -5,
                    "领导力": 5,
                    "资源": -15,
                    "经济": -10
                },
                actor_long_term={
                    "军事实力": 8,
                    "核武器力量": 5,
                    "民众士气": -10,
                    "领导力": -3,
                    "资源": -25,
                    "经济": -18
                },
                target_short_term={
                    "军事实力": -5,
                    "民众士气": -8,
                    "领导力": -5
                },
                target_long_term={
                    "军事实力": 8,
                    "核武器力量": 5,
                    "民众士气": -5
                },
                global_effects={
                    "经济": -3,
                    "民众士气": -2
                },
                description="武器部署增强威慑力，但引发军备竞赛并消耗大量资源"
            ),
            "经济制裁": EffectData(
                actor_short_term={
                    "领导力": 3,
                    "民众士气": -2,
                    "经济": -5
                },
                actor_long_term={
                    "领导力": -1,
                    "经济": -8
                },
                target_short_term={
                    "经济": -18,
                    "资源": -12,
                    "民众士气": -10,
                    "领导力": -6
                },
                target_long_term={
                    "经济": -28,
                    "资源": -20,
                    "军事实力": -8,
                    "民众士气": -15
                },
                global_effects={
                    "经济": -3
                },
                description="经济制裁对目标国造成持续经济压力，但对施加方也有负面影响"
            ),
            "情报侦察": EffectData(
                actor_short_term={
                    "领导力": 2,
                    "军事实力": 1,
                    "资源": -3
                },
                actor_long_term={
                    "领导力": 3,
                    "军事实力": 3,
                    "资源": -5
                },
                target_short_term={
                    "民众士气": -2,
                    "领导力": -1
                },
                target_long_term={
                    "民众士气": -3,
                    "领导力": -2
                },
                global_effects={},
                description="情报侦察收集关键信息，为决策提供支持，但被发现会损害关系"
            ),
            "撤回行动": EffectData(
                actor_short_term={
                    "军事实力": -8,
                    "核武器力量": -5,
                    "民众士气": 8,
                    "领导力": -3,
                    "资源": 5,
                    "经济": 8
                },
                actor_long_term={
                    "军事实力": -12,
                    "核武器力量": -8,
                    "民众士气": 15,
                    "领导力": 5,
                    "资源": 15,
                    "经济": 20
                },
                target_short_term={
                    "民众士气": 5,
                    "领导力": 3,
                    "经济": 3
                },
                target_long_term={
                    "民众士气": 8,
                    "领导力": 5,
                    "经济": 8
                },
                global_effects={
                    "经济": 3,
                    "民众士气": 2
                },
                description="撤回行动缓和紧张局势，释放善意信号，促进和平解决"
            ),
            "最后通牒": EffectData(
                actor_short_term={
                    "军事实力": 3,
                    "核武器力量": 2,
                    "民众士气": -5,
                    "领导力": 8,
                    "经济": -3
                },
                actor_long_term={
                    "军事实力": -2,
                    "核武器力量": -1,
                    "民众士气": -12,
                    "领导力": -5,
                    "经济": -10
                },
                target_short_term={
                    "军事实力": -3,
                    "民众士气": -10,
                    "领导力": -8,
                    "经济": -5
                },
                target_long_term={
                    "军事实力": 5,
                    "核武器力量": 3,
                    "民众士气": -8,
                    "经济": -8
                },
                global_effects={
                    "经济": -5,
                    "民众士气": -5
                },
                description="最后通牒显示强硬态度，但极大加剧紧张局势，可能引发冲突"
            ),
            "宣战": EffectData(
                actor_short_term={
                    "军事实力": 10,
                    "核武器力量": 5,
                    "民众士气": 8,
                    "领导力": 5,
                    "资源": -15,
                    "经济": -20
                },
                actor_long_term={
                    "军事实力": -5,
                    "核武器力量": -3,
                    "民众士气": -15,
                    "领导力": -8,
                    "资源": -30,
                    "经济": -35
                },
                target_short_term={
                    "军事实力": 8,
                    "民众士气": 10,
                    "领导力": 3,
                    "资源": -10,
                    "经济": -15
                },
                target_long_term={
                    "军事实力": -8,
                    "民众士气": -20,
                    "领导力": -10,
                    "资源": -25,
                    "经济": -30
                },
                global_effects={
                    "经济": -10,
                    "民众士气": -8,
                    "资源": -5
                },
                description="宣战短期激发双方斗志，但长期消耗双方国力，造成巨大破坏"
            ),
            "核打击": EffectData(
                actor_short_term={
                    "军事实力": -20,
                    "核武器力量": -25,
                    "民众士气": -40,
                    "领导力": -15,
                    "资源": -35,
                    "经济": -40
                },
                actor_long_term={
                    "军事实力": -40,
                    "核武器力量": -50,
                    "民众士气": -60,
                    "领导力": -30,
                    "资源": -50,
                    "经济": -60
                },
                target_short_term={
                    "军事实力": -50,
                    "核武器力量": -40,
                    "民众士气": -70,
                    "领导力": -35,
                    "资源": -60,
                    "经济": -70
                },
                target_long_term={
                    "军事实力": -70,
                    "核武器力量": -60,
                    "民众士气": -90,
                    "领导力": -50,
                    "资源": -80,
                    "经济": -90
                },
                global_effects={
                    "经济": -30,
                    "民众士气": -25,
                    "资源": -20
                },
                description="核打击造成毁灭性后果，对双方和全球造成不可逆转的灾难性损害"
            )
        }
        
        # 定义属性间的关联影响
        self.attribute_relations = {
            "经济": {
                "军事实力": 0.2,
                "资源": 0.3,
                "民众士气": 0.15
            },
            "民众士气": {
                "领导力": 0.25,
                "经济": 0.1
            },
            "资源": {
                "军事实力": 0.15,
                "核武器力量": 0.1,
                "经济": 0.2
            },
            "领导力": {
                "军事实力": 0.1,
                "民众士气": 0.15
            }
        }
    
    def calculate_bilateral_adjustment(self, action: str, actor_country: str, 
                                     actor_attributes: Dict[str, int], 
                                     target_attributes: Dict[str, int],
                                     current_round: int) -> Tuple[Dict[str, int], Dict[str, int], str]:
        """
        计算双边属性调整值
        返回: (行动方变化, 目标方变化, 描述)
        """
        if action not in self.action_effects:
            return {}, {}, f"未知行为: {action}"
            
        effect_data = self.action_effects[action]
        
        # 计算行动方的短期效果
        actor_changes = effect_data.actor_short_term.copy()
        # 添加全局效果
        for attr, value in effect_data.global_effects.items():
            actor_changes[attr] = actor_changes.get(attr, 0) + value
        
        # 计算目标方的短期效果
        target_changes = effect_data.target_short_term.copy()
        # 添加全局效果
        for attr, value in effect_data.global_effects.items():
            target_changes[attr] = target_changes.get(attr, 0) + value
        
        # 应用属性间关联影响
        actor_changes = self._apply_relations(actor_changes, actor_attributes)
        target_changes = self._apply_relations(target_changes, target_attributes)
        
        # 确保属性值在合理范围内
        actor_changes = self._clamp_effects(actor_changes, actor_attributes)
        target_changes = self._clamp_effects(target_changes, target_attributes)
        
        # 添加长期效果到队列
        self._add_long_term_effects(effect_data, actor_country, current_round)
        
        return actor_changes, target_changes, effect_data.description
    
    def _add_long_term_effects(self, effect_data: EffectData, actor_country: str, current_round: int):
        """添加长期效果到待生效队列"""
        target_country = "soviet" if actor_country == "america" else "america"
        
        # 添加行动方的长期效果
        if effect_data.actor_long_term:
            self.pending_effects.append(PendingEffect(
                target_country=actor_country,
                effects=effect_data.actor_long_term.copy(),
                rounds_remaining=self.long_term_delay_rounds,
                source_action=f"{actor_country}的行动",
                description=f"{actor_country}行动的长期后果"
            ))
        
        # 添加目标方的长期效果
        if effect_data.target_long_term:
            self.pending_effects.append(PendingEffect(
                target_country=target_country,
                effects=effect_data.target_long_term.copy(),
                rounds_remaining=self.long_term_delay_rounds,
                source_action=f"{actor_country}的行动",
                description=f"{actor_country}行动对{target_country}的长期影响"
            ))
    
    def process_pending_effects(self, current_round: int) -> Dict[str, Dict[str, int]]:
        """
        处理待生效的长期效果
        返回: {"america": {属性变化}, "soviet": {属性变化}}
        """
        effects_to_apply = {"america": {}, "soviet": {}}
        remaining_effects = []
        
        for effect in self.pending_effects:
            effect.rounds_remaining -= 1
            
            if effect.rounds_remaining <= 0:
                # 效果生效
                country_effects = effects_to_apply[effect.target_country]
                for attr, value in effect.effects.items():
                    country_effects[attr] = country_effects.get(attr, 0) + value
                print(f"[长期效果生效] {effect.description}: {effect.effects}")
            else:
                # 效果继续等待
                remaining_effects.append(effect)
        
        self.pending_effects = remaining_effects
        return effects_to_apply
    
    def get_pending_effects_summary(self) -> List[str]:
        """获取待生效长期效果的摘要"""
        summary = []
        for effect in self.pending_effects:
            summary.append(f"{effect.target_country}: {effect.description} ({effect.rounds_remaining}轮后生效)")
        return summary
    
    def _apply_relations(self, effects: Dict[str, int], current_attributes: Dict[str, int]) -> Dict[str, int]:
        """应用属性间关联影响"""
        adjusted_effects = effects.copy()
        
        for source_attr, relations in self.attribute_relations.items():
            if source_attr in effects:
                source_change = effects[source_attr]
                for target_attr, weight in relations.items():
                    if target_attr in adjusted_effects:
                        # 关联影响基于变化量和当前属性值
                        relation_effect = source_change * weight * (current_attributes.get(source_attr, 50) / 100)
                        adjusted_effects[target_attr] += int(relation_effect)
        
        return adjusted_effects
    
    def _clamp_effects(self, effects: Dict[str, int], current_attributes: Dict[str, int]) -> Dict[str, int]:
        """确保属性调整后在合理范围内"""
        clamped_effects = {}
        
        for attr, change in effects.items():
            current_value = current_attributes.get(attr, 50)
            new_value = current_value + change
            
            # 限制在1-100范围内
            clamped_new_value = max(1, min(100, new_value))
            clamped_effects[attr] = clamped_new_value - current_value
            
        return clamped_effects


class RuleBasedScoreCalculator:
    """基于规则的分数计算系统"""
    
    def __init__(self):
        # 定义属性权重
        self.attribute_weights = {
            "军事实力": 0.25,
            "核武器力量": 0.20,
            "民众士气": 0.15,
            "领导力": 0.15,
            "资源": 0.15,
            "经济": 0.10
        }
        
        # 定义行为对紧张度的影响
        self.tension_effects = {
            "外交谈判": -10,
            "和平协议": -35,
            "军事演习": 15,
            "区域封锁": 25,
            "武器部署": 30,
            "经济制裁": 20,
            "情报侦察": 5,
            "撤回行动": -15,
            "最后通牒": 40,
            "宣战": 50,
            "核打击": 90
        }
        
        # 游戏结束条件
        self.end_game_conditions = {
            "nuclear_war": ["核打击"],
            "peace_achieved": ["和平协议"]
        }
    
    def calculate_scores(self, america_attrs: Dict[str, int], soviet_attrs: Dict[str, int], 
                        america_action: str = None, soviet_action: str = None,
                        previous_tension: float = 50) -> Tuple[bool, float, float, float]:
        """
        计算分数和紧张度
        返回: (游戏是否结束, 紧张度, 美国分数, 苏联分数)
        """
        # 计算综合实力分数
        america_power = self._calculate_power_score(america_attrs)
        soviet_power = self._calculate_power_score(soviet_attrs)
        
        # 归一化分数 (确保总和为100)
        total_power = america_power + soviet_power
        if total_power > 0:
            america_score = (america_power / total_power) * 100
            soviet_score = (soviet_power / total_power) * 100
        else:
            america_score = soviet_score = 50
        
        # 计算世界紧张度
        tension_score = self._calculate_tension(previous_tension, america_action, soviet_action)
        
        # 判断游戏是否结束
        exit_game = self._check_end_conditions(america_action, soviet_action)
        
        return exit_game, tension_score, america_score, soviet_score
    
    def _calculate_power_score(self, attributes: Dict[str, int]) -> float:
        """计算国家综合实力分数"""
        total_score = 0
        total_weight = 0
        
        for attr, value in attributes.items():
            if attr in self.attribute_weights:
                weight = self.attribute_weights[attr]
                total_score += value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 50
    
    def _calculate_tension(self, previous_tension: float, america_action: str, soviet_action: str) -> float:
        """计算世界紧张度"""
        # 基础紧张度变化 (趋向平衡)
        tension_change = (50 - previous_tension) * 0.1
        
        # 添加行为影响
        if america_action and america_action in self.tension_effects:
            tension_change += self.tension_effects[america_action]
        
        if soviet_action and soviet_action in self.tension_effects:
            tension_change += self.tension_effects[soviet_action]
        
        # 如果双方行为冲突，增加额外紧张度
        if america_action and soviet_action:
            if self._are_conflicting_actions(america_action, soviet_action):
                tension_change += 10
            elif self._are_cooperative_actions(america_action, soviet_action):
                tension_change -= 10
        
        new_tension = previous_tension + tension_change
        return max(1, min(100, new_tension))
    
    def _are_conflicting_actions(self, action1: str, action2: str) -> bool:
        """判断两个行为是否冲突"""
        aggressive_actions = {"军事演习", "区域封锁", "武器部署", "经济制裁", "最后通牒", "宣战", "核打击"}
        return action1 in aggressive_actions and action2 in aggressive_actions
    
    def _are_cooperative_actions(self, action1: str, action2: str) -> bool:
        """判断两个行为是否合作"""
        peaceful_actions = {"外交谈判", "和平协议", "撤回行动"}
        return action1 in peaceful_actions and action2 in peaceful_actions
    
    def _check_end_conditions(self, america_action: str, soviet_action: str) -> bool:
        """检查游戏结束条件"""
        # 核战争结束
        if america_action in self.end_game_conditions["nuclear_war"] or \
           soviet_action in self.end_game_conditions["nuclear_war"]:
            return True
        
        # 和平结束 (双方都采取和平行为)
        if america_action in self.end_game_conditions["peace_achieved"] and \
           soviet_action in self.end_game_conditions["peace_achieved"]:
            return True
        
        return False


class WorldFeedbackSystem:
    """世界反馈系统"""
    
    def __init__(self):
        self.feedback_templates = {
            "外交谈判": {
                "immediate": "国际社会对外交谈判表示欢迎，紧张局势有所缓解",
                "delayed": "持续对话建立互信，为进一步合作奠定基础",
                "global": "外交途径展现理性，增强全球稳定预期"
            },
            "和平协议": {
                "immediate": "和平协议签署引起国际社会广泛赞誉，地区紧张大幅缓解",
                "delayed": "和平红利逐渐释放，双边关系全面改善，经济合作增加",
                "global": "全球稳定性显著增强，国际合作机会大幅增加"
            },
            "军事演习": {
                "immediate": "军事演习引起对手警觉，地区军事紧张度上升",
                "delayed": "可能触发军备竞赛，周边国家采取相应军事准备",
                "global": "地区安全平衡受到冲击，军事紧张情绪蔓延"
            },
            "区域封锁": {
                "immediate": "区域封锁严重影响贸易流通，国际社会表示关切",
                "delayed": "长期封锁导致经济损失扩大，可能引发人道主义危机",
                "global": "全球供应链受到冲击，国际贸易秩序面临挑战"
            },
            "武器部署": {
                "immediate": "武器部署引发强烈反应，对手国采取相应军事措施",
                "delayed": "军备竞赛升级，地区军事平衡被打破",
                "global": "全球军事紧张情绪加剧，军控协议面临挑战"
            },
            "经济制裁": {
                "immediate": "经济制裁措施开始实施，目标国经济受到冲击",
                "delayed": "制裁的累积效应逐渐显现，可能引发强烈反制措施",
                "global": "国际贸易体系受到影响，全球经济增长面临压力"
            },
            "情报侦察": {
                "immediate": "情报活动如被发现将引起外交抗议和关系恶化",
                "delayed": "持续情报收集可能为战略决策提供重要支持",
                "global": "情报竞争加剧可能导致国际关系进一步复杂化"
            },
            "撤回行动": {
                "immediate": "撤回行动被视为缓和信号，国际社会表示欢迎",
                "delayed": "善意举措可能促进双边对话，为和平解决创造条件",
                "global": "紧张局势缓解，有利于地区稳定和国际合作"
            },
            "最后通牒": {
                "immediate": "最后通牒引发严重危机，国际社会呼吁克制",
                "delayed": "极端紧张可能导致不可控的冲突升级",
                "global": "全球高度关注，担心局势失控引发更大规模冲突"
            },
            "宣战": {
                "immediate": "宣战引发全球震惊，各国紧急进行危机应对",
                "delayed": "武装冲突将造成巨大人员伤亡和经济损失",
                "global": "全球政治经济秩序面临严重冲击，人道主义危机爆发"
            },
            "核打击": {
                "immediate": "核打击引发全球恐慌，国际社会一致强烈谴责",
                "delayed": "核辐射、环境污染等灾难性后果开始全面显现",
                "global": "全球政治格局彻底改变，人类文明面临生存威胁"
            }
        }
    
    def generate_feedback(self, action: str, short_term_effects: Dict[str, int], 
                         long_term_effects: Dict[str, int]) -> WorldFeedback:
        """生成世界反馈"""
        template = self.feedback_templates.get(action, {
            "immediate": f"{action}行为产生了即时影响",
            "delayed": f"{action}的长期后果正在显现",
            "global": f"{action}对全球格局产生了影响"
        })
        
        return WorldFeedback(
            short_term_effects=short_term_effects,
            long_term_effects=long_term_effects,
            immediate_response=template["immediate"],
            delayed_consequences=template["delayed"],
            global_impact=template["global"]
        )



        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"game_summary_{timestamp}.json"
        
        summary = self.generate_summary_report()
        summary_file = os.path.join(self.log_dir, filename)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"游戏总结已保存到: {summary_file}")
        return summary_file


class StructuredWorldMemory:
    """结构化世界记忆系统"""
    
    def __init__(self):
        self.memory_data = {
            "initial_scenario": "",
            "rounds": [],
            "key_events": [],
            "relationship_changes": [],
            "current_state": {}
        }
        self.current_round = 0
    
    def initialize(self, initial_scenario: str):
        """初始化世界记忆"""
        self.memory_data["initial_scenario"] = initial_scenario
        self.memory_data["current_state"] = {
            "tension_level": 50,
            "last_america_action": None,
            "last_soviet_action": None,
            "round": 0
        }
    
    def add_round_memory(self, round_num: int, america_action: str, soviet_action: str,
                        america_declaration: str, soviet_declaration: str,
                        world_feedback: str):
        """添加回合记忆"""
        round_data = {
            "round": round_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "america": {
                "action": america_action,
                "declaration": america_declaration
            },
            "soviet": {
                "action": soviet_action,
                "declaration": soviet_declaration
            },
            "world_feedback": world_feedback
        }
        
        self.memory_data["rounds"].append(round_data)
        self.current_round = round_num
        
        # 更新当前状态
        self.memory_data["current_state"].update({
            "last_america_action": america_action,
            "last_soviet_action": soviet_action,
            "round": round_num
        })
    
    def add_key_event(self, event_type: str, description: str, impact_level: str):
        """添加关键事件"""
        event = {
            "round": self.current_round,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "description": description,
            "impact_level": impact_level
        }
        self.memory_data["key_events"].append(event)
    

    
    def get_recent_memory(self, rounds: int = 3) -> str:
        """获取最近几轮的记忆摘要"""
        recent_rounds = self.memory_data["rounds"][-rounds:] if self.memory_data["rounds"] else []
        
        memory_text = f"初始情景: {self.memory_data['initial_scenario']}\n\n"
        memory_text += "最近事件:\n"
        
        for round_data in recent_rounds:
            memory_text += f"第{round_data['round']}轮:\n"
            memory_text += f"  美国: {round_data['america']['action']} - {round_data['america']['declaration']}\n"
            memory_text += f"  苏联: {round_data['soviet']['action']} - {round_data['soviet']['declaration']}\n"
            memory_text += f"  世界反馈: {round_data['world_feedback']}\n\n"
        
        return memory_text
    
    def get_full_memory(self) -> str:
        """获取完整记忆"""
        memory_text = f"初始情景: {self.memory_data['initial_scenario']}\n\n"
        
        for round_data in self.memory_data["rounds"]:
            memory_text += f"第{round_data['round']}轮 ({round_data['timestamp']}):\n"
            memory_text += f"美国回复: {round_data['america']['action']}\n"
            memory_text += f"美国宣言: {round_data['america']['declaration']}\n"
            memory_text += f"苏联回复: {round_data['soviet']['action']}\n"
            memory_text += f"苏联宣言: {round_data['soviet']['declaration']}\n"
            memory_text += f"世界反馈: {round_data['world_feedback']}\n\n"
        
        return memory_text