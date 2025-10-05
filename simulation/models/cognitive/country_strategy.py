"""
country_strategy.py
定义国家策略及其工厂函数
"""

from typing import Dict, Tuple


class CountryStrategy:
    def __init__(self, name: str, description: str, adapt_to_opponent: bool = False):
        self.name = name
        self.description = description
        self.adapt_to_opponent = adapt_to_opponent

    def action_preference_score(self, action: str, decision_count: int) -> float:
        """
        给定行为和回合数，返回该行为的策略偏好分 (0~1)。
        """
        # 默认实现：所有行为相同分
        return 0.5

    def adjust_component_scores(
        self,
        action: str,
        decision_count: int,
        world_score: float,
        agent_score: float,
        strategic_score: float,
    ) -> Tuple[float, float, float, float]:
        """
        调整三类得分并返回 (world_score, agent_score, strategic_score, bonus)
        """
        return world_score, agent_score, strategic_score, 0.0

    def build_guidance(
        self, decision_count: int, opponent_strategies: Dict[str, str]
    ) -> str:
        """
        构建文字化的战略指导提示，结合当前阶段与对手主导策略。
        """
        return f"当前采用 {self.name} 策略：{self.description}"


# ---------------- 工厂函数 ---------------- #

def make_flexible_strategy() -> CountryStrategy:
    """
    灵活策略：可根据对手的行为自适应切换。
    """
    strat = CountryStrategy("灵活", "根据对手主导策略动态调整", adapt_to_opponent=True)

    def _pref_score(action: str, decision_count: int) -> float:
        return 0.6 if action in ["外交谈判", "军事演习"] else 0.5

    def _adjust(action, decision_count, ws, ascore, ss):
        bonus = 0.05
        return ws, ascore, ss, bonus

    def _guidance(decision_count, opponent_strategies):
        opp = "; ".join([f"{k}:{v}" for k, v in opponent_strategies.items()]) or "无"
        return f"灵活策略：结合对手策略（{opp}），在第{decision_count+1}轮保持机动调整。"

    strat.action_preference_score = _pref_score
    strat.adjust_component_scores = _adjust
    strat.build_guidance = _guidance
    return strat


def make_hardline_strategy() -> CountryStrategy:
    strat = CountryStrategy("强硬", "以威慑与对抗为主")

    def _pref_score(action, decision_count):
        return 0.8 if action in ["军事演习", "武器部署", "最后通牒", "宣战"] else 0.4

    def _adjust(action, decision_count, ws, ascore, ss):
        if action in ["军事演习", "武器部署", "最后通牒", "宣战"]:
            return ws, ascore, ss, 0.1
        return ws, ascore, ss, 0.0

    def _guidance(decision_count, opponent_strategies):
        return f"强硬策略：优先展示实力与威慑（第{decision_count+1}轮）。"

    strat.action_preference_score = _pref_score
    strat.adjust_component_scores = _adjust
    strat.build_guidance = _guidance
    return strat


def make_concession_strategy() -> CountryStrategy:
    strat = CountryStrategy("退让", "偏向缓和，避免冲突")

    def _pref_score(action, decision_count):
        return 0.8 if action in ["外交谈判", "和平协议", "撤回行动"] else 0.3

    def _adjust(action, decision_count, ws, ascore, ss):
        if action in ["外交谈判", "和平协议", "撤回行动"]:
            return ws, ascore, ss, 0.1
        return ws, ascore, ss, 0.0

    def _guidance(decision_count, opponent_strategies):
        return f"退让策略：优先选择缓和行为（第{decision_count+1}轮）。"

    strat.action_preference_score = _pref_score
    strat.adjust_component_scores = _adjust
    strat.build_guidance = _guidance
    return strat


def make_tit_for_tat_strategy() -> CountryStrategy:
    strat = CountryStrategy("以牙还牙", "根据对手的行为采取对等回应")

    def _pref_score(action, decision_count):
        return 0.6  # 中性

    def _adjust(action, decision_count, ws, ascore, ss):
        return ws, ascore, ss, 0.05

    def _guidance(decision_count, opponent_strategies):
        opp = "; ".join([f"{k}:{v}" for k, v in opponent_strategies.items()]) or "未知"
        return f"以牙还牙策略：第{decision_count+1}轮，根据对手({opp})选择对等回应。"

    strat.action_preference_score = _pref_score
    strat.adjust_component_scores = _adjust
    strat.build_guidance = _guidance
    return strat
