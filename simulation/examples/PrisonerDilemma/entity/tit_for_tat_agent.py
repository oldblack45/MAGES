"""
一报还一报策略Agent
实现经典的Tit-for-Tat策略：第一轮合作，之后模仿对手上轮行为
"""

from typing import Dict, List, Any, Optional
from simulation.models.cognitive.experiment_logger import log_print


class TitForTatAgent:
    """一报还一报策略Agent"""
    
    def __init__(self, agent_name: str):
        """
        初始化一报还一报Agent
        
        Args:
            agent_name: Agent名称
        """
        self.agent_name = agent_name
        self.opponent_last_action = None
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
        
        log_print(f"{agent_name}: 初始化一报还一报策略Agent", level="INFO")
    
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        """
        决定下一步行动
        策略：第一轮合作，之后模仿对手上轮行为
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            行动选择 ('cooperate' 或 'defect')
        """
        current_round = game_state.get('current_round', 0)
        
        # 第一轮始终合作
        if current_round == 1 or self.opponent_last_action is None:
            action = 'cooperate'
            log_print(f"{self.agent_name}: 第{current_round}轮 - 首轮合作", level="DEBUG")
        else:
            # 模仿对手上轮行为
            action = self.opponent_last_action
            log_print(
                f"{self.agent_name}: 第{current_round}轮 - 模仿对手上轮行为: {action}",
                level="DEBUG"
            )
        
        # 记录自己的行动
        self.my_actions.append(action)
        
        return action
    
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        """
        接收回合结果反馈
        
        Args:
            opponent_action: 对手的行动
            my_payoff: 自己获得的收益
            round_result: 完整的回合结果
        """
        # 更新对手上轮行动记录
        self.opponent_last_action = opponent_action
        self.opponent_actions.append(opponent_action)
        self.game_history.append(round_result)
        
        log_print(
            f"{self.agent_name}: 收到反馈 - 对手行动: {opponent_action}, "
            f"我的收益: {my_payoff}",
            level="DEBUG"
        )
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return "一报还一报策略：第一轮合作，之后模仿对手上轮的行为"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.my_actions:
            return {
                'total_rounds': 0,
                'cooperation_rate': 0.0,
                'defection_rate': 0.0,
                'strategy_type': 'Tit-for-Tat'
            }
        
        total_rounds = len(self.my_actions)
        cooperations = sum(1 for action in self.my_actions if action == 'cooperate')
        
        return {
            'total_rounds': total_rounds,
            'cooperation_rate': cooperations / total_rounds,
            'defection_rate': (total_rounds - cooperations) / total_rounds,
            'strategy_type': 'Tit-for-Tat',
            'strategy_description': self.get_strategy_description()
        }
    
    def reset_game(self):
        """重置游戏状态"""
        self.opponent_last_action = None
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
        
        log_print(f"{self.agent_name}: 重置游戏状态", level="DEBUG")


class AlwaysCooperateAgent:
    """总是合作的Agent（用于测试）"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.game_history = []
        self.my_actions = []
        
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        """总是选择合作"""
        action = 'cooperate'
        self.my_actions.append(action)
        return action
    
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        """接收反馈"""
        self.game_history.append(round_result)
    
    def get_strategy_description(self) -> str:
        return "总是合作策略：无论对手如何行动，始终选择合作"
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_rounds': len(self.my_actions),
            'cooperation_rate': 1.0,
            'defection_rate': 0.0,
            'strategy_type': 'Always-Cooperate'
        }
    
    def reset_game(self):
        self.game_history = []
        self.my_actions = []


class GrimTriggerAgent:
    """Grim Trigger：若对手曾背叛，则永久背叛；否则合作"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
        self.grim_triggered = False
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        if self.grim_triggered:
            action = 'defect'
        else:
            action = 'cooperate'
        self.my_actions.append(action)
        return action
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        self.opponent_actions.append(opponent_action)
        self.game_history.append(round_result)
        if opponent_action == 'defect':
            self.grim_triggered = True
    def get_strategy_description(self) -> str:
        return "Grim Trigger：对手一旦背叛，我方永久背叛"
    def reset_game(self):
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
        self.grim_triggered = False


class PavlovAgent:
    """Pavlov (Win-Stay, Lose-Shift)：双赢或双输时保持，否则切换"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        if not self.my_actions:
            action = 'cooperate'
        else:
            prev_same = (self.my_actions[-1] == self.opponent_actions[-1])
            action = self.my_actions[-1] if prev_same else ('cooperate' if self.my_actions[-1]=='defect' else 'defect')
        self.my_actions.append(action)
        return action
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        self.opponent_actions.append(opponent_action)
        self.game_history.append(round_result)
    def get_strategy_description(self) -> str:
        return "Pavlov：赢就保持，输就切换"
    def reset_game(self):
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []


class SuspiciousTFTAgent:
    """可疑的一报还一报：首轮背叛，其后模仿对手"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        current_round = game_state.get('current_round', 0)
        if current_round == 1 or not self.opponent_actions:
            action = 'defect'
        else:
            action = self.opponent_actions[-1]
        self.my_actions.append(action)
        return action
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        self.opponent_actions.append(opponent_action)
        self.game_history.append(round_result)
    def get_strategy_description(self) -> str:
        return "Suspicious TFT：首轮背叛，其后模仿"
    def reset_game(self):
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []


class GenerousTFTAgent:
    """宽恕型一报还一报：大体模仿，但有概率原谅（合作）"""
    def __init__(self, agent_name: str, forgive_prob: float = 0.2):
        self.agent_name = agent_name
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
        self.forgive_prob = forgive_prob
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        import random
        current_round = game_state.get('current_round', 0)
        if current_round == 1 or not self.opponent_actions:
            action = 'cooperate'
        else:
            if self.opponent_actions[-1] == 'defect' and random.random() < self.forgive_prob:
                action = 'cooperate'
            else:
                action = self.opponent_actions[-1]
        self.my_actions.append(action)
        return action
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        self.opponent_actions.append(opponent_action)
        self.game_history.append(round_result)
    def get_strategy_description(self) -> str:
        return f"Generous TFT：有{self.forgive_prob:.0%}概率原谅背叛"
    def reset_game(self):
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
class AlwaysDefectAgent:
    """总是背叛的Agent（用于测试）"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.game_history = []
        self.my_actions = []
        
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        """总是选择背叛"""
        action = 'defect'
        self.my_actions.append(action)
        return action
    
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        """接收反馈"""
        self.game_history.append(round_result)
    
    def get_strategy_description(self) -> str:
        return "总是背叛策略：无论对手如何行动，始终选择背叛"
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_rounds': len(self.my_actions),
            'cooperation_rate': 0.0,
            'defection_rate': 1.0,
            'strategy_type': 'Always-Defect'
        }
    
    def reset_game(self):
        self.game_history = []
        self.my_actions = []


 