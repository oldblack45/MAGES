"""
囚徒困境博弈世界
实现囚徒困境的游戏环境，用于测试认知Agent的策略识别能力
"""

import time
import os
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from simulation.models.cognitive import CognitiveAgent
from simulation.models.cognitive.experiment_logger import (
    ExperimentLogger, init_logger, log_print, get_logger
)


class PrisonerDilemmaGame:
    """囚徒困境游戏环境"""
    
    # 游戏收益矩阵 (自己的收益, 对手的收益)
    PAYOFF_MATRIX = {
        ('cooperate', 'cooperate'): (3, 3),    # 双方合作
        ('cooperate', 'defect'): (0, 5),       # 自己合作，对方背叛
        ('defect', 'cooperate'): (5, 0),       # 自己背叛，对方合作
        ('defect', 'defect'): (1, 1)           # 双方背叛
    }
    
    def __init__(self, player1_name: str, player2_name: str, max_rounds: int = 100):
        """
        初始化囚徒困境游戏
        
        Args:
            player1_name: 玩家1名称
            player2_name: 玩家2名称
            max_rounds: 最大回合数
        """
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.max_rounds = max_rounds
        self.current_round = 0
        
        # 游戏历史记录
        self.game_history = []
        self.player1_score = 0
        self.player2_score = 0
        
        # 玩家引用
        self.player1 = None
        self.player2 = None
        
    def register_players(self, player1, player2):
        """注册玩家"""
        self.player1 = player1
        self.player2 = player2
        
    def get_available_actions(self) -> List[str]:
        """获取可用行动"""
        return ['cooperate', 'defect']
    
    def play_round(self) -> Dict[str, Any]:
        """进行一轮游戏"""
        if self.current_round >= self.max_rounds:
            return None
            
        self.current_round += 1
        
        # 获取双方决策
        player1_action = self.player1.decide_action(self.get_game_state())
        player2_action = self.player2.decide_action(self.get_game_state())
        
        # 计算收益
        payoffs = self.PAYOFF_MATRIX.get((player1_action, player2_action), (0, 0))
        player1_payoff, player2_payoff = payoffs
        
        # 更新得分
        self.player1_score += player1_payoff
        self.player2_score += player2_payoff
        
        # 记录历史
        round_result = {
            'round': self.current_round,
            'player1_action': player1_action,
            'player2_action': player2_action,
            'player1_payoff': player1_payoff,
            'player2_payoff': player2_payoff,
            'player1_total_score': self.player1_score,
            'player2_total_score': self.player2_score
        }
        
        self.game_history.append(round_result)
        
        # 通知双方本轮结果
        self.player1.receive_feedback(player2_action, player1_payoff, round_result)
        self.player2.receive_feedback(player1_action, player2_payoff, round_result)
        
        return round_result
    
    def get_game_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        return {
            'current_round': self.current_round,
            'max_rounds': self.max_rounds,
            'game_history': self.game_history[-10:],  # 最近10轮历史
            'my_total_score': 0,  # 这个会在各自的Agent中被替换
            'opponent_total_score': 0,  # 这个会在各自的Agent中被替换
            'available_actions': self.get_available_actions()
        }
    
    def is_game_over(self) -> bool:
        """判断游戏是否结束"""
        return self.current_round >= self.max_rounds
    
    def get_winner(self) -> Optional[str]:
        """获取获胜者"""
        if not self.is_game_over():
            return None
            
        if self.player1_score > self.player2_score:
            return self.player1_name
        elif self.player2_score > self.player1_score:
            return self.player2_name
        else:
            return "tie"
    
    def get_final_results(self) -> Dict[str, Any]:
        """获取最终结果"""
        return {
            'game_finished': self.is_game_over(),
            'total_rounds': self.current_round,
            'player1_name': self.player1_name,
            'player2_name': self.player2_name,
            'player1_final_score': self.player1_score,
            'player2_final_score': self.player2_score,
            'winner': self.get_winner(),
            'game_history': self.game_history,
            'cooperation_rates': self._calculate_cooperation_rates()
        }
    
    def _calculate_cooperation_rates(self) -> Dict[str, float]:
        """计算合作率"""
        if not self.game_history:
            return {'player1': 0.0, 'player2': 0.0}
            
        player1_cooperations = sum(1 for record in self.game_history 
                                  if record['player1_action'] == 'cooperate')
        player2_cooperations = sum(1 for record in self.game_history 
                                  if record['player2_action'] == 'cooperate')
        
        total_rounds = len(self.game_history)
        
        return {
            'player1': player1_cooperations / total_rounds,
            'player2': player2_cooperations / total_rounds
        }


class PrisonerDilemmaWorld:
    """囚徒困境世界模拟器"""
    
    def __init__(self, experiment_name: str = "prisoner_dilemma_experiment"):
        """初始化世界模拟器"""
        self.experiment_name = experiment_name
        self.experiment_logger = None
        self.games = []
        
        # 初始化实验日志系统
        self._init_experiment_logging()
        
    def _init_experiment_logging(self):
        """初始化实验日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_experiment_name = f"{self.experiment_name}_{timestamp}"
        
        # 初始化实验日志器（ExperimentLogger: experiment_dir = base_dir/experiment_name）
        self.experiment_logger = init_logger(
            experiment_name=unique_experiment_name,
            base_dir="./experiments"
        )
        
        log_print(f"囚徒困境实验开始: {unique_experiment_name}", level="INFO")
    
    def create_game(self, player1, player2, max_rounds: int = 100) -> PrisonerDilemmaGame:
        """创建新游戏"""
        game = PrisonerDilemmaGame(
            player1_name=player1.agent_name,
            player2_name=player2.agent_name,
            max_rounds=max_rounds
        )
        game.register_players(player1, player2)
        self.games.append(game)
        return game
    
    def run_game(self, game: PrisonerDilemmaGame) -> Dict[str, Any]:
        """运行完整游戏"""
        log_print(f"开始游戏: {game.player1_name} vs {game.player2_name}", level="INFO")
        
        round_results = []
        
        while not game.is_game_over():
            round_result = game.play_round()
            if round_result:
                round_results.append(round_result)
                
                # 记录每轮结果
                log_print(
                    f"第{round_result['round']}轮: "
                    f"{game.player1_name}:{round_result['player1_action']} vs "
                    f"{game.player2_name}:{round_result['player2_action']} -> "
                    f"收益: ({round_result['player1_payoff']}, {round_result['player2_payoff']})",
                    level="DEBUG"
                )
        
        final_results = game.get_final_results()
        
        log_print(
            f"游戏结束! 最终得分: "
            f"{game.player1_name}:{final_results['player1_final_score']} vs "
            f"{game.player2_name}:{final_results['player2_final_score']} "
            f"胜者: {final_results['winner']}",
            level="INFO"
        )
        
        return final_results
    
    def save_experiment_results(self, results: List[Dict[str, Any]]):
        """保存实验结果"""
        if self.experiment_logger:
            summary_data = {
                'experiment_name': self.experiment_name,
                'total_games': len(results),
                'games_results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            self.experiment_logger.save_experiment_summary(summary_data)
            log_print("实验结果已保存", level="INFO")