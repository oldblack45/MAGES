"""
认知Agent类
集成世界认知建模、Agent侧写认知建模和假设推理功能的增强Agent
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from ..agents.LLMAgent import LLMAgent
from .world_cognition import WorldCognitionDB
from .agent_profile import MultiAgentProfileManager
from .hypothesis_reasoning import HypothesisReasoning, ReasoningResult
from .experiment_logger import ExperimentLogger, log_print


class CognitiveAgent(LLMAgent):
    """
    认知Agent：集成世界模型认知、Agent侧写认知和假设推理功能
    """
    
    def __init__(self, agent_name: str, other_agents: List[str] = None, 
                 experiment_logger: ExperimentLogger = None, **llm_kwargs):
        """
        初始化认知Agent
        
        Args:
            agent_name: Agent名称
            other_agents: 其他参与博弈的Agent列表
            experiment_logger: 实验日志器（必需）
            **llm_kwargs: LLM相关参数
        """
        super().__init__(agent_name=agent_name, **llm_kwargs)
        
        # 实验日志系统（必需）
        if not experiment_logger:
            raise ValueError("experiment_logger is required")
        self.experiment_logger = experiment_logger
        
        # 认知组件
        self.world_cognition = WorldCognitionDB(agent_name)
        self.agent_profiles = MultiAgentProfileManager(agent_name)
        
        # 设置实时日志器
        log_print(f"{agent_name}: 设置实时日志器", level="DEBUG")
        self.world_cognition.set_realtime_logger(self.experiment_logger)
        self.agent_profiles.set_realtime_logger(self.experiment_logger)
        
        # 添加其他Agent
        if other_agents:
            for other_agent in other_agents:
                log_print(f"{agent_name}: 添加目标Agent {other_agent}", level="DEBUG")
                self.agent_profiles.add_target_agent(other_agent)
                # 为新添加的profile设置日志器
                profile_db = self.agent_profiles.get_profile_db(other_agent)
                if profile_db:
                    log_print(f"{agent_name}: 为{other_agent}的profile设置日志器", level="DEBUG")
                    profile_db.set_realtime_logger(self.experiment_logger)
        
        # 历史记录
        self.reasoning_history = []  # 推理历史
        self.decision_history = []   # 决策历史
        
        # 假设推理引擎（传递决策历史引用）
        self.hypothesis_reasoning = HypothesisReasoning(
            agent_name, self.world_cognition, self.agent_profiles, self, self.decision_history
        )
        
        # 学习参数
        self.learning_enabled = True
        
        # 保存初始预训练数据的备份
        self.initial_world_training = []
        self.initial_agent_training = {}
    
    def set_reasoning_parameters(self, max_steps: int = 3, satisfaction_threshold: float = 0.6,
                               confidence_threshold: float = 0.5):
        """设置推理参数"""
        self.hypothesis_reasoning.set_reasoning_parameters(
            max_steps, satisfaction_threshold, confidence_threshold
        )

    def set_reasoning_feature_flags(self, enable_world_cognition: bool = True,
                                    enable_agent_profiles: bool = True):
        """设置推理中是否启用世界认知与角色侧写（用于消融）"""
        if hasattr(self, 'hypothesis_reasoning') and self.hypothesis_reasoning is not None:
            self.hypothesis_reasoning.use_world_cognition = enable_world_cognition
            self.hypothesis_reasoning.use_agent_profiles = enable_agent_profiles
    
    def pre_train_world_cognition(self, training_data: List[Dict]):
        """预训练世界认知"""
        # 保存初始训练数据备份
        self.initial_world_training = training_data.copy()
        self.world_cognition.pre_train(training_data)
        # 数据已通过实时保存机制自动保存
    
    def pre_train_agent_profiles(self, training_data: Dict[str, List[Dict]]):
        """预训练Agent侧写"""
        # 保存初始训练数据备份
        import copy
        self.initial_agent_training = copy.deepcopy(training_data)
        
        for target_agent, data in training_data.items():
            profile_db = self.agent_profiles.get_profile_db(target_agent)
            if profile_db is not None:
                log_print(f"{self.agent_name}: 预训练{target_agent}的侧写", level="DEBUG")
                profile_db.pre_train(data)
        # 数据已通过实时保存机制自动保存
    
    def cognitive_decision_making(self, candidate_actions: List[str], 
                                current_context: Dict[str, Any]) -> Tuple[str, ReasoningResult]:
        """
        基于认知的决策制定
        
        Args:
            candidate_actions: 候选行为列表
            current_context: 当前情景信息
            
        Returns:
            (最佳行为, 推理结果)
        """
        # 记录决策开始时间
        start_time = datetime.now()
        
        # 进行假设推理，从初始候选行为中选择最佳行为
        best_action, reasoning_result = self.hypothesis_reasoning.hypothesis_reasoning(
            candidate_actions, current_context
        )
        
        # 记录决策历史
        decision_record = {
            'timestamp': start_time.isoformat(),
            'candidate_actions': candidate_actions,
            'chosen_action': best_action,
            'context': current_context,
            'reasoning_summary': self.hypothesis_reasoning.get_reasoning_summary(reasoning_result),
            'satisfaction_score': reasoning_result.final_satisfaction_score,
            'reasoning_depth': reasoning_result.reasoning_depth
        }
        
        # 使用实验日志器记录决策
        self.experiment_logger.log_decision(self.agent_name, decision_record)
            
        # 更新历史记录
        self.decision_history.append(decision_record)
        self.reasoning_history.append(reasoning_result)
        
        return best_action, reasoning_result
    
    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取认知统计信息"""
        stats = {
            'agent_name': self.agent_name,
            'world_cognition': {
                'total_recognitions': len(self.world_cognition),
                'action_stats': self.world_cognition.get_action_statistics()
            },
            'agent_profiles': {},
            'decision_history': len(self.decision_history),
            'reasoning_history': len(self.reasoning_history)
        }
        
        for target_agent, profile_db in self.agent_profiles.profile_dbs.items():
            stats['agent_profiles'][target_agent] = {
                'total_profiles': len(profile_db),
                'dominant_strategy': profile_db.get_dominant_strategy(),
                'action_stats': profile_db.get_action_statistics()
            }
        
        return stats
    
    def export_cognition_report(self, output_file: str = None):
        """导出认知报告到实验总结"""
        stats = self.get_cognition_statistics()
        
        report_data = {
            "agent_name": self.agent_name,
            "generation_time": datetime.now().isoformat(),
            "world_cognition_stats": stats['world_cognition'],
            "agent_profile_stats": stats['agent_profiles'],
            "decision_count": len(self.reasoning_history),
            "reasoning_count": len(self.reasoning_history)
        }
        
        # 保存到实验总结
        self.experiment_logger.save_experiment_summary({
            f"{self.agent_name}_cognition_report": report_data
        })
        
        log_print("认知报告已保存到实验总结", level="INFO")
        return report_data