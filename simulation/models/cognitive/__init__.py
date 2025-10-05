"""
认知建模模块

实现基于论文内容的多智能体博弈认知系统，包括：
1. 世界模型认知建模 - Action-Feedback-Experience三元组
2. Agent侧写认知建模 - Action-Reaction-Strategy-Experience四元组
3. 假设推理系统 - 多步预测推演和决策优化
"""

from .world_cognition import WorldRecognition, WorldCognitionDB
from .agent_profile import AgentProfile, AgentProfileDB, MultiAgentProfileManager
from .hypothesis_reasoning import HypothesisReasoning, ReasoningResult, SatisfactionLevel
from .cognitive_agent import CognitiveAgent

__all__ = [
    'WorldRecognition', 'WorldCognitionDB',
    'AgentProfile', 'AgentProfileDB', 'MultiAgentProfileManager', 
    'HypothesisReasoning', 'ReasoningResult', 'SatisfactionLevel',
    'CognitiveAgent'
]

__version__ = '1.0.0'
__author__ = 'MACE Team'