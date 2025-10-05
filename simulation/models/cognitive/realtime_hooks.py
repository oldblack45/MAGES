"""
实时保存钩子
在认知学习系统中添加实时保存功能
"""

from typing import Any, Dict
from datetime import datetime
from .experiment_logger import ExperimentLogger


class RealtimeHooks:
    """
    实时保存钩子类
    提供在认知数据更新时的实时保存功能
    """
    
    def __init__(self, experiment_logger: ExperimentLogger = None):
        self.experiment_logger = experiment_logger
    
    def on_world_cognition_update(self, agent_name: str, world_cognition_db, 
                                 update_type: str = "modified"):
        """
        世界认知更新时的钩子
        
        Args:
            agent_name: agent名称
            world_cognition_db: 世界认知数据库实例
            update_type: 更新类型 (added, modified, weight_updated)
        """
        if not self.experiment_logger:
            return
            
        try:
            # 记录更新事件
            update_event = {
                "agent_name": agent_name,
                "update_type": update_type,
                "timestamp": datetime.now().isoformat(),
                "total_recognitions": len(world_cognition_db.recognitions),
                "latest_recognition": None
            }
            
            # 如果有认知记录，包含最新的一条
            if world_cognition_db.recognitions:
                latest = world_cognition_db.recognitions[-1]
                update_event["latest_recognition"] = {
                    "action": latest.action,
                    "feedback": latest.feedback[:100] + "..." if len(latest.feedback) > 100 else latest.feedback,
                    "weight": latest.weight
                }
            
            # 记录到实验日志
            self.experiment_logger.log_game_event(
                f"world_cognition_{update_type}", update_event
            )
            
        except Exception as e:
            print(f"世界认知更新钩子出错: {e}")
    
    def on_agent_profile_update(self, agent_name: str, target_agent: str, 
                               profile_db, update_type: str = "modified"):
        """
        Agent侧写更新时的钩子
        
        Args:
            agent_name: 观察者agent名称
            target_agent: 目标agent名称  
            profile_db: Agent侧写数据库实例
            update_type: 更新类型 (added, modified, weight_updated)
        """
        if not self.experiment_logger:
            return
            
        try:
            # 记录更新事件
            update_event = {
                "observer_agent": agent_name,
                "target_agent": target_agent,
                "update_type": update_type,
                "timestamp": datetime.now().isoformat(),
                "total_profiles": len(profile_db.profiles),
                "latest_profile": None
            }
            
            # 如果有侧写记录，包含最新的一条
            if profile_db.profiles:
                latest = profile_db.profiles[-1]
                update_event["latest_profile"] = {
                    "action": latest.action,
                    "reaction": latest.reaction,
                    "weight": latest.weight
                }
            
            # 记录到实验日志
            self.experiment_logger.log_game_event(
                f"agent_profile_{update_type}", update_event
            )
            
        except Exception as e:
            print(f"Agent侧写更新钩子出错: {e}")
    
    def on_learning_update(self, agent_name: str, learning_result: Dict[str, Any]):
        """
        学习更新时的钩子
        
        Args:
            agent_name: agent名称
            learning_result: 学习结果数据
        """
        if not self.experiment_logger:
            return
            
        try:
            # 记录学习事件
            learning_event = {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "learning_result": learning_result
            }
            
            # 记录到实验日志
            self.experiment_logger.log_game_event("learning_update", learning_event)
            
        except Exception as e:
            print(f"学习更新钩子出错: {e}")
    
    def on_hypothesis_reasoning(self, agent_name: str, reasoning_data: Dict[str, Any]):
        """
        假设推理时的钩子
        
        Args:
            agent_name: agent名称
            reasoning_data: 推理过程数据
        """
        if not self.experiment_logger:
            return
            
        try:
            # 记录推理事件
            reasoning_event = {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "reasoning_depth": reasoning_data.get("reasoning_depth", 0),
                "satisfaction_score": reasoning_data.get("satisfaction_score", 0),
                "candidate_actions": reasoning_data.get("candidate_actions", []),
                "chosen_action": reasoning_data.get("chosen_action", None)
            }
            
            # 记录到实验日志
            self.experiment_logger.log_game_event("hypothesis_reasoning", reasoning_event)
            
        except Exception as e:
            print(f"假设推理钩子出错: {e}")
    
    def on_experiment_milestone(self, milestone_name: str, milestone_data: Dict[str, Any]):
        """
        实验里程碑事件钩子
        
        Args:
            milestone_name: 里程碑名称
            milestone_data: 里程碑数据
        """
        if not self.experiment_logger:
            return
            
        try:
            # 记录里程碑事件
            milestone_event = {
                "milestone_name": milestone_name,
                "timestamp": datetime.now().isoformat(),
                "data": milestone_data
            }
            
            # 记录到实验日志
            self.experiment_logger.log_game_event(f"milestone_{milestone_name}", milestone_event)
            
        except Exception as e:
            print(f"实验里程碑钩子出错: {e}")


class HookManager:
    """
    钩子管理器
    统一管理所有实时保存钩子
    """
    
    def __init__(self, experiment_logger: ExperimentLogger = None):
        self.hooks = RealtimeHooks(experiment_logger)
        self.enabled = True
    
    def enable_hooks(self):
        """启用钩子"""
        self.enabled = True
    
    def disable_hooks(self):
        """禁用钩子"""
        self.enabled = False
    
    def trigger_world_cognition_hook(self, *args, **kwargs):
        """触发世界认知更新钩子"""
        if self.enabled:
            self.hooks.on_world_cognition_update(*args, **kwargs)
    
    def trigger_agent_profile_hook(self, *args, **kwargs):
        """触发Agent侧写更新钩子"""
        if self.enabled:
            self.hooks.on_agent_profile_update(*args, **kwargs)
    
    def trigger_learning_hook(self, *args, **kwargs):
        """触发学习更新钩子"""
        if self.enabled:
            self.hooks.on_learning_update(*args, **kwargs)
    
    def trigger_reasoning_hook(self, *args, **kwargs):
        """触发推理钩子"""
        if self.enabled:
            self.hooks.on_hypothesis_reasoning(*args, **kwargs)
    
    def trigger_milestone_hook(self, *args, **kwargs):
        """触发里程碑钩子"""
        if self.enabled:
            self.hooks.on_experiment_milestone(*args, **kwargs)