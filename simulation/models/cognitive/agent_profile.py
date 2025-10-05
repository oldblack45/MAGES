"""
Agent侧写认知建模模块
实现基于四元组（Action-Reaction-Strategy-Experience）的Agent侧写认知库
"""

import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from .experiment_logger import ExperimentLogger, log_print

class AgentProfile:
    """Agent侧写四元组：Action-Reaction-Strategy-Experience"""
    
    def __init__(self, action: str, reaction: str, strategy: str, experience: str, weight: float = 1.0):
        self.action = action           # 自己的行为
        self.reaction = reaction       # 对方的反应
        self.strategy = strategy       # 从反应中体现的对方博弈策略
        self.experience = experience   # 应对该策略的经验
        self.weight = weight          # 权重，用于强化或削弱
        self.created_time = datetime.now()
        self.update_count = 0         # 更新次数
        
    def update_weight(self, adjustment: float, min_weight: float = 0.1, max_weight: float = 3.0):
        """更新权重，限制在合理范围内"""
        self.weight = max(min_weight, min(max_weight, self.weight + adjustment))
        self.update_count += 1
        
    def to_dict(self) -> dict:
        """转换为字典格式便于序列化"""
        return {
            'action': self.action,
            'reaction': self.reaction,
            'strategy': self.strategy,
            'experience': self.experience,
            'weight': self.weight,
            'created_time': self.created_time.isoformat(),
            'update_count': self.update_count
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建实例"""
        profile = cls(
            action=data['action'],
            reaction=data['reaction'],
            strategy=data['strategy'],
            experience=data['experience'],
            weight=data['weight']
        )
        profile.created_time = datetime.fromisoformat(data['created_time'])
        profile.update_count = data['update_count']
        return profile


class AgentProfileDB:
    """Agent侧写认知库：管理对其他Agent的认知"""
    
    def __init__(self, agent_name: str, target_agent: str):
        self.agent_name = agent_name      # 自己的名字
        self.target_agent = target_agent  # 目标Agent的名字
        self.profiles: List[AgentProfile] = []
        self.weight_threshold = 0.3       # 权重低于此值时修改反应
        
        # 实时保存相关
        self.realtime_logger: ExperimentLogger = None  # 实验日志器
        self.auto_save_enabled = True  # 是否启用实时保存
        log_print(f"AgentProfileDB({agent_name}->{target_agent}): 初始化完成", level="DEBUG")
        
    def set_realtime_logger(self, logger):
        """设置实时日志器"""
        self.realtime_logger = logger
    
    def _realtime_save(self):
        """实时保存侧写数据"""        
        if self.auto_save_enabled and self.realtime_logger:
            try:
                # 准备保存的数据
                data = {
                    'agent_name': self.agent_name,
                    'target_agent': self.target_agent,
                    'profiles': [p.to_dict() for p in self.profiles],
                    'weight_threshold': self.weight_threshold,
                    'total_count': len(self.profiles)
                }
                # 实时保存到实验日志
                self.realtime_logger.log_cognition_update(
                    self.agent_name, f'agent_profile_{self.target_agent}', data
                )
                log_print(f"侧写实时保存成功: {self.agent_name} -> {self.target_agent}", level="DEBUG")
            except Exception as e:
                log_print(f"实时保存Agent侧写时出错: {e}", level="ERROR")
        else:
            if not self.auto_save_enabled:
                log_print("自动保存未启用", level="DEBUG")
            if not self.realtime_logger:
                log_print("实时日志器未设置", level="DEBUG")
    
    def add_profile(self, action: str, reaction: str, strategy: str, experience: str, weight: float = 1.0):
        """添加新的Agent侧写"""
        profile = AgentProfile(action, reaction, strategy, experience, weight)
        self.profiles.append(profile)
        # 实时保存
        self._realtime_save()
        
    def predict_reaction(self, action: str) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """
        根据行为预测对方反应、策略和应对经验
        返回：(predicted_reaction, predicted_strategy, predicted_experience, confidence)
        """
        matching_profiles = [p for p in self.profiles if p.action == action]
        
        if not matching_profiles:
            log_print(f"没有找到{action}的侧写", level="WARNING")
            return None, None, None, 0.0
            
        # 选择权重最高的侧写
        best_profile = max(matching_profiles, key=lambda x: x.weight)
        
        # 计算置信度（基于权重和匹配数量）
        confidence = min(1.0, best_profile.weight / max(1.0, len(matching_profiles)))
        
        return best_profile.reaction, best_profile.strategy, best_profile.experience, confidence

    def predict_reaction_with_fallback(self, action: str, llm_agent=None) -> Tuple[str, str, str, float]:
        """
        带LLM降级机制的预测Agent反应
        返回：(predicted_reaction, predicted_strategy, predicted_experience, confidence)
        """
        # 第1层：基于侧写库预测
        reaction, strategy, experience, confidence = self.predict_reaction(action)
        
        if reaction is not None:
            return reaction, strategy, experience, confidence
        
        # 第2层：LLM降级预测
        if llm_agent:
            try:
                llm_result = self._llm_predict_reaction(action, llm_agent)
                log_print("使用LLM预测Agent反应成功并添加侧写", level="DEBUG")
                self.add_profile(action, llm_result['reaction'], llm_result['strategy'], llm_result['experience'], 1.0)
                return llm_result['reaction'], llm_result['strategy'], llm_result['experience'], 0.3
            except Exception as e: 
                log_print(f"LLM降级预测Agent反应失败: {e}", level="ERROR")
        
        # 第3层：通用默认预测
        default_reaction = None
        default_strategy = None
        default_experience = None
        return default_reaction, default_strategy, default_experience, 0.1

    def _llm_predict_reaction(self, action: str, llm_agent) -> Dict[str, str]:
        available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]
        """使用LLM预测Agent反应"""
        log_print(f"使用LLM预测{self.target_agent}对{action}的反应", level="INFO")
        prompt = f"""
        请预测当我方采取"{action}"行为时，{self.target_agent}可能的反应和策略：
        
        我方行为：{action}
        目标对象：{self.target_agent}
        可选反应：
        {available_actions}
        请分析：
        1. {self.target_agent}可能采取什么反应
        2. 这种反应体现了什么策略思路
        3. 我方应该如何应对这种反应
        
        回答格式：
        {{{{
            "reaction": "预测的对方反应（可选反应中的一个）",
            "strategy": "体现的策略特点（对方的博弈思路）",
            "experience": "应对经验（我方如何处理这种反应）"
        }}}}
        """
        
        try:
            response = llm_agent.get_response(prompt)
            if isinstance(response, dict):
                return {
                    'reaction': response.get('reaction', None),
                    'strategy': response.get('strategy', None),
                    'experience': response.get('experience', None)
                }
            else:
                return {
                    'reaction': None,
                    'strategy': None,
                    'experience': None
                }
        except Exception as e:
            log_print(f"LLM预测Agent反应失败: {e}", level="ERROR")
            return {
                'reaction': None,
                'strategy': None,
                'experience': None
            }
    
    def update_experience_and_strategies(self, profile: AgentProfile, llm_agent):
        """更新单个侧写的策略和经验"""
            
        prompt = f"""
        请分析以下交互情况中对方({self.target_agent})的策略和应对经验：
        
        我方行为：{profile.action}
        对方反应：{profile.reaction}
        
        请分析：
        1. 对方的博弈策略特点
        2. 针对该策略的应对经验
        
        回答格式：
        {{{{
            "strategy": "策略分析，仅为一个字符串，不要有其他内容",
            "experience": "应对经验，仅为一个字符串，不要有其他内容"
        }}}}
        """
        
        try:
            response = llm_agent.get_response(prompt)
            if isinstance(response, dict):
                profile.strategy = response.get('strategy', f"针对{profile.action}采取{profile.reaction}")
                profile.experience = response.get('experience', f"应对{self.target_agent}的{profile.reaction}反应")
            else:
                # 如果返回的不是字典，尝试解析或使用默认值
                profile.strategy = f"针对{profile.action}采取{profile.reaction}"
                profile.experience = f"应对{self.target_agent}的{profile.reaction}反应"
        except Exception as e:
            log_print(f"更新策略时出错: {e}", level="ERROR")
            profile.strategy = f"针对{profile.action}采取{profile.reaction}"
            profile.experience = f"应对{self.target_agent}的{profile.reaction}反应"
        
        # 更新后实时保存
        self._realtime_save()
    
    def get_dominant_strategy(self) -> Optional[str]:
        """获取目标Agent的主导策略"""
        if not self.profiles:
            return None
            
        # 根据权重和出现频率分析主导策略
        strategy_weights = {}
        for profile in self.profiles:
            strategy = profile.strategy
            if strategy not in strategy_weights:
                strategy_weights[strategy] = 0
            strategy_weights[strategy] += profile.weight
            
        if strategy_weights:
            return max(strategy_weights, key=strategy_weights.get)
        return None
    
    def get_action_statistics(self) -> Dict[str, Dict]:
        """获取各种行为对应反应的统计信息"""
        stats = {}
        for profile in self.profiles:
            action = profile.action
            if action not in stats:
                stats[action] = {
                    'count': 0,
                    'avg_weight': 0.0,
                    'reactions': [],
                    'strategies': []
                }
            
            stats[action]['count'] += 1
            stats[action]['avg_weight'] += profile.weight
            stats[action]['reactions'].append(profile.reaction)
            stats[action]['strategies'].append(profile.strategy)
            
        # 计算平均权重
        for action in stats:
            stats[action]['avg_weight'] /= stats[action]['count']
            
        return stats
    
    def pre_train(self, training_data: List[Dict]):
        """预认知训练：预先植入Agent侧写"""
        for data in training_data:
            self.add_profile(
                action=data['action'],
                reaction=data['reaction'],
                strategy=data['strategy'],
                experience=data['experience'],
                weight=data.get('weight', 1.0)
            )

    def save_to_file(self, filepath: str):
        """保存侧写库到文件"""
        data = {
            'agent_name': self.agent_name,
            'target_agent': self.target_agent,
            'profiles': [p.to_dict() for p in self.profiles],
            'weight_threshold': self.weight_threshold
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str):
        """从文件加载侧写库"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.agent_name = data['agent_name']
            self.target_agent = data['target_agent']
            self.weight_threshold = data['weight_threshold']
            self.profiles = [AgentProfile.from_dict(p) for p in data['profiles']]
            
        except Exception as e:
            log_print(f"加载Agent侧写库失败: {e}")
    
    def __len__(self):
        return len(self.profiles)
    
    def __str__(self):
        return f"AgentProfileDB(agent={self.agent_name}, target={self.target_agent}, profiles={len(self.profiles)})"


class MultiAgentProfileManager:
    """多Agent侧写管理器：管理对多个其他Agent的侧写认知库"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.profile_dbs: Dict[str, AgentProfileDB] = {}
        
        # 实时保存相关
        self.realtime_logger = None  # 实验日志器
        
    def set_realtime_logger(self, logger):
        """设置实时日志器，并传递给所有现有的侧写库"""
        self.realtime_logger = logger
        for target_agent, profile_db in self.profile_dbs.items():
            profile_db.set_realtime_logger(logger)
    
    def add_target_agent(self, target_agent: str):
        """添加目标Agent"""
        if target_agent not in self.profile_dbs:
            profile_db = AgentProfileDB(self.agent_name, target_agent)
            # 总是设置日志器，即使是None
            profile_db.set_realtime_logger(self.realtime_logger)
            self.profile_dbs[target_agent] = profile_db
    
    def get_profile_db(self, target_agent: str) -> Optional[AgentProfileDB]:
        """获取指定目标Agent的侧写库"""
        return self.profile_dbs.get(target_agent)
    
    def predict_all_reactions(self, action: str) -> Dict[str, Tuple]:
        """预测所有目标Agent对指定行为的反应"""
        predictions = {}
        for target_agent, profile_db in self.profile_dbs.items():
            predictions[target_agent] = profile_db.predict_reaction(action)
        return predictions
    
    def get_all_dominant_strategies(self) -> Dict[str, str]:
        """获取所有目标Agent的主导策略"""
        strategies = {}
        for target_agent, profile_db in self.profile_dbs.items():
            strategy = profile_db.get_dominant_strategy()
            if strategy:
                strategies[target_agent] = strategy
        return strategies
    
    def save_all_to_dir(self, directory: str):
        """保存所有侧写库到目录"""
        import os
        os.makedirs(directory, exist_ok=True)
        for target_agent, profile_db in self.profile_dbs.items():
            filepath = os.path.join(directory, f"{self.agent_name}_to_{target_agent}_profile.json")
            profile_db.save_to_file(filepath)
    
    def load_all_from_dir(self, directory: str):
        """从目录加载所有侧写库"""
        import os
        if not os.path.exists(directory):
            return
            
        for filename in os.listdir(directory):
            if filename.endswith('_profile.json') and filename.startswith(f"{self.agent_name}_to_"):
                target_agent = filename.replace(f"{self.agent_name}_to_", "").replace("_profile.json", "")
                filepath = os.path.join(directory, filename)
                
                if target_agent not in self.profile_dbs:
                    self.profile_dbs[target_agent] = AgentProfileDB(self.agent_name, target_agent)
                self.profile_dbs[target_agent].load_from_file(filepath)