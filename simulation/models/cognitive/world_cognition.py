"""
世界模型认知建模模块
实现基于三元组（Action-Feedback-Experience）的世界认知库
"""

import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from .experiment_logger import ExperimentLogger, log_print

class WorldRecognition:
    """世界认知三元组：Action-Feedback-Experience"""
    
    def __init__(self, action: str, feedback: str, experience: str, weight: float = 1.0):
        self.action = action           # 行为输入
        self.feedback = feedback       # 环境输出/反馈
        self.experience = experience   # 从中形成的经验
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
            'feedback': self.feedback, 
            'experience': self.experience,
            'weight': self.weight,
            'created_time': self.created_time.isoformat(),
            'update_count': self.update_count
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建实例"""
        recognition = cls(
            action=data['action'],
            feedback=data['feedback'],
            experience=data['experience'],
            weight=data['weight']
        )
        recognition.created_time = datetime.fromisoformat(data['created_time'])
        recognition.update_count = data['update_count']
        return recognition


class WorldCognitionDB:
    """世界认知库：管理世界认知三元组"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.recognitions: List[WorldRecognition] = []
        self.weight_threshold = 0.3  # 权重低于此值时修改反馈
        
        # 实时保存相关
        self.realtime_logger: ExperimentLogger = None  # 实验日志器
        self.auto_save_enabled = True  # 是否启用实时保存
        log_print(f"WorldCognitionDB({agent_name}): 初始化完成", level="DEBUG")
        
    def set_realtime_logger(self, logger):
        """设置实时日志器"""
        self.realtime_logger = logger
    
    def _realtime_save(self):
        """实时保存认知数据"""
        
        if self.auto_save_enabled and self.realtime_logger:
            try:
                # 准备保存的数据
                data = {
                    'agent_name': self.agent_name,
                    'recognitions': [r.to_dict() for r in self.recognitions],
                    'weight_threshold': self.weight_threshold,
                    'total_count': len(self.recognitions)
                }
                # 实时保存到实验日志
                self.realtime_logger.log_cognition_update(
                    self.agent_name, 'world_cognition', data
                )
                log_print(f"世界认知实时保存成功: {self.agent_name}", level="DEBUG")
            except Exception as e:
                log_print(f"实时保存世界认知时出错: {e}", level="ERROR")
        else:
            if not self.auto_save_enabled:
                log_print("自动保存未启用", level="DEBUG")
            if not self.realtime_logger:
                log_print("实时日志器未设置", level="DEBUG")
    
    def add_recognition(self, action: str, feedback: str, experience: str, weight: float = 1.0):
        """添加新的世界认知"""
        recognition = WorldRecognition(action, feedback, experience, weight)
        self.recognitions.append(recognition)
        # 实时保存
        self._realtime_save()
    def update_world_experiences(self, recognition: WorldRecognition, llm_agent):
        """更新单个世界认知的经验"""
        
        prompt = f"""
        请为以下行为-反馈对生成经验总结：
        
        行为：{recognition.action}
        反馈：{recognition.feedback}
        
        请总结这次经历能得到什么经验教训，用于指导未来决策。
        回复格式：
        {{{{
            "experience": "经验总结，仅为一个字符串，不要有其他内容"
        }}}}
        """
        
        try:
            response = llm_agent.get_response(prompt)
            if isinstance(response, dict):
                recognition.experience = response.get('experience', str(response))
            else:
                recognition.experience = str(response)
        except Exception as e:
            print(f"更新世界认知经验时出错: {e}")
            recognition.experience = f"执行{recognition.action}得到{recognition.feedback}"
        
        # 更新后实时保存
        self._realtime_save()
    
    def predict_feedback(self, action: str) -> Tuple[Optional[str], Optional[str], float]:
        """
        根据行为预测反馈和经验
        返回：(predicted_feedback, predicted_experience, confidence)
        """
        matching_recognitions = [r for r in self.recognitions if r.action == action]
        
        if not matching_recognitions:
            # print(f"没有找到{action}的世界反馈")
            return None, None, 0.0
            
        # 选择权重最高的认知
        best_recognition = max(matching_recognitions, key=lambda x: x.weight)
        
        # 计算置信度（基于权重和匹配数量）
        total_weight = sum(r.weight for r in matching_recognitions)
        confidence = min(1.0, best_recognition.weight / max(1.0, len(matching_recognitions)))
        
        return best_recognition.feedback, best_recognition.experience, confidence

    def predict_feedback_with_fallback(self, action: str, llm_agent=None) -> Tuple[str, str, float]:
        """
        带LLM降级机制的预测反馈
        返回：(predicted_feedback, predicted_experience, confidence)
        """
        # 第1层：基于认知库预测
        feedback, experience, confidence = self.predict_feedback(action)
        
        if feedback is not None:
            return feedback, experience, confidence
        
        # 第2层：LLM降级预测
        if llm_agent:
            try:
                llm_result = self._llm_predict_feedback(action, llm_agent)
                log_print("使用LLM预测世界反馈成功并添加认知", level="DEBUG")
                self.add_recognition(action, llm_result['feedback'], llm_result['experience'], 1.0)
                return llm_result['feedback'], llm_result['experience'], 0.3  # 降级预测置信度较低
            except Exception as e: 
                log_print(f"LLM降级预测失败: {e}", level="ERROR")
        
        # 第3层：通用默认预测
        default_feedback = None
        default_experience = None
        return default_feedback, default_experience, 0.1

    def _llm_predict_feedback(self, action: str, llm_agent) -> Dict[str, str]:
        """使用LLM预测世界反馈"""
        log_print(f"使用LLM预测{action}的世界反馈", level="INFO")
        prompt = f"""
        请预测执行"{action}"这个行为可能得到的世界反馈和经验：
        
        行为：{action}
        
        请分析：
        1. 这个行为可能导致什么样的世界反馈/后果
        2. 从这个行为中能获得什么经验教训
        
        回答格式：
        {{{{
            "feedback": "预测的世界反馈（一句话描述可能的后果）,类似于短期效果：***，长期影响：***",
            "experience": "从中获得的经验（指导未来决策的教训）"
        }}}}
        """
        
        try:
            response = llm_agent.get_response(prompt)
            if isinstance(response, dict):
                return {
                    'feedback': response.get('feedback', None),
                    'experience': response.get('experience', None)
                }
            else:
                return {
                    'feedback': None,
                    'experience': None
                }
        except Exception as e:
            log_print(f"LLM预测失败: {e}", level="ERROR")
            return {
                'feedback': None,
                'experience': None
            }
    
    def get_action_statistics(self) -> Dict[str, Dict]:
        """获取各种行为的统计信息"""
        stats = {}
        for recognition in self.recognitions:
            action = recognition.action
            if action not in stats:
                stats[action] = {
                    'count': 0,
                    'avg_weight': 0.0,
                    'feedbacks': []
                }
            
            stats[action]['count'] += 1
            stats[action]['avg_weight'] += recognition.weight
            stats[action]['feedbacks'].append(recognition.feedback)
            
        # 计算平均权重
        for action in stats:
            stats[action]['avg_weight'] /= stats[action]['count']
            
        return stats
    
    def pre_train(self, training_data: List[Dict]):
        """预认知训练：预先植入世界认识"""
        for data in training_data:
            self.add_recognition(
                action=data['action'],
                feedback=data['feedback'], 
                experience=data['experience'],
                weight=data.get('weight', 1.0)
            )

    def save_to_file(self, filepath: str):
        """保存认知库到文件"""
        data = {
            'agent_name': self.agent_name,
            'recognitions': [r.to_dict() for r in self.recognitions],
            'weight_threshold': self.weight_threshold
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str):
        """从文件加载认知库"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.agent_name = data['agent_name']
            self.weight_threshold = data['weight_threshold']
            self.recognitions = [WorldRecognition.from_dict(r) for r in data['recognitions']]
            
        except Exception as e:
            log_print(f"加载认知库失败: {e}", level="ERROR")
    
    def __len__(self):
        return len(self.recognitions)
    
    def __str__(self):
        return f"WorldCognitionDB(agent={self.agent_name}, recognitions={len(self.recognitions)})"