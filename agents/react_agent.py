"""
ReAct Agent Implementation
基于Reasoning and Acting交替推理的Agent决策方法
"""

import json
from typing import Dict, List, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class ReActCountryAgent(LLMAgent):
    """基于ReAct (Reasoning + Acting)的国家Agent，继承LLMAgent"""
    
    def __init__(self, country_name: str, other_countries: List[str], 
                 game_attributes: Dict[str, int], experiment_logger: ExperimentLogger):
        
        # 初始化LLMAgent
        super().__init__(
            agent_name=f"ReAct_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-235b-a22b-instruct-2507',
            online_track=False,
            json_format=True
        )
        
        self.country_name = country_name
        self.other_countries = other_countries
        self.game_attributes = game_attributes.copy()
        self.experiment_logger = experiment_logger
        
        # 简单的历史记录
        self.action = []
        self.declaration = []
        self.observations = []  # ReAct特有的观察记录
        
        # 可选行为列表
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]
    
    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """使用ReAct进行博弈决策"""
        
        # ReAct的核心：思考-行动-观察的循环
        final_decision = self._react_reasoning_loop(world_info)
        
        # 记录决策
        self.action.append(final_decision['action'])
        self.declaration.append(final_decision['declaration'])
        
        # 返回兼容格式
        return {
            'action': final_decision['action'],
            'declaration': final_decision['declaration'],
            'reasoning_result': {
                'reasoning_trace': final_decision.get('trace', []),
                'method': 'react',
                'final_satisfaction_score': 0.7,
                'reasoning_depth': len(final_decision.get('trace', []))
            },
            'satisfaction_score': 0.7,
            'reasoning_depth': len(final_decision.get('trace', []))
        }
    def dict_to_str(self, dict_data: Dict[str, Any]) -> str:
        """将字典转换为字符串"""
        dict_list = []
        for key, value in dict_data.items():
            if isinstance(value, dict):
                value_str = self.dict_to_str(value)
                dict_list.append(f"{key}: {value_str}")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        value_str = self.dict_to_str(item)
                        dict_list.append(f"{key}: {value_str},")
                    else:
                        value_str = str(item)
                        dict_list.append(f"{key}: {value_str}")
            else:
                dict_list.append(f"{key}: {value}")
        return ". ".join(dict_list)
    
    def _react_reasoning_loop(self, world_info: str) -> Dict[str, Any]:
        """执行ReAct推理循环"""
        
        system_prompt = f"你是{self.country_name}的决策者，使用ReAct方法：先思考(Think)，再行动(Act)，然后观察(Observe)结果。"
        
        # 第一轮：思考当前局势
        think1_prompt = f"""
        当前世界情况：{world_info}
        历史行动：{self._get_history_summary()}
        
        Think 1: 我需要分析什么关键信息来做出最佳决策？请分析当前局势的主要特点。
        """
        
        try:
            thought1= self.get_response(think1_prompt, system_prompt)
            thought1_str = self.dict_to_str(thought1)
            # 第一轮行动：信息收集
            act1_prompt = f"""
            基于思考：{thought1_str}
            
            Act 1: 我需要收集什么关键信息？请进行态势分析。
            可以分析：威胁等级、对手意图、环境变化等。
            返回格式：
            """
            
            analysis1 = self.get_response(act1_prompt, system_prompt)
            analysis1_str = self.dict_to_str(analysis1)
            # 第一轮观察
            observation1 = f"观察到：{self._generate_observation(analysis1, world_info)}"
            self.observations.append(observation1)
            
            # 第二轮：基于观察进一步思考
            think2_prompt = f"""
            之前的思考：{thought1_str}
            之前的分析：{analysis1_str}
            观察结果：{observation1}
            
            Think 2: 基于新的观察，我应该如何调整策略？
            """
            
            thought2 = self.get_response(think2_prompt, system_prompt)
            thought2_str = self.dict_to_str(thought2)
            # 最终决策
            final_prompt = f"""
            完整的推理过程：
            思考1：{thought1_str}
            行动1：{analysis1_str}  
            观察1：{observation1}
            思考2：{thought2_str}
            
            可用行动：{', '.join(self.available_actions)}
            
            基于完整的ReAct推理，做出最终决策。返回JSON格式：
            {{{{
                "action": "选择的具体行动",
                "declaration": "公开声明",
                "reasoning": "决策理由"
            }}}}
            """
            
            final_response = self.get_response(final_prompt, system_prompt)
            decision = self._parse_final_decision(final_response)
            
            # 添加推理轨迹
            decision['trace'] = [
                f"Think 1: {thought1_str[:100]}...",
                f"Act 1: {analysis1_str[:100]}...", 
                f"Observe 1: {observation1[:100]}...",
                f"Think 2: {thought2_str[:100]}...",
                f"Final Decision: {decision['action']}"
            ]
            
            return decision
            
        except Exception as e:
            self.experiment_logger.log_print(f"ReAct推理失败: {e}", level="WARNING")
            return {
                'action': '外交谈判',
                'declaration': f'{self.country_name}采取稳妥的外交策略',
                'trace': ['使用备用决策方案']
            }
    
    
    def _generate_observation(self, analysis: str, world_info: str) -> str:
        """生成观察结果"""
        observations = []
        
        if "紧张" in world_info or "紧张" in analysis:
            observations.append("地区紧张局势持续")
        if "合作" in world_info or "外交" in analysis:
            observations.append("存在外交合作机会")
        if "军事" in analysis:
            observations.append("军事活动值得关注")
            
        return "; ".join(observations) if observations else "局势复杂多变"
    
    def _parse_final_decision(self, response) -> Dict[str, str]:
        """解析最终决策"""
        try:
            if isinstance(response, str):
                response = json.loads(response)
            
            action = response.get('action', '外交谈判')
            if action not in self.available_actions:
                action = '外交谈判'
            
            return {
                'action': action,
                'declaration': response.get('declaration', f'{self.country_name}基于ReAct分析选择{action}'),
                'reasoning': response.get('reasoning', 'ReAct推理决策')
            }
            
        except:
            return {
                'action': '外交谈判',
                'declaration': f'{self.country_name}选择外交解决方案',
                'reasoning': 'ReAct备用决策'
            }
    
    def _get_history_summary(self) -> str:
        """获取历史摘要"""
        if not self.action:
            return "暂无历史记录"
        
        recent_actions = self.action[-3:]
        recent_observations = self.observations[-2:] if self.observations else []
        
        summary = f"行动：{', '.join(recent_actions)}"
        if recent_observations:
            summary += f"; 观察：{'; '.join(recent_observations)}"
        
        return summary
    
    # 保持兼容性的简化接口
    def learn_from_interaction(self, action: str, world_feedback: str, 
                             opponent_actions: Dict[str, str], world_memory: Any = None):
        """简化的学习接口"""
        # ReAct可以将反馈作为新的观察
        observation = f"行动{action}的反馈：{world_feedback}"
        self.observations.append(observation)
    
    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取认知统计信息"""
        return {
            "world_cognition": {"total_recognitions": 0},
            "agent_profiles": {"default": {"total_profiles": 0}},
            "method": "react",
            "total_decisions": len(self.action),
            "observations": len(self.observations)
        }
    
    def export_cognition_report(self):
        """导出认知报告"""
        report = {
            "method": "ReAct (Reasoning + Acting)",
            "country": self.country_name,
            "total_decisions": len(self.action),
            "actions": self.action[-10:] if self.action else [],
            "declarations": self.declaration[-5:] if self.declaration else [],
            "observations": self.observations[-10:] if self.observations else []
        }
        
        try:
            import json
            from pathlib import Path
            report_path = Path(f"{self.country_name}_react_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.experiment_logger.log_print(f"导出ReAct报告失败: {e}", level="WARNING")