"""
Werewolf-Inspired Agent Implementation
基于狼人杀游戏机制的Agent决策方法
"""

import json
from typing import Dict, List, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class WerewolfCountryAgent(LLMAgent):
    """基于狼人杀游戏机制的国家Agent，继承LLMAgent"""
    
    def __init__(self, country_name: str, other_countries: List[str], 
                 game_attributes: Dict[str, int], experiment_logger: ExperimentLogger):
        
        # 初始化LLMAgent
        super().__init__(
            agent_name=f"Werewolf_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-max',
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
        
        # 狼人杀机制的简化记录
        self.opponent_analysis = {}  # 对手分析
        self.trust_levels = {}       # 信任度
        self.hidden_intentions = []   # 隐藏意图记录
        
        # 初始化对手信任度
        for opponent in other_countries:
            if opponent != country_name:
                self.trust_levels[opponent] = 0.5  # 初始中性信任
        
        # 可选行为列表
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]
    
    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """使用狼人杀机制进行博弈决策"""
        
        # 狼人杀的核心：身份推理、欺骗检测、策略隐藏
        decision = self._werewolf_analysis(world_info)
        
        # 记录决策
        self.action.append(decision['action'])
        self.declaration.append(decision['declaration'])
        if 'hidden_intention' in decision:
            self.hidden_intentions.append(decision['hidden_intention'])
        
        # 返回兼容格式
        return {
            'action': decision['action'],
            'declaration': decision['declaration'],
            'reasoning_result': {
                'werewolf_analysis': decision.get('analysis', {}),
                'method': 'werewolf-inspired',
                'final_satisfaction_score': 0.7,
                'reasoning_depth': 4  # 四个分析维度
            },
            'satisfaction_score': 0.7,
            'reasoning_depth': 4
        }
    def dict_to_str(self, dict_data: Dict[str, Any]) -> str:
        """将字典转换为字符串"""
        dict_list = []
        for key, value in dict_data.items():
            if isinstance(value, dict):
                value_str = self.dict_to_str(value)
                dict_list.append(f"{key}: {value_str}")
            else:
                dict_list.append(f"{key}: {value}")
        return ". ".join(dict_list)
    
    def _werewolf_analysis(self, world_info: str) -> Dict[str, Any]:
        """执行狼人杀分析"""
        
        system_prompt = f"你是{self.country_name}的战略决策者，擅长身份推理、欺骗检测和策略隐藏，就像狼人杀游戏中的高手。"
        
        # 第一步：身份分析
        identity_prompt = f"""
        当前情况：{world_info}
        历史记录：{self._get_history_summary()}
        
        请分析对手的真实立场和策略倾向：
        1. 对手最可能是什么类型：激进派、温和派、实用派？
        2. 对手的声明和行动是否一致？
        3. 对手下一步最可能采取什么行动？
        
        """
        
        try:
            identity_response = self.get_response(identity_prompt, system_prompt)
            identity_response_str = self.dict_to_str(identity_response)
            # 第二步：欺骗检测
            deception_prompt = f"""
            基于身份分析：{identity_response_str}
            
            请检测对手可能的欺骗行为：
            1. 声明与行动是否矛盾？
            2. 是否存在隐藏意图？
            3. 可信度如何评估？
            
            返回JSON格式，包含trust_level（0-1之间的数值）。
            """
            
            deception_response = self.get_response(deception_prompt, system_prompt)
            # 更新信任度
            self._update_trust(deception_response)
            trust_level_list = []
            for key, value in self.trust_levels.items():
                trust_level_list.append(f"{key}: {value}")
            trust_level_str = ". ".join(trust_level_list)
            # 第三步：策略隐藏
            concealment_prompt = f"""
            我的真实意图是维护{self.country_name}的最大利益。
            当前信任状况：{trust_level_str}
            
            请设计策略隐藏方案：
            1. 我应该如何掩盖真实意图？
            2. 向对手传递什么误导信息？
            3. 如何在合作中获取更多利益？
            
            可用行动：{', '.join(self.available_actions)}
            
            返回JSON格式：
            {{{{
                "action": "选择的行动",
                "declaration": "公开声明",
                "hidden_intention": "真实意图",
                "misleading_element": "误导成分"
            }}}}
            """
            
            final_response = self.get_response(concealment_prompt, system_prompt)
            final_decision = self._parse_final_decision(final_response)
            
            # 组合分析结果
            final_decision['analysis'] = {
                'identity_analysis': identity_response,
                'deception_analysis': deception_response,
                'trust_levels': self.trust_levels.copy()
            }
            
            return final_decision
            
        except Exception as e:
            self.experiment_logger.log_print(f"Werewolf分析失败: {e}", level="WARNING")
            return {
                'action': '外交谈判',
                'declaration': f'{self.country_name}希望通过对话解决分歧',
                'hidden_intention': '试探对手真实意图',
                'analysis': {'method': 'fallback'}
            }
    
    def _parse_response(self, response) -> Dict:
        """解析LLM响应"""
        try:
            if isinstance(response, str):
                return {'content': response}
            elif isinstance(response, dict):
                return response
            else:
                return {'content': str(response)}
        except:
            return {'content': '分析结果'}
    
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
                'declaration': response.get('declaration', f'{self.country_name}采取{action}策略'),
                'hidden_intention': response.get('hidden_intention', '维护国家利益'),
                'misleading_element': response.get('misleading_element', '适度隐藏真实意图')
            }
            
        except:
            return {
                'action': '外交谈判',
                'declaration': f'{self.country_name}选择外交解决方案',
                'hidden_intention': '稳妥行事',
                'misleading_element': '显示合作态度'
            }
    
    def _update_trust(self, deception_analysis: Dict):
        """更新信任度"""
        try:
            trust_level = deception_analysis.get('trust_level', 0.5)
            if isinstance(trust_level, (int, float)):
                for opponent in self.trust_levels:
                    # 简单的信任度更新
                    self.trust_levels[opponent] = min(1.0, max(0.1, trust_level))
        except:
            pass  # 保持原有信任度
    
    def _get_history_summary(self) -> str:
        """获取历史摘要"""
        if not self.action:
            return "暂无历史记录"
        
        recent_actions = self.action[-3:]
        recent_intentions = self.hidden_intentions[-2:] if self.hidden_intentions else []
        
        summary = f"行动：{', '.join(recent_actions)}"
        if recent_intentions:
            summary += f"; 策略：{', '.join(recent_intentions)}"
        
        return summary
    
    # 保持兼容性的简化接口
    def learn_from_interaction(self, action: str, world_feedback: str, 
                             opponent_actions: Dict[str, str], world_memory: Any = None):
        """简化的学习接口"""
        # 根据反馈调整信任度
        if "合作" in world_feedback or "缓解" in world_feedback:
            for opponent in self.trust_levels:
                self.trust_levels[opponent] = min(0.9, self.trust_levels[opponent] + 0.1)
        elif "对抗" in world_feedback or "紧张" in world_feedback:
            for opponent in self.trust_levels:
                self.trust_levels[opponent] = max(0.1, self.trust_levels[opponent] - 0.1)
    
    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取认知统计信息"""
        return {
            "world_cognition": {"total_recognitions": 0},
            "agent_profiles": {"identity_tracker": {"total_profiles": len(self.opponent_analysis)}},
            "method": "werewolf-inspired",
            "total_decisions": len(self.action),
            "trust_relationships": len(self.trust_levels),
            "hidden_strategies": len(self.hidden_intentions)
        }
    
    def export_cognition_report(self):
        """导出认知报告"""
        report = {
            "method": "Werewolf-Inspired Strategy",
            "country": self.country_name,
            "total_decisions": len(self.action),
            "actions": self.action[-10:] if self.action else [],
            "declarations": self.declaration[-5:] if self.declaration else [],
            "trust_levels": self.trust_levels,
            "hidden_intentions": self.hidden_intentions[-5:] if self.hidden_intentions else [],
            "opponent_analysis": self.opponent_analysis
        }
        
        try:
            import json
            from pathlib import Path
            report_path = Path(f"{self.country_name}_werewolf_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.experiment_logger.log_print(f"导出Werewolf报告失败: {e}", level="WARNING")