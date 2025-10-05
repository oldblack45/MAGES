"""
Chain-of-Thought Agent Implementation
基于思维链推理的Agent决策方法
"""

import json
from typing import Dict, List, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class CoTCountryAgent(LLMAgent):
    """基于Chain-of-Thought的国家Agent，继承LLMAgent"""
    
    def __init__(self, country_name: str, other_countries: List[str], 
                 game_attributes: Dict[str, int], experiment_logger: ExperimentLogger):
        
        # 初始化LLMAgent
        super().__init__(
            agent_name=f"CoT_{country_name}",
            has_chat_history=False,  # 简化实现，不使用对话历史
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
        
        # 可选行为列表
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]
    
    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """使用Chain-of-Thought进行博弈决策"""
        
        # 构建CoT提示词
        system_prompt = f"你是{self.country_name}的决策者，需要使用逐步思考的方式分析局势并做出决策。"
        
        user_prompt = f"""
        当前世界情况：{world_info}
        
        历史行动：{self._get_history_summary()}
        
        可用行动：{', '.join(self.available_actions)}
        
        请使用Chain-of-Thought方法，按以下步骤逐步思考：
        
        步骤1：分析当前局势
        - 识别关键信息和威胁
        - 评估紧张程度和风险等级
        
        步骤2：回顾历史经验
        - 总结之前决策的效果
        - 识别成功和失败的模式
        
        步骤3：评估各个行动选项
        - 分析每个可用行动的利弊
        - 预测每个行动的可能后果
        
        步骤4：考虑长期战略目标
        - 评估行动对长期目标的影响
        - 权衡短期收益和长期风险
        
        步骤5：做出最终决策
        - 综合所有分析结果
        - 选择最优行动并说明理由
        
        请严格按照上述步骤分析，最后返回JSON格式：
        {{{{
            "action": "选择的具体行动",
            "declaration": "公开声明",
            "reasoning_steps": ["步骤1的分析", "步骤2的分析", "步骤3的分析", "步骤4的分析", "步骤5的分析"]
        }}}}
        """
        
        try:
            # 调用LLMAgent的get_response方法
            response = self.get_response(
                user_template=user_prompt,
                new_system_template=system_prompt
            )
            
            # 解析LLM响应
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except:
                    response = self._parse_fallback_response(response)
            
            action = response.get('action', '外交谈判')
            declaration = response.get('declaration', f'{self.country_name}选择{action}')
            reasoning_steps = response.get('reasoning_steps', ['进行了CoT分析'])
            
            # 确保行动有效
            if action not in self.available_actions:
                action = '外交谈判'
                declaration = f'{self.country_name}选择外交途径解决问题'
            
        except Exception as e:
            self.experiment_logger.log_print(f"CoT决策失败: {e}", level="WARNING")
            # 使用备用方案
            action = '外交谈判'
            declaration = f'{self.country_name}采取谨慎的外交策略'
            reasoning_steps = ['采用了备用决策方案']
        
        # 记录决策
        self.action.append(action)
        self.declaration.append(declaration)
        
        # 返回兼容格式
        return {
            'action': action,
            'declaration': declaration,
            'reasoning_result': {
                'reasoning_steps': reasoning_steps,
                'method': 'chain-of-thought',
                'final_satisfaction_score': 0.7,
                'reasoning_depth': len(reasoning_steps)
            },
            'satisfaction_score': 0.7,
            'reasoning_depth': len(reasoning_steps)
        }
    
    def _get_history_summary(self) -> str:
        """获取历史行动摘要"""
        if not self.action:
            return "暂无历史行动"
        
        recent_actions = self.action[-3:]  # 最近3次行动
        return f"最近行动：{', '.join(recent_actions)}"
    
    def _parse_fallback_response(self, text: str) -> Dict:
        """备用响应解析"""
        # 尝试从文本中提取行动
        action = '外交谈判'
        for available_action in self.available_actions:
            if available_action in text:
                action = available_action
                break
        
        return {
            'action': action,
            'declaration': f'经过CoT分析，{self.country_name}决定{action}',
            'reasoning_steps': ['进行了系统性分析']
        }
    
    # 保持兼容性的简化接口
    def learn_from_interaction(self, action: str, world_feedback: str, 
                             opponent_actions: Dict[str, str], world_memory: Any = None):
        """简化的学习接口（不执行复杂学习）"""
        pass
    
    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取认知统计信息"""
        return {
            "world_cognition": {"total_recognitions": 0},
            "agent_profiles": {"default": {"total_profiles": 0}},
            "method": "chain-of-thought",
            "total_decisions": len(self.action)
        }
    
    def export_cognition_report(self):
        """导出认知报告"""
        report = {
            "method": "Chain-of-Thought",
            "country": self.country_name,
            "total_decisions": len(self.action),
            "actions": self.action[-10:] if self.action else [],
            "declarations": self.declaration[-5:] if self.declaration else []
        }
        
        try:
            import json
            from pathlib import Path
            report_path = Path(f"{self.country_name}_cot_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.experiment_logger.log_print(f"导出CoT报告失败: {e}", level="WARNING")