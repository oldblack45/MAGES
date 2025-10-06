"""
Hypothetical Minds Framework Implementation
基于心智理论（Theory of Mind）的战略适应框架

核心哲学：
- 通过显式模拟其他智能体的心智状态实现适应性智能
- 理论驱动（theory-driven）而非经验驱动
- 构建因果模型而非统计相关性
- 在线学习和策略适应

架构组件：
1. Perception - 感知模块
2. Memory - 记忆模块
3. Theory of Mind (ToM) - 心智理论模块（核心）
   - Hypothesis Generation: 假设生成
   - Hypothesis Evaluation: 假设评估
   - Hypothesis Refinement: 假设修正
4. Hierarchical Planning - 层级化规划
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class OpponentHypothesis:
    """对手策略假设"""

    def __init__(self, hypothesis_id: int, strategy_description: str,
                 predicted_behaviors: List[str], confidence: float = 0.5):
        self.hypothesis_id = hypothesis_id
        self.strategy_description = strategy_description  # 自然语言描述的策略假设
        self.predicted_behaviors = predicted_behaviors  # 预测的行为模式
        self.confidence = confidence  # 置信度得分
        self.prediction_history = []  # 预测历史 [(predicted, actual, correct)]
        self.creation_time = 0
        self.last_update_time = 0

    def evaluate_prediction(self, predicted_action: str, actual_action: str) -> bool:
        """评估预测准确性"""
        # 简化的匹配：检查预测是否包含实际行动的关键词
        correct = predicted_action.lower() in actual_action.lower() or \
                  actual_action.lower() in predicted_action.lower()

        self.prediction_history.append({
            'predicted': predicted_action,
            'actual': actual_action,
            'correct': correct
        })

        # 更新置信度：正确预测提升，错误预测降低
        if correct:
            self.confidence = min(1.0, self.confidence + 0.15)
        else:
            self.confidence = max(0.0, self.confidence - 0.20)

        return correct

    def get_accuracy(self) -> float:
        """获取预测准确率"""
        if not self.prediction_history:
            return 0.5

        correct_count = sum(1 for p in self.prediction_history if p['correct'])
        return correct_count / len(self.prediction_history)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hypothesis_id': self.hypothesis_id,
            'strategy_description': self.strategy_description,
            'predicted_behaviors': self.predicted_behaviors,
            'confidence': self.confidence,
            'accuracy': self.get_accuracy(),
            'prediction_count': len(self.prediction_history)
        }


class TheoryOfMindModule:
    """心智理论（ToM）模块 - Hypothetical Minds 的核心"""

    def __init__(self, agent_name: str, llm_agent: LLMAgent):
        self.agent_name = agent_name
        self.llm_agent = llm_agent

        # 对每个对手维护一组假设
        self.opponent_hypotheses: Dict[str, List[OpponentHypothesis]] = defaultdict(list)
        self.hypothesis_counter = 0

        # 记忆：观察到的对手行为历史
        self.observation_history: Dict[str, List[Dict]] = defaultdict(list)

        # 当前最佳假设（MAP估计）
        self.best_hypotheses: Dict[str, Optional[OpponentHypothesis]] = {}

    def observe_opponent_action(self, opponent_name: str, action: str,
                                context: str, round_num: int):
        """观察并记录对手行为"""
        observation = {
            'round': round_num,
            'action': action,
            'context': context,
            'timestamp': round_num
        }
        self.observation_history[opponent_name].append(observation)

    def generate_hypotheses(self, opponent_name: str, num_hypotheses: int = 3) -> List[OpponentHypothesis]:
        """假设生成：为对手的行为生成多种策略假设"""

        # 获取对手的行为历史
        history = self.observation_history[opponent_name]
        if not history:
            return []

        # 构建提示词
        history_summary = "\n".join([
            f"第{obs['round']}轮: {obs['action']} (情境: {obs['context'][:50]}...)"
            for obs in history[-5:]  # 最近5轮
        ])

        system_prompt = f"""你是一个具有心智理论能力的战略分析专家。
你的任务是推断对手的潜在策略和意图，而不仅仅是描述他们的行为。"""

        hypothesis_prompt = f"""
基于以下观察到的{opponent_name}的行为历史，生成{num_hypotheses}个不同的策略假设。

行为历史：
{history_summary}

每个假设应该：
1. 提供一个关于对手潜在策略的因果性解释（为什么这样做）
2. 预测对手在不同情境下可能采取的3个具体行动

请以JSON格式回复：
{{{{
    "hypotheses": [
        {{{{
            "strategy_description": "策略的自然语言描述，解释对手的意图和目标",
            "predicted_behaviors": ["预测行为1", "预测行为2", "预测行为3"]
        }}}},
        ...
    ]
}}}}
"""

        try:
            response = self.llm_agent.get_response(
                hypothesis_prompt,
                new_system_template=system_prompt,
                is_first_call=True
            )

            hypotheses = []
            if isinstance(response, dict) and 'hypotheses' in response:
                for hyp_data in response['hypotheses'][:num_hypotheses]:
                    self.hypothesis_counter += 1
                    hypothesis = OpponentHypothesis(
                        hypothesis_id=self.hypothesis_counter,
                        strategy_description=hyp_data.get('strategy_description', ''),
                        predicted_behaviors=hyp_data.get('predicted_behaviors', []),
                        confidence=0.5  # 初始置信度
                    )
                    hypothesis.creation_time = len(history)
                    hypotheses.append(hypothesis)

            return hypotheses

        except Exception as e:
            print(f"[ToM] 假设生成失败: {e}")
            return []

    def evaluate_hypotheses(self, opponent_name: str, actual_action: str):
        """假设评估：根据实际行为评估现有假设的准确性"""

        hypotheses = self.opponent_hypotheses[opponent_name]
        if not hypotheses:
            return

        for hypothesis in hypotheses:
            # 对每个假设，检查其预测行为是否与实际行为匹配
            best_match = None
            for predicted_behavior in hypothesis.predicted_behaviors:
                if self._behavior_matches(predicted_behavior, actual_action):
                    best_match = predicted_behavior
                    break

            if best_match:
                hypothesis.evaluate_prediction(best_match, actual_action)
            else:
                # 没有匹配的预测，记录为预测失败
                hypothesis.evaluate_prediction("未预测到此行为", actual_action)

    def _behavior_matches(self, predicted: str, actual: str) -> bool:
        """检查预测行为是否与实际行为匹配"""
        # 简化匹配：关键词重叠
        predicted_lower = predicted.lower()
        actual_lower = actual.lower()

        # 提取关键动作词
        key_actions = ['外交', '谈判', '军事', '演习', '封锁', '部署', '制裁',
                       '情报', '撤回', '通牒', '宣战', '核', '和平', '协议']

        for action in key_actions:
            if action in predicted_lower and action in actual_lower:
                return True

        return False

    def refine_hypotheses(self, opponent_name: str, max_hypotheses: int = 5):
        """假设修正：保留最佳假设，淘汰低置信度假设"""

        hypotheses = self.opponent_hypotheses[opponent_name]
        if not hypotheses:
            return

        # 按置信度排序
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        # 保留前max_hypotheses个
        self.opponent_hypotheses[opponent_name] = hypotheses[:max_hypotheses]

        # 更新最佳假设（MAP估计）
        if hypotheses:
            self.best_hypotheses[opponent_name] = hypotheses[0]

    def get_best_hypothesis(self, opponent_name: str) -> Optional[OpponentHypothesis]:
        """获取当前最佳假设"""
        return self.best_hypotheses.get(opponent_name)

    def update_tom_model(self, opponent_name: str, actual_action: str, round_num: int):
        """完整的ToM更新循环：生成-评估-修正"""

        # 1. 评估现有假设
        self.evaluate_hypotheses(opponent_name, actual_action)

        # 2. 如果假设不足或置信度普遍较低，生成新假设
        current_hypotheses = self.opponent_hypotheses[opponent_name]
        if len(current_hypotheses) < 3 or \
           (current_hypotheses and max(h.confidence for h in current_hypotheses) < 0.4):
            new_hypotheses = self.generate_hypotheses(opponent_name, num_hypotheses=2)
            self.opponent_hypotheses[opponent_name].extend(new_hypotheses)

        # 3. 修正假设（淘汰低质量假设）
        self.refine_hypotheses(opponent_name)

    def get_statistics(self) -> Dict[str, Any]:
        """获取ToM模块统计信息"""
        stats = {
            'total_hypotheses_generated': self.hypothesis_counter,
            'opponents_modeled': list(self.opponent_hypotheses.keys()),
            'opponent_details': {}
        }

        for opponent, hypotheses in self.opponent_hypotheses.items():
            stats['opponent_details'][opponent] = {
                'active_hypotheses': len(hypotheses),
                'best_hypothesis': self.best_hypotheses[opponent].to_dict() if opponent in self.best_hypotheses else None,
                'observation_count': len(self.observation_history[opponent])
            }

        return stats


class HypotheticalMindsAgent(LLMAgent):
    """Hypothetical Minds 框架的国家智能体"""

    def __init__(self, country_name: str, other_countries: List[str],
                 game_attributes: Dict[str, int], experiment_logger: ExperimentLogger):

        super().__init__(
            agent_name=f"HM_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-max',
            online_track=False,
            json_format=True
        )

        self.country_name = country_name
        self.other_countries = other_countries
        self.game_attributes = game_attributes.copy()
        self.experiment_logger = experiment_logger

        # 核心：心智理论模块
        self.tom_module = TheoryOfMindModule(country_name, self)

        # 记忆模块
        self.action_history = []
        self.declaration_history = []
        self.round_counter = 0

        # 行动空间
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]

    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """基于心智理论的决策流程"""

        self.round_counter += 1

        # 阶段1：感知 - 解析世界信息，提取对手行为
        opponent_actions = self._perceive_opponent_actions(world_info)

        # 阶段2：心智理论更新 - 为每个对手更新ToM模型
        for opponent, action in opponent_actions.items():
            if action:
                self.tom_module.observe_opponent_action(
                    opponent, action, world_info, self.round_counter
                )
                self.tom_module.update_tom_model(opponent, action, self.round_counter)

        # 阶段3：层级化规划 - 基于最佳假设制定策略
        action, reasoning = self._hierarchical_planning(world_info)

        # 阶段4：生成宣言
        declaration = self._generate_declaration(action, reasoning, world_info)

        # 记录
        self.action_history.append(action)
        self.declaration_history.append(declaration)

        return {
            'action': action,
            'declaration': declaration,
            'reasoning_result': {
                'tom_statistics': self.tom_module.get_statistics(),
                'reasoning': reasoning,
                'method': 'hypothetical_minds',
                'final_satisfaction_score': 0.75,
                'reasoning_depth': 3  # ToM: 生成-评估-修正
            },
            'satisfaction_score': 0.75,
            'reasoning_depth': 3
        }

    def _perceive_opponent_actions(self, world_info: str) -> Dict[str, str]:
        """感知模块：从世界信息中提取对手的行动"""
        opponent_actions = {}

        # 简单的文本解析，提取上一轮对手行动
        for opponent in self.other_countries:
            # 从world_info中查找对手的行动记录
            if f"{opponent}=" in world_info:
                # 提取行动信息
                start = world_info.find(f"{opponent}=") + len(f"{opponent}=")
                end = world_info.find(",", start)
                if end == -1:
                    end = world_info.find("\n", start)
                if end == -1:
                    end = len(world_info)

                action = world_info[start:end].strip()
                opponent_actions[opponent] = action

        return opponent_actions

    def _hierarchical_planning(self, world_info: str) -> Tuple[str, str]:
        """层级化规划：基于ToM模型制定对策"""

        # 收集所有对手的最佳假设
        opponent_models = {}
        for opponent in self.other_countries:
            best_hyp = self.tom_module.get_best_hypothesis(opponent)
            if best_hyp:
                opponent_models[opponent] = {
                    'strategy': best_hyp.strategy_description,
                    'confidence': best_hyp.confidence,
                    'predicted_behaviors': best_hyp.predicted_behaviors
                }

        system_prompt = f"""你是{self.country_name}的战略决策者，具有理解对手心智的能力。
你需要基于对对手策略的推断，制定最优的反制策略。"""

        # 构建对手模型描述
        opponent_model_text = ""
        if opponent_models:
            opponent_model_text = "\n".join([
                f"{opp}: {model['strategy']} (置信度: {model['confidence']:.2f})"
                for opp, model in opponent_models.items()
            ])
        else:
            opponent_model_text = "尚未建立对手心智模型"

        planning_prompt = f"""
当前局势：
{world_info}

你对对手的心智理论模型：
{opponent_model_text}

可选行动：
{', '.join(self.available_actions)}

历史行动：{', '.join(self.action_history[-3:]) if self.action_history else '无'}

基于你对对手策略的理解，制定最优行动。请以JSON格式回复：
{{{{
    "chosen_action": "从可选行动中选择",
    "reasoning": "决策理由，说明如何针对对手的推断策略进行反制或利用",
    "counter_strategy": "你的反制策略概述"
}}}}
"""

        try:
            response = self.get_response(
                planning_prompt,
                new_system_template=system_prompt,
                is_first_call=True
            )

            action = response.get('chosen_action', '外交谈判')
            reasoning = response.get('reasoning', '基于当前局势的判断')

            # 确保行动在可选范围内
            if action not in self.available_actions:
                action = '外交谈判'

            return action, reasoning

        except Exception as e:
            print(f"[HM] 规划失败: {e}")
            return '外交谈判', '默认保守策略'

    def _generate_declaration(self, action: str, reasoning: str, world_info: str) -> str:
        """生成行动宣言"""

        system_prompt = f"你是{self.country_name}的外交发言人。"

        declaration_prompt = f"""
请为以下行动撰写简短声明：

行动：{action}
决策理由：{reasoning}
当前局势：{world_info[:200]}...

要求：简洁有力，1-2句话。直接返回声明文本。
"""

        try:
            original_json_format = self.json_format
            self.json_format = False

            response = self.get_response(
                declaration_prompt,
                new_system_template=system_prompt,
                is_first_call=True
            )

            self.json_format = original_json_format

            if isinstance(response, dict):
                return response.get("declaration", f"{self.country_name}决定{action}")
            return str(response).strip()

        except Exception as e:
            print(f"[HM] 生成宣言失败: {e}")
            return f"{self.country_name}决定{action}，以维护国家利益。"

    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取认知统计信息"""
        stats = {
            "framework": "Hypothetical Minds",
            "total_decisions": len(self.action_history),
            "tom_statistics": self.tom_module.get_statistics(),
            "action_distribution": self._get_action_distribution()
        }
        return stats

    def _get_action_distribution(self) -> Dict[str, int]:
        """获取行动分布"""
        from collections import Counter
        return dict(Counter(self.action_history))

    def export_cognition_report(self):
        """导出认知报告"""
        report = {
            "agent_name": self.country_name,
            "framework": "Hypothetical Minds",
            "tom_module": self.tom_module.get_statistics(),
            "decision_history": {
                "actions": self.action_history,
                "declarations": self.declaration_history
            },
            "statistics": self.get_cognition_statistics()
        }

        if self.experiment_logger:
            import json
            report_file = f"{self.experiment_logger.experiment_dir}/{self.country_name}_hm_report.json"
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[HM] 导出报告失败: {e}")

        return report

    def learn_from_interaction(self, own_action: str, world_feedback: str,
                               other_actions: Dict[str, str], world_memory: Any):
        """从交互中学习（Hypothetical Minds是在线学习，通过ToM模块自动完成）"""
        # ToM模块已经在decision阶段完成在线学习
        # 这里可以添加额外的记忆巩固逻辑
        pass

    @property
    def action(self):
        """属性访问器"""
        return self.action_history

    @property
    def declaration(self):
        """属性访问器"""
        return self.declaration_history

