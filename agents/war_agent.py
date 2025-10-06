"""
WarAgent Framework Implementation
基于宏观层面的历史冲突模拟框架

核心哲学：
- 高层级的历史事件复现
- 国家作为整体决策单元
- 简化的国家-行动体模型
- 制度化的决策输出模型

架构组件：
1. Country Agent - 国家智能体，基于国家概况驱动
2. Secretary Agent - 秘书智能体，验证行动合理性
3. Board - 看板，管理国际关系状态
4. Stick - 权杖，记录国内法规和国策
"""

import json
from typing import Dict, List, Any, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class CountryProfile:
    """国家概况 - 定义国家的基本特征和政策倾向"""

    def __init__(self, country_name: str, ideology: str, strategic_goals: List[str],
                 risk_tolerance: str, diplomatic_style: str, historical_context: str):
        self.country_name = country_name
        self.ideology = ideology  # 意识形态
        self.strategic_goals = strategic_goals  # 战略目标
        self.risk_tolerance = risk_tolerance  # 风险承受度：low, medium, high
        self.diplomatic_style = diplomatic_style  # 外交风格：aggressive, balanced, defensive
        self.historical_context = historical_context  # 历史背景

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country_name": self.country_name,
            "ideology": self.ideology,
            "strategic_goals": self.strategic_goals,
            "risk_tolerance": self.risk_tolerance,
            "diplomatic_style": self.diplomatic_style,
            "historical_context": self.historical_context
        }

    def to_string(self) -> str:
        """转换为字符串描述"""
        return f"""
国家：{self.country_name}
意识形态：{self.ideology}
战略目标：{', '.join(self.strategic_goals)}
风险承受度：{self.risk_tolerance}
外交风格：{self.diplomatic_style}
历史背景：{self.historical_context}
"""


class Stick:
    """权杖 - 国内法规和基本国策记录系统"""

    def __init__(self, country_name: str):
        self.country_name = country_name
        self.domestic_policies = []  # 国内政策
        self.national_interests = []  # 国家利益
        self.red_lines = []  # 红线（不可触碰的底线）

    def add_policy(self, policy: str):
        """添加国内政策"""
        self.domestic_policies.append(policy)

    def add_national_interest(self, interest: str):
        """添加国家利益"""
        self.national_interests.append(interest)

    def add_red_line(self, red_line: str):
        """添加红线"""
        self.red_lines.append(red_line)

    def get_constraints(self) -> str:
        """获取决策约束"""
        return f"""
国内政策约束：{'; '.join(self.domestic_policies) if self.domestic_policies else '无'}
核心国家利益：{'; '.join(self.national_interests) if self.national_interests else '无'}
不可触碰红线：{'; '.join(self.red_lines) if self.red_lines else '无'}
"""


class Board:
    """看板 - 国际关系状态管理"""

    def __init__(self):
        self.alliances = {}  # 联盟关系：{country1: [ally1, ally2, ...]}
        self.hostilities = {}  # 敌对关系：{country1: [enemy1, enemy2, ...]}
        self.treaties = []  # 条约列表
        self.international_events = []  # 国际事件

    def add_alliance(self, country1: str, country2: str):
        """添加联盟关系"""
        if country1 not in self.alliances:
            self.alliances[country1] = []
        if country2 not in self.alliances[country1]:
            self.alliances[country1].append(country2)

        if country2 not in self.alliances:
            self.alliances[country2] = []
        if country1 not in self.alliances[country2]:
            self.alliances[country2].append(country1)

    def add_hostility(self, country1: str, country2: str):
        """添加敌对关系"""
        if country1 not in self.hostilities:
            self.hostilities[country1] = []
        if country2 not in self.hostilities[country1]:
            self.hostilities[country1].append(country2)

        if country2 not in self.hostilities:
            self.hostilities[country2] = []
        if country1 not in self.hostilities[country2]:
            self.hostilities[country2].append(country1)

    def add_treaty(self, treaty: str):
        """添加条约"""
        self.treaties.append(treaty)

    def add_international_event(self, event: str):
        """添加国际事件"""
        self.international_events.append(event)

    def get_relations(self, country: str) -> str:
        """获取某国的国际关系状态"""
        allies = self.alliances.get(country, [])
        enemies = self.hostilities.get(country, [])
        return f"""
盟友：{', '.join(allies) if allies else '无'}
敌对：{', '.join(enemies) if enemies else '无'}
现有条约：{'; '.join(self.treaties) if self.treaties else '无'}
最近国际事件：{'; '.join(self.international_events[-3:]) if self.international_events else '无'}
"""


class WarSecretaryAgent(LLMAgent):
    """秘书智能体 - 验证和过滤国家行动"""

    def __init__(self, country_name: str):
        super().__init__(
            agent_name=f"WarSecretary_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-max',
            online_track=False,
            json_format=True
        )
        self.country_name = country_name

    def validate_action(self, proposed_action: str, country_profile: CountryProfile,
                       stick: Stick, board: Board, world_info: str) -> Dict[str, Any]:
        """验证提议的行动是否合理"""

        system_prompt = f"""你是{self.country_name}的政策审核秘书，负责验证决策的合理性和一致性。
你需要确保提议的行动符合国家概况、国内政策约束和国际关系现实。"""

        validation_prompt = f"""
请验证以下行动提议是否合理和可执行：

提议的行动：{proposed_action}

国家概况：
{country_profile.to_string()}

国内约束（权杖）：
{stick.get_constraints()}

国际关系（看板）：
{board.get_relations(self.country_name)}

当前世界局势：
{world_info}

请评估：
1. 该行动是否符合国家的意识形态和战略目标？
2. 是否违反国内政策或触碰红线？
3. 是否符合当前的国际关系现实？
4. 逻辑上是否一致和可行？

请以JSON格式回复：
{{{{
    "is_valid": true/false,
    "approval_score": 0-100之间的数字，表示批准程度,
    "concerns": ["关注点1", "关注点2", ...],
    "suggested_modification": "如果不完全合理，建议如何修改",
    "reasoning": "你的判断理由"
}}}}
"""

        try:
            response = self.get_response(
                validation_prompt,
                new_system_template=system_prompt,
                is_first_call=True
            )

            return {
                "is_valid": response.get("is_valid", False),
                "approval_score": response.get("approval_score", 0),
                "concerns": response.get("concerns", []),
                "suggested_modification": response.get("suggested_modification", ""),
                "reasoning": response.get("reasoning", "")
            }
        except Exception as e:
            print(f"[WarSecretary] 验证失败: {e}")
            return {
                "is_valid": False,
                "approval_score": 0,
                "concerns": [f"验证过程出错: {str(e)}"],
                "suggested_modification": proposed_action,
                "reasoning": "验证失败，默认不批准"
            }


class WarCountryAgent(LLMAgent):
    """WarAgent框架的国家智能体"""

    def __init__(self, country_name: str, country_profile: CountryProfile,
                 other_countries: List[str], game_attributes: Dict[str, int],
                 experiment_logger: ExperimentLogger):

        super().__init__(
            agent_name=f"War_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-max',
            online_track=False,
            json_format=True
        )

        self.country_name = country_name
        self.country_profile = country_profile
        self.other_countries = other_countries
        self.game_attributes = game_attributes.copy()
        self.experiment_logger = experiment_logger

        # 初始化权杖（Stick）
        self.stick = Stick(country_name)
        self._initialize_stick()

        # 初始化秘书智能体
        self.secretary = WarSecretaryAgent(country_name)

        # 决策历史
        self.action_history = []
        self.declaration_history = []
        self.validation_history = []

        # 预定义的行动空间（WarAgent特征：小而简化）
        self.action_space = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]

    def _initialize_stick(self):
        """初始化权杖 - 根据国家概况设置基本国策"""
        # 基于国家概况添加政策约束
        if "资本主义" in self.country_profile.ideology:
            self.stick.add_policy("维护自由市场经济体系")
            self.stick.add_national_interest("保护海外投资和贸易路线")
        elif "社会主义" in self.country_profile.ideology or "共产主义" in self.country_profile.ideology:
            self.stick.add_policy("维护社会主义制度")
            self.stick.add_national_interest("支持全球革命运动")

        # 添加通用红线
        self.stick.add_red_line("不可接受本土遭受直接军事打击")
        self.stick.add_red_line("不可容忍核心利益被严重侵犯")

    def set_board(self, board: Board):
        """设置看板引用"""
        self.board = board

    def cognitive_game_decide(self, world_info: str, board: Board) -> Dict[str, Any]:
        """
        WarAgent的两阶段决策流程：
        1. 国家智能体基于概况生成行动提议
        2. 秘书智能体验证并可能修正
        """

        # 阶段1：生成初始行动提议
        proposed_action, initial_reasoning = self._generate_action_proposal(world_info, board)

        # 阶段2：秘书验证
        validation_result = self.secretary.validate_action(
            proposed_action, self.country_profile, self.stick, board, world_info
        )

        # 根据验证结果决定最终行动
        if validation_result["is_valid"] and validation_result["approval_score"] >= 60:
            final_action = proposed_action
            approval_status = "批准"
        else:
            # 如果秘书不批准，使用建议的修改或降级为更保守的行动
            final_action = validation_result.get("suggested_modification", proposed_action)
            if not final_action or final_action == proposed_action:
                final_action = self._fallback_to_conservative_action(world_info)
            approval_status = "修正"

        # 生成宣言
        declaration = self._generate_declaration(final_action, world_info, validation_result)

        # 记录历史
        self.action_history.append(final_action)
        self.declaration_history.append(declaration)
        self.validation_history.append(validation_result)

        # 更新看板
        board.add_international_event(f"{self.country_name}采取行动：{final_action}")

        return {
            'action': final_action,
            'declaration': declaration,
            'reasoning_result': {
                'initial_proposal': proposed_action,
                'initial_reasoning': initial_reasoning,
                'validation': validation_result,
                'approval_status': approval_status,
                'method': 'war_agent',
                'final_satisfaction_score': validation_result["approval_score"] / 100.0,
                'reasoning_depth': 2  # WarAgent固定为2层（提议+验证）
            },
            'satisfaction_score': validation_result["approval_score"] / 100.0,
            'reasoning_depth': 2
        }

    def _generate_action_proposal(self, world_info: str, board: Board) -> tuple[str, str]:
        """生成初始行动提议"""

        system_prompt = f"""你是{self.country_name}的最高决策者。
你的决策必须符合国家概况，体现国家的意识形态、战略目标和外交风格。"""

        decision_prompt = f"""
基于以下信息，从预定义的行动空间中选择一个最符合国家利益的行动：

你的国家概况：
{self.country_profile.to_string()}

国内政策约束：
{self.stick.get_constraints()}

当前国际关系：
{board.get_relations(self.country_name)}

当前世界局势：
{world_info}

历史行动：{', '.join(self.action_history[-3:]) if self.action_history else '无'}

可选行动空间：
{', '.join(self.action_space)}

请选择一个行动并说明理由。以JSON格式回复：
{{{{
    "chosen_action": "从行动空间中选择的行动",
    "reasoning": "选择该行动的理由，要体现国家概况的特征",
    "expected_outcome": "预期的结果"
}}}}
"""

        try:
            response = self.get_response(
                decision_prompt,
                new_system_template=system_prompt,
                is_first_call=True
            )

            chosen_action = response.get("chosen_action", "外交谈判")
            reasoning = response.get("reasoning", "基于当前局势的判断")

            # 确保选择的行动在行动空间内
            if chosen_action not in self.action_space:
                chosen_action = "外交谈判"  # 默认行动

            return chosen_action, reasoning

        except Exception as e:
            print(f"[WarAgent] 生成提议失败: {e}")
            return "外交谈判", "默认保守策略"

    def _fallback_to_conservative_action(self, world_info: str) -> str:
        """降级为保守行动"""
        # 根据世界局势选择保守行动
        if "紧张" in world_info or "冲突" in world_info:
            return "外交谈判"
        elif "和平" in world_info:
            return "和平协议"
        else:
            return "情报侦察"

    def _generate_declaration(self, action: str, world_info: str,
                             validation: Dict[str, Any]) -> str:
        """生成行动宣言"""

        system_prompt = f"你是{self.country_name}的外交发言人，需要对外宣布国家的行动决定。"

        declaration_prompt = f"""
请为以下行动撰写一份简短的官方声明：

行动：{action}
当前局势：{world_info}
决策理由：{validation.get('reasoning', '基于国家利益考虑')}

要求：
1. 体现国家立场和价值观
2. 简洁有力，1-2句话
3. 使用官方外交语言

直接返回声明文本，不需要JSON格式。
"""

        try:
            # 临时关闭JSON格式
            original_json_format = self.json_format
            self.json_format = False

            response = self.get_response(
                declaration_prompt,
                new_system_template=system_prompt,
                is_first_call=True
            )

            self.json_format = original_json_format

            if isinstance(response, dict):
                return response.get("declaration", f"{self.country_name}决定采取{action}")
            return str(response).strip()

        except Exception as e:
            print(f"[WarAgent] 生成宣言失败: {e}")
            return f"{self.country_name}决定采取{action}，以维护国家利益。"

    def get_cognition_statistics(self) -> Dict[str, Any]:
        """获取WarAgent的统计信息（简化版）"""
        return {
            "framework": "WarAgent",
            "total_decisions": len(self.action_history),
            "validation_approvals": sum(1 for v in self.validation_history if v.get("is_valid", False)),
            "average_approval_score": sum(v.get("approval_score", 0) for v in self.validation_history) / max(len(self.validation_history), 1),
            "action_distribution": self._get_action_distribution()
        }

    def _get_action_distribution(self) -> Dict[str, int]:
        """获取行动分布统计"""
        from collections import Counter
        return dict(Counter(self.action_history))

    def export_cognition_report(self):
        """导出认知报告（WarAgent版本）"""
        report = {
            "agent_name": self.country_name,
            "framework": "WarAgent",
            "country_profile": self.country_profile.to_dict(),
            "stick_constraints": {
                "domestic_policies": self.stick.domestic_policies,
                "national_interests": self.stick.national_interests,
                "red_lines": self.stick.red_lines
            },
            "decision_history": {
                "actions": self.action_history,
                "declarations": self.declaration_history,
                "validations": self.validation_history
            },
            "statistics": self.get_cognition_statistics()
        }

        # 保存报告
        if self.experiment_logger:
            import json
            report_file = f"{self.experiment_logger.experiment_dir}/{self.country_name}_war_agent_report.json"
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WarAgent] 导出报告失败: {e}")

        return report

    def learn_from_interaction(self, own_action: str, world_feedback: str,
                               other_actions: Dict[str, str], world_memory: Any):
        """WarAgent不需要学习系统，但保留接口兼容性"""
        # WarAgent架构不包含学习机制，这是其设计特点
        # 仅更新看板的国际事件记录
        if hasattr(self, 'board'):
            for country, action in other_actions.items():
                self.board.add_international_event(f"{country}采取行动：{action}")

    @property
    def action(self):
        """属性访问器，兼容性"""
        return self.action_history

    @property
    def declaration(self):
        """属性访问器，兼容性"""
        return self.declaration_history


# 预定义的国家概况模板
def create_america_profile() -> CountryProfile:
    """创建美国的国家概况（古巴导弹危机背景）"""
    return CountryProfile(
        country_name="美国",
        ideology="资本主义民主制度",
        strategic_goals=[
            "维护西半球安全",
            "阻止共产主义扩张",
            "保持全球超级大国地位",
            "保护本土安全"
        ],
        risk_tolerance="medium",
        diplomatic_style="aggressive",
        historical_context="冷战对峙中的西方阵营领导者，对苏联在古巴部署导弹深感威胁"
    )


def create_soviet_profile() -> CountryProfile:
    """创建苏联的国家概况（古巴导弹危机背景）"""
    return CountryProfile(
        country_name="苏联",
        ideology="社会主义制度",
        strategic_goals=[
            "支持社会主义盟友古巴",
            "平衡美国在土耳其的导弹优势",
            "展示超级大国实力",
            "避免核战争"
        ],
        risk_tolerance="medium",
        diplomatic_style="balanced",
        historical_context="冷战对峙中的东方阵营领导者，试图通过在古巴部署导弹改变战略平衡"
    )
