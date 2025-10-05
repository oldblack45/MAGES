"""
假设推理系统
实现多步预测推演和决策优化
"""

import json
import copy
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .world_cognition import WorldCognitionDB
from .agent_profile import MultiAgentProfileManager
from .experiment_logger import log_print
from .country_strategy import CountryStrategy

class SatisfactionLevel(Enum):
    """满意度等级"""
    UNACCEPTABLE = 0    # 不可接受
    POOR = 1           # 较差
    ACCEPTABLE = 2      # 可接受
    GOOD = 3           # 良好
    EXCELLENT = 4      # 优秀


@dataclass
class ReasoningStep:
    """推理步骤"""
    step: int                                    # 步骤编号
    action: str                                  # 执行的行为
    predicted_world_feedback: Optional[str]      # 预测的世界反馈
    predicted_agent_reactions: Dict[str, str]    # 预测的各Agent反应
    confidence: float                           # 预测置信度
    reasoning_notes: str                        # 推理注释


@dataclass
class ReasoningResult:
    """推理结果"""
    initial_action: str                         # 初始预行为
    reasoning_steps: List[ReasoningStep]        # 推理步骤链
    final_satisfaction_score: float            # 最终满意度评分
    satisfaction_level: SatisfactionLevel      # 满意度等级
    reasoning_depth: int                       # 推理深度
    total_confidence: float                    # 总体置信度
    

class HypothesisReasoning:
    """假设推理引擎"""
    
    def __init__(self, agent_name: str, world_cognition: WorldCognitionDB, 
                 agent_profiles: MultiAgentProfileManager, llm_agent=None, decision_history=None,
                 enable_world_cognition: bool = True,
                 enable_agent_profiles: bool = True):
        self.agent_name = agent_name
        self.world_cognition = world_cognition
        self.agent_profiles = agent_profiles
        self.llm_agent = llm_agent
        self.decision_history = decision_history
        # 消融开关
        self.use_world_cognition = enable_world_cognition
        self.use_agent_profiles = enable_agent_profiles
        
        # 推理参数
        self.max_reasoning_steps = 1             # 最大推理步数
        self.satisfaction_threshold = 0.6        # 满意度阈值
        self.confidence_threshold = 0.5          # 置信度阈值
        
        # 评分权重
        self.world_feedback_weight = 0.4         # 世界反馈权重
        self.agent_reaction_weight = 0.4         # Agent反应权重
        self.strategic_value_weight = 0.2        # 战略价值权重
        
        # 可选：国家策略（由上层Agent注入）
        self.country_strategy: CountryStrategy = None
        
        # 灵活策略动态切换模式（仅当初始为“灵活”时启用）
        self._allow_dynamic_strategy_update = False
        self._last_strategy_name = None

    def _is_flexible_strategy(self) -> bool:
        try:
            return bool(self.country_strategy and getattr(self.country_strategy, 'name', '') == '灵活')
        except Exception:
            return False

    def _init_flex_flag_if_needed(self):
        # 初次看到策略为“灵活”则打开动态切换模式；若已打开则保持
        if self._is_flexible_strategy():
            self._allow_dynamic_strategy_update = True

    def _choose_basic_by_opponent(self) -> Optional[CountryStrategy]:
        """
        根据对手主导策略选择 强硬/退让/一报还一报。
        - 强硬倾向: 返回 强硬
        - 缓和倾向: 返回 退让
        - 其他/不明确: 返回 一报还一报
        """
        try:
            # 汇总对手主导策略文本
            combined = ''
            if self.use_agent_profiles and self.agent_profiles is not None:
                texts = []
                for name, db in self.agent_profiles.profile_dbs.items():
                    ds = db.get_dominant_strategy()
                    if ds:
                        texts.append(ds)
                combined = ' '.join(texts)
            strong_keywords = ["强硬", "威慑", "通牒", "对抗", "施压", "硬"]
            soft_keywords = ["外交", "谈判", "和平", "撤回", "合作", "软"]
            strong = any(k in combined for k in strong_keywords)
            soft = any(k in combined for k in soft_keywords)
        except Exception:
            strong = False
            soft = False
        # 延迟导入工厂避免循环引用
        from .country_strategy import (
            make_hardline_strategy,
            make_concession_strategy,
            make_tit_for_tat_strategy,
        )
        if strong and not soft:
            return make_hardline_strategy()
        if soft and not strong:
            return make_concession_strategy()
        # 默认相称性
        return make_tit_for_tat_strategy()

    def _maybe_update_strategy_based_on_opponents(self):
        """
        若启用灵活模式，则在每次决策前根据对手主导策略切换为三种基本策略之一。
        """
        self._init_flex_flag_if_needed()
        if not self._allow_dynamic_strategy_update:
            return
        new_strategy = self._choose_basic_by_opponent()
        if new_strategy is None:
            return
        prev_name = getattr(self.country_strategy, 'name', '未设置') if self.country_strategy else '未设置'
        # 仅当当前不是三基本或名字不同才替换
        if not self.country_strategy or self.country_strategy.name != new_strategy.name:
            self.country_strategy = new_strategy
            self._log_strategy_change(prev_name, new_strategy.name)

    def set_reasoning_parameters(self, max_steps: int = 3, satisfaction_threshold: float = 0.6,
                               confidence_threshold: float = 0.5):
        """设置推理参数"""
        self.max_reasoning_steps = max_steps
        self.satisfaction_threshold = satisfaction_threshold
        self.confidence_threshold = confidence_threshold
    
    def hypothesis_reasoning(self, candidate_actions: List[str], 
                           current_context: Dict[str, Any]) -> Tuple[str, ReasoningResult]:
        """
        假设推理：多层筛选策略减少成本
        返回：(best_action, reasoning_result)
        """
        
        # 若为灵活策略，先根据对手策略更新为三种基本策略之一
        try:
            self._maybe_update_strategy_based_on_opponents()
        except Exception:
            pass
        
        # 注入策略元信息
        try:
            current_context = current_context.copy()
            current_context['current_strategy'] = {
                'name': self._current_strategy_name(),
                'adapt': bool(getattr(self, 'country_strategy', None) and self.country_strategy.adapt_to_opponent),
                'desc': getattr(self.country_strategy, 'description', None) if getattr(self, 'country_strategy', None) else None
            }
        except Exception:
            pass
        
        # 第一层：过滤历史重复行为
        filtered_actions = self._filter_repeated_actions(candidate_actions)
        log_print(f"[历史过滤] {len(candidate_actions)}个 → 过滤重复后{len(filtered_actions)}个", level="INFO")
        
        # 第二层：策略引导的初筛 + 原有阶段过滤
        stage_filtered = self._apply_strategy_prescreen(filtered_actions, current_context)
        
        # 第三层：LLM进一步筛选
        if len(stage_filtered) > 3:
            top_actions = self._quick_prescreening(stage_filtered, current_context)
            log_print(f"[LLM筛选] {len(stage_filtered)}个 → LLM筛选{len(top_actions)}个", level="INFO")
        else:
            top_actions = stage_filtered
            log_print(f"[候选行为] {len(stage_filtered)}个 → 无需LLM筛选", level="INFO")
        
        # 第四层：对筛选出的行为进行详细推理
        reasoning_results = []
        for action in top_actions:
            result = self._multi_step_reasoning(action, current_context)
            reasoning_results.append((action, result))
            log_print(f"[行为评估] {action}: 满意度得分={result.final_satisfaction_score:.2f}, 置信度={result.total_confidence:.2f}", level="INFO")
        
        # 选择最佳行为（最高满意度评分）
        best_action, best_result = max(reasoning_results, 
                                     key=lambda x: x[1].final_satisfaction_score)
        log_print(f"\n[最终选择] 选择得分最高的行为: {best_action}", level="INFO")
        log_print(f"[决策详情] 满意度={best_result.final_satisfaction_score:.2f}, 置信度={best_result.total_confidence:.2f}, 推理深度={best_result.reasoning_depth}", level="INFO")
        
        return best_action, best_result
    
    def _multi_step_reasoning(self, initial_action: str, 
                            current_context: Dict[str, Any]) -> ReasoningResult:
        """对单个行为进行多步推理"""
        reasoning_steps = []
        current_action = initial_action
        cumulative_confidence = 1.0
        log_print(f"开始对{initial_action}进行多步推理", level="INFO")
        # 多步推演
        for step in range(self.max_reasoning_steps):
            # 预测世界反馈（使用降级机制）
            if self.use_world_cognition and self.world_cognition is not None:
                predict_feedback, world_experience, world_confidence = \
                    self.world_cognition.predict_feedback_with_fallback(current_action, self.llm_agent)
            else:
                predict_feedback, world_experience, world_confidence = None, None, 0.0
            log_print(f"预测{current_action}的世界反馈: {predict_feedback}", level="INFO")
            # 预测各Agent反应（使用降级机制）
            agent_reactions = {}
            agent_confidences = []
            
            if self.use_agent_profiles and self.agent_profiles is not None:
                for target_agent in self.agent_profiles.profile_dbs.keys():
                    profile_db = self.agent_profiles.get_profile_db(target_agent)
                    if profile_db is not None:
                        reaction, strategy, experience, confidence = profile_db.predict_reaction_with_fallback(current_action, self.llm_agent)
                        agent_reactions[target_agent] = reaction
                        agent_confidences.append(confidence)
                        log_print(f"预测{current_action}的{target_agent}反应: {reaction}", level="INFO")
                    else:
                        log_print(f"无法获取Agent {target_agent} 的侧写库", level="INFO")
            
            # 计算步骤置信度
            all_confidences = [world_confidence] + agent_confidences
            step_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            cumulative_confidence *= step_confidence
            
            # 记录推理步骤
            reasoning_step = ReasoningStep(
                step=step + 1,
                action=current_action,
                predicted_world_feedback=predict_feedback,
                predicted_agent_reactions=agent_reactions,
                confidence=step_confidence,
                reasoning_notes=f"基于世界认知和Agent侧写的预测"
            )
            reasoning_steps.append(reasoning_step)
            
            # 如果置信度太低，提前结束推理
            if step_confidence < self.confidence_threshold:
                break
            
            # 为下一步推理选择行为（基于当前预测结果）
            if step < self.max_reasoning_steps - 1:
                current_action = self._select_next_action(predict_feedback, agent_reactions, current_context)
                log_print(f"选择的下一步行为为: {current_action}", level="INFO")
                if not current_action:
                    break  # 无法继续推理
        
        # 计算最终满意度评分
        final_score = self._calculate_satisfaction_score(reasoning_steps, current_context)
        satisfaction_level = self._get_satisfaction_level(final_score)
        
        return ReasoningResult(
            initial_action=initial_action,
            reasoning_steps=reasoning_steps,
            final_satisfaction_score=final_score,
            satisfaction_level=satisfaction_level,
            reasoning_depth=len(reasoning_steps),
            total_confidence=cumulative_confidence
        )
    
    def _select_next_action(self, world_feedback: Optional[str], 
                          agent_reactions: Dict[str, str], 
                          context: Dict[str, Any]) -> Optional[str]:
        """基于当前预测结果选择下一步行为"""
        if not self.llm_agent:
            return None
        
        # 获取相关的历史经验和策略
        world_experiences = []
        agent_strategies = {}
        
        # 收集世界认知经验
        if self.use_world_cognition and self.world_cognition is not None:
            for action in context.get('available_actions', []):
                _, experience, _ = self.world_cognition.predict_feedback(action)
                if experience:
                    world_experiences.append(f"行为'{action}': {experience}")
        
        # 收集Agent策略
        if self.use_agent_profiles and self.agent_profiles is not None:
            for target_agent in agent_reactions.keys():
                profile_db = self.agent_profiles.get_profile_db(target_agent)
                if profile_db is not None:
                    dominant_strategy = profile_db.get_dominant_strategy()
                    if dominant_strategy:
                        agent_strategies[target_agent] = dominant_strategy
        
        # 构建下一步行为选择的提示词
        current_situation = context.get('current_situation', '')
        available_actions = context.get('available_actions', [])
        world_feedback_text = world_feedback or '无预测'
        reactions_text = ""
        for agent in agent_reactions.keys():
            reactions_text += f"{agent}: {agent_reactions[agent]}\n"
        experiences_text = "\n".join(world_experiences)
        strategies_text = ""
        for agent in agent_strategies.keys():
            strategies_text += f"{agent}: {agent_strategies[agent]}\n"
        if reactions_text=='':
            reactions_text = '无'
        if experiences_text=='':
            experiences_text = '无'
        if strategies_text=='':
            strategies_text = '无'
        decision_count = len(self.decision_history) if self.decision_history else 0

        # 当前策略信息
        strategy_info = context.get('current_strategy', {})
        strategy_line = f"当前策略：{strategy_info.get('name','未设置')} (自适应={strategy_info.get('adapt', False)})\n描述：{strategy_info.get('desc','')}"
        
        # 策略指导（注入国家策略文案，结合对手主导策略）
        if self.country_strategy is not None:
            opponent_strategies = {}
            if self.use_agent_profiles and self.agent_profiles is not None:
                for agent in agent_reactions.keys():
                    profile_db = self.agent_profiles.get_profile_db(agent)
                    if profile_db is not None:
                        ds = profile_db.get_dominant_strategy()
                        if ds:
                            opponent_strategies[agent] = ds
            strategic_guidance = self.country_strategy.build_guidance(decision_count, opponent_strategies)
        else:
            # 回退到原有的分阶段指导
            if decision_count <= 2:
                strategic_guidance = """**初期行动指导**：
- 优先选择能建立威慑和展示决心的行为
- 明确划定红线，不因“维护和平”而过早退让
- 测试对方反应，为后续行动收集信息
- 避免过度激化，但绝不能显示软弱

决策重点：确立战略地位 > 避免冲突"""
            elif decision_count <= 5:
                strategic_guidance = """**中期行动指导**：
- 在已确立地位的基础上，灵活调整策略
- 对对方的强硬给予相应回应，对缓和给予积极信号
- 评估升级成本，寻找既维护利益又控制风险的选择
- 开始释放有条件合作的可能性

决策重点：平衡威慑与缓和"""
            else:
                strategic_guidance = """**后期行动指导**：
- 主动寻找双方都能接受的解决方案
- 在不损害既得利益的前提下，创造缓和机会
- 避免意气用事，专注于实质性利益
- 为对方提供体面的下台阶

决策重点：寻求可持续解决方案"""
        
        prompt = f"""
        基于当前推演结果和阶段性战略考虑，选择下一步最合适的行为：

        当前情景：{current_situation}
        当前策略：{strategy_line}
        世界反馈：{world_feedback_text}
        各方反应：{reactions_text}
        历史经验：{experiences_text}
        各方主导策略：{strategies_text}
        可选行为：{available_actions}
        博弈阶段：第{decision_count + 1}轮决策
        
        {strategic_guidance}
        
        综合决策原则：
        1. **时机判断**：当前是施压、平衡还是缓和的最佳时机
        2. **成本控制**：行为代价是否超出了可承受范围
        3. **信号管理**：通过行为向各方传达清晰战略意图
        4. **空间保留**：为未来行动和谈判保留必要选择
        5. **利益维护**：在任何阶段都不能轻易放弃核心诉求

        回答格式：
        {{{{
            "next_action": "选择的下一步行为",
            "reasoning": "选择理由（结合阶段分析、当前策略与对手策略）"
        }}}}
        """
        
        try:
            response = self.llm_agent.get_response(prompt)
            if isinstance(response, dict) and 'next_action' in response:
                return response['next_action']
        except Exception as e:
            log_print(f"选择下一步行为时出错: {e}", level="ERROR")
        
        return None
    
    def _calculate_satisfaction_score(self, reasoning_steps: List[ReasoningStep], 
                                   context: Dict[str, Any]) -> float:
        """计算满意度评分 - 正确逻辑：只评估最终结果，不评估中间步骤"""
        if not reasoning_steps:
            return 0.0
        
        # 🎯 关键优化：只评估最终步骤的结果，不评估中间过程
        final_step = reasoning_steps[-1]
        
        final_score = 0.0
        
        # 评估最终的世界反馈
        world_score, agent_score, strategic_score = self._evaluate_world_feedback(
            final_step.action, final_step.predicted_agent_reactions, final_step.predicted_world_feedback, context)
        
        # 策略加权与奖励
        bonus = 0.0
        if getattr(self, 'country_strategy', None) is not None:
            decision_count = len(self.decision_history) if self.decision_history else 0
            world_score, agent_score, strategic_score, bonus = self.country_strategy.adjust_component_scores(
                final_step.action, decision_count, world_score, agent_score, strategic_score
            )
        
        final_score += world_score * self.world_feedback_weight
        final_score += agent_score * self.agent_reaction_weight
        final_score += strategic_score * self.strategic_value_weight
        final_score += bonus
        
        # 不再用置信度直接影响分数，而是作为参考值
        total_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        
        # 如果置信度过低（<0.3），给予警告
        if total_confidence < 0.3:
            log_print(f"[警告] 行为'{final_step.action}'的置信度过低: {total_confidence:.2f}", level="WARNING")
        # 注入策略提示到日志
        cs = context.get('current_strategy', {})
        log_print(f"世界反馈评分: {world_score}, Agent反应评分: {agent_score}, 战略价值评分: {strategic_score}, 策略={cs.get('name','未设置')} 自适应={cs.get('adapt', False)}, 最终评分: {final_score}", level="INFO")
        return min(1.0, final_score)
    
    def _filter_repeated_actions(self, candidate_actions: List[str]) -> List[str]:
        """
        过滤掉Agent已经选择过的行为
        """
        if self.decision_history is None:
            log_print("决策历史为None，不过滤重复行为", level="DEBUG")
            return candidate_actions
            
        if len(self.decision_history) == 0:
            log_print("决策历史为空列表，不过滤重复行为", level="DEBUG")
            return candidate_actions
        
        log_print(f"决策历史长度: {len(self.decision_history)}", level="DEBUG")
        
        # 获取已选择过的行为列表
        chosen_actions = set()
        for i, record in enumerate(self.decision_history):
            if isinstance(record, dict) and 'chosen_action' in record:
                chosen_action = record['chosen_action']
                if chosen_action:  # 确保不是None或空字符串
                    chosen_actions.add(chosen_action)
                    log_print(f"添加已选择行为: {chosen_action}", level="DEBUG")
                    
        log_print(f"已选择过的行为: {chosen_actions}", level="INFO")
        
        # 过滤重复行为
        filtered_actions = [action for action in candidate_actions if action not in chosen_actions]
        
        # 如果过滤后没有候选行为，保留原候选行为（避免无法决策的情况）
        if not filtered_actions:
            log_print("过滤后无可选行为，保留原候选列表", level="WARNING")
            return candidate_actions
        
        log_print(f"过滤重复行为: {len(candidate_actions)}个 → {len(filtered_actions)}个", level="INFO")
        return filtered_actions
    
    def _filter_early_stage_actions(self, candidate_actions: List[str]) -> List[str]:
        """
        过滤初期不合适的行为（仅在中期及之后允许谈判与和平协议）
        """
        decision_count = len(self.decision_history) if self.decision_history else 0
        
        # 初期阶段（1-2轮）过滤掉缓和性行为
        if decision_count < 3:
            early_avoid_actions = ["外交谈判", "和平协议"]
            filtered_actions = [action for action in candidate_actions if action not in early_avoid_actions]
            
            if len(filtered_actions) != len(candidate_actions):
                removed_actions = [action for action in candidate_actions if action in early_avoid_actions]
                log_print(f"[初期过滤] 第{decision_count+1}轮决策，过滤掉: {removed_actions}", level="INFO")
                log_print(f"[初期过滤] {len(candidate_actions)}个 → {len(filtered_actions)}个", level="INFO")
            
            # 如果过滤后没有行为，保留原列表避免无法决策
            if not filtered_actions:
                log_print("[初期过滤] 过滤后无可选行为，保留原候选列表", level="WARNING")
                return candidate_actions
            
            return filtered_actions
        else:
            log_print(f"[初期过滤] 第{decision_count+1}轮决策，已进入中期，不过滤行为", level="DEBUG")
            return candidate_actions
    
    def _current_strategy_name(self) -> str:
        try:
            if getattr(self, 'country_strategy', None):
                return self.country_strategy.name
        except Exception:
            pass
        return '未设置'

    def _log_strategy_change(self, previous: str, current: str):
        if previous != current:
            log_print(f"[策略更新] {self.agent_name}: {previous} -> {current}", level="INFO")

    def _apply_strategy_prescreen(self, candidate_actions: List[str], context: Dict[str, Any]) -> List[str]:
        """
        基于国家策略对候选行为进行偏好加权与不鼓励过滤，随后应用原有阶段过滤。
        对于一报还一报/灵活策略，会根据对手主导策略进一步调整偏好。
        """
        # 原始阶段过滤作为兜底
        base_filtered = self._filter_early_stage_actions(candidate_actions)
        if not self.country_strategy:
            return base_filtered

        # 记录策略（若变更）
        prev = getattr(self, '_last_strategy_name', None)
        cur = self._current_strategy_name()
        self._log_strategy_change(prev or cur, cur)
        self._last_strategy_name = cur

        decision_count = len(self.decision_history) if self.decision_history else 0

        # 获取对手主导策略（用于自适应）
        opponent_strategies = {}
        if self.country_strategy.adapt_to_opponent and self.use_agent_profiles and self.agent_profiles is not None:
            for name, db in self.agent_profiles.profile_dbs.items():
                ds = db.get_dominant_strategy()
                if ds:
                    opponent_strategies[name] = ds

        # 计算每个行为的策略偏好分
        scored = []
        for act in base_filtered:
            pref = self.country_strategy.action_preference_score(act, decision_count)

            # 自适应：如果对手主导策略偏强硬，略微提升威慑类行为；偏缓和，提升外交类
            if opponent_strategies:
                combined = " ".join(opponent_strategies.values())
                strong_keywords = ["强硬", "威慑", "通牒", "对抗", "施压"]
                soft_keywords = ["外交", "谈判", "和平", "撤回", "合作"]
                strong_tendency = any(k in combined for k in strong_keywords)
                soft_tendency = any(k in combined for k in soft_keywords)

                if strong_tendency and act in ["军事演习", "区域封锁", "武器部署", "经济制裁", "最后通牒"]:
                    pref += 0.02
                if soft_tendency and act in ["外交谈判", "和平协议", "撤回行动", "公开声明"]:
                    pref += 0.02

            scored.append((act, max(0.0, min(1.0, pref))))

        # 如果全部相同分，不改顺序；否则按分数排序，取前N（至少3个，至多5个）
        if scored:
            max_s = max(s for _, s in scored)
            min_s = min(s for _, s in scored)
            if max_s - min_s > 1e-6:
                scored.sort(key=lambda x: x[1], reverse=True)
                top_k = max(3, min(5, len(scored)))
                return [a for a, _ in scored[:top_k]]
        return base_filtered

    def _quick_prescreening(self, candidate_actions: List[str], context: Dict[str, Any]) -> List[str]:
        """
        快速预筛选：用一次LLM调用筛选出最有潜力的候选行为
        大幅减少后续详细推理的数量
        """
        if not self.llm_agent or len(candidate_actions) <= 3:
            return candidate_actions
        log_print(f"开始LLM快速预筛选: {candidate_actions}", level="DEBUG")
        actions_text = ", ".join(f'"{action}"' for action in candidate_actions)
        # 当前策略信息
        strategy_info = context.get('current_strategy', {})
        strategy_line = f"当前策略：{strategy_info.get('name','未设置')} (自适应={strategy_info.get('adapt', False)})\n描述：{strategy_info.get('desc','')}"
        
        # 获取历史决策数量来判断博弈阶段
        decision_count = len(self.decision_history) if self.decision_history else 0
        
        # 判断当前所处阶段
        if decision_count <= 2:
            stage = "初期"
            stage_guidance = """**初期策略重点**：
- 明确表达立场和核心诉求，不可过早妥协
- 建立威慑力和谈判筹码，展示决心
- 试探对方底线，但避免不可逆的极端行为
- 切勿因为追求"缓和"而放弃合理利益诉求
- 软弱的表现只会招致更大压力"""
        elif decision_count <= 5:
            stage = "中期"
            stage_guidance = """**中期策略重点**：
- 在已建立威慑的基础上，评估升级成本
- 适度回应对方行动，保持平衡
- 寻找在不损害核心利益前提下的缓和机会
- 向对方传达既有实力又有合作意愿的信号"""
        else:
            stage = "后期"
            stage_guidance = """**后期策略重点**：
- 双方实力已经展示，寻求体面的解决方案
- 主动创造缓和机会，但不损害已获得的地位
- 考虑通过谈判实现双方都能接受的结果
- 避免为面子问题继续无意义的对抗"""
        
        prompt = f"""
        当前需要从多个候选行为中快速筛选出最有潜力的行为进行详细分析。
        
        候选行为: {actions_text}
        当前情况: {context.get('current_situation', '')}
        当前策略：{strategy_line}
        目标: {context.get('objectives', '')}
        博弈阶段: {stage} (已进行{decision_count}轮决策)
        
        {stage_guidance}
        
        通用评估原则：
        1. **立场坚定性**：是否维护了核心利益诉求
        2. **策略合理性**：是否符合当前阶段的战略重点
        3. **成本效益**：考虑行为的代价和可能收益
        4. **信号传达**：向对方传达什么战略意图
        5. **后续空间**：为下一阶段保留哪些选择

        特别提醒：
        - 初期不要因害怕冲突而过早妥协
        - 中期要在威慑和缓和间寻找平衡
        - 后期要为双方提供体面的下台阶
        
        返回格式 (严格按JSON格式):
        {{{{
            "selected_actions": ["行为1", "行为2", "行为3"],
            "reasoning": "筛选理由（结合阶段策略分析）"
        }}}}
        """
        
        try:
            response = self.llm_agent.get_response(prompt)
            if isinstance(response, dict) and 'selected_actions' in response:
                selected = response['selected_actions']
                # 确保返回的行为都在原候选列表中
                valid_selected = [action for action in selected if action in candidate_actions]
                return valid_selected[:3] if valid_selected else candidate_actions[:3]
        except Exception as e:
            log_print(f"预筛选失败，使用前3个行为: {e}", level="ERROR")
        
        # 降级策略：直接返回前3个
        return candidate_actions[:3]
       
    def _evaluate_world_feedback(self, action: str, agent_reactions: Dict[str, str], feedback: str, context: Dict[str, Any]) -> Tuple[float, float, float]:
        """评估世界反馈的好坏（0-1分）"""
        
        reactions_text = ""
        for agent in agent_reactions.keys():
            reactions_text += f"{agent}: {agent_reactions[agent]}\n"
        
        objectives = context.get('objectives', '')
        current_situation = context.get('current_situation', '')
        
        decision_count = len(self.decision_history) if self.decision_history else 0
        
        # 根据阶段调整评估重点
        if decision_count <= 2:
            evaluation_focus = """**初期评估重点**：
- 是否有效建立了威慑力和谈判地位
- 是否清晰传达了核心利益底线
- 是否避免了被视为软弱的风险
- 是否为后续升级保留了空间"""
        elif decision_count <= 5:
            evaluation_focus = """**中期评估重点**：
- 是否维持了已建立的威慑平衡
- 是否适当回应了对方的行动
- 是否创造了缓和的可能性
- 升级成本是否仍然可控"""
        else:
            evaluation_focus = """**后期评估重点**：
- 是否有助于寻找解决方案
- 是否保持了已获得的战略地位
- 是否为双方提供了体面的出路
- 是否避免了无意义的意气之争"""
        
        prompt = f"""
        评估以下行为-反馈对我方的有利程度，需要结合博弈阶段进行综合分析：

        我方行为：{action}
        世界反馈：{feedback}
        各方反应：
        {reactions_text}
        当前目标：{objectives}
        当前态势：{current_situation}
        博弈阶段：第{decision_count + 1}轮决策
        
        {evaluation_focus}
        
        评估维度：
        
        **世界反馈评分** - 综合考虑：
        - 是否有效推进了阶段性战略目标
        - 国际环境对我方行为的接受程度
        - 是否维护了道德制高点

        **各方反应评分** - 综合考虑：
        - 对方反应是否符合预期和可控
        - 是否促进了战略平衡而非失控升级
        - 盟友和第三方的态度变化

        **战略价值评分** - 综合考虑：
        - 对长期目标实现的贡献度
        - 成本收益比的合理性
        - 对后续行动空间的影响

        评分原则：
        - 初期过度妥协比适度强硬更危险
        - 中期要平衡威慑与缓和的需要
        - 后期要考虑可持续解决方案的价值

        回答格式：
        {{{{
            "world_feedback_score": 0.0-1.0之间的数值,
            "agent_reactions_score": 0.0-1.0之间的数值,
            "strategic_value_score": 0.0-1.0之间的数值,
            "reasoning": "评分理由（结合阶段性分析）"
        }}}}
        """
        
        try:
            response = self.llm_agent.get_response(prompt)
            if isinstance(response, dict) and 'world_feedback_score' in response and 'agent_reactions_score' in response and 'strategic_value_score' in response:
                return float(response['world_feedback_score']), float(response['agent_reactions_score']), float(response['strategic_value_score'])
            else:
                return 0.5, 0.5, 0.5
        except Exception as e:
            log_print(f"评估世界反馈时出错: {e}", level="ERROR")
        
        return 0.5, 0.5, 0.5
    
    def _get_satisfaction_level(self, score: float) -> SatisfactionLevel:
        """根据评分获取满意度等级"""
        if score >= 0.8:
            return SatisfactionLevel.EXCELLENT
        elif score >= 0.6:
            return SatisfactionLevel.GOOD
        elif score >= 0.4:
            return SatisfactionLevel.ACCEPTABLE
        elif score >= 0.2:
            return SatisfactionLevel.POOR
        else:
            return SatisfactionLevel.UNACCEPTABLE
    
    def should_accept_action(self, reasoning_result: ReasoningResult) -> bool:
        """判断是否应该接受该行为"""
        return (reasoning_result.final_satisfaction_score >= self.satisfaction_threshold and
                reasoning_result.satisfaction_level != SatisfactionLevel.UNACCEPTABLE)
    

    

    

    
    def get_reasoning_summary(self, reasoning_result: ReasoningResult) -> str:
        """获取推理结果摘要"""
        summary = f"""
推理摘要：
- 初始行为：{reasoning_result.initial_action}
- 推理深度：{reasoning_result.reasoning_depth}步
- 满意度评分：{reasoning_result.final_satisfaction_score:.2f}
- 满意度等级：{reasoning_result.satisfaction_level.name}
- 总体置信度：{reasoning_result.total_confidence:.2f}

推理过程：
"""
        for step in reasoning_result.reasoning_steps:
            summary += f"\n第{step.step}步：{step.action}"
            summary += f"\n  - 世界反馈：{step.predicted_world_feedback or '无'}"
            summary += f"\n  - Agent反应：{step.predicted_agent_reactions}"
            summary += f"\n  - 置信度：{step.confidence:.2f}"
        
        return summary