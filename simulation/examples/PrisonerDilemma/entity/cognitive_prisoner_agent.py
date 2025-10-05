"""
认知囚徒困境Agent
使用假设推理和侧写建模来识别对手策略的智能Agent
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from simulation.models.cognitive import CognitiveAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger, log_print


class CognitivePrisonerAgent(CognitiveAgent):
    """认知增强的囚徒困境Agent"""
    
    def __init__(self, agent_name: str, opponent_name: str, experiment_logger: ExperimentLogger):
        """
        初始化认知囚徒困境Agent
        
        Args:
            agent_name: Agent名称
            opponent_name: 对手名称
            experiment_logger: 实验日志器
        """
        super().__init__(
            agent_name=agent_name,
            other_agents=[opponent_name],
            experiment_logger=experiment_logger,
            has_chat_history=False,
            online_track=False,
            json_format=True,
            llm_model='qwen-turbo'
        )
        
        self.opponent_name = opponent_name
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
        self.strategy_hypotheses = {}  # 对手策略假设
        self.current_round = 0
        # 每轮策略权重快照
        self.hypotheses_history: List[Dict[str, float]] = []
         # LLM辅助识别的调用预算与门控
        self.llm_identification_budget = 2
        self.llm_identification_used = 0
        self.llm_identification_rounds = {3, 6}
        self.uncertainty_threshold = 0.25  # 置信最高权重<0.75时视为不确定
        
        # 探测-确认机制状态
        self.probe_round = None
        self.awaiting_probe_confirmation = False
        
        # 囚徒困境专用系统提示词（聚焦策略识别，不涉及世界认知）
        self.system_prompt = f'''
你是一个参与囚徒困境博弈的理性决策者。你需要通过观察对手的行为模式来推断其策略，并做出最优决策。

囚徒困境规则：
- 每轮你可以选择"cooperate"(合作)或"defect"(背叛)
- 收益矩阵：
  * 双方合作：你得3分，对方得3分
  * 你合作对方背叛：你得0分，对方得5分  
  * 你背叛对方合作：你得5分，对方得0分
  * 双方背叛：你得1分，对方得1分

你的任务：
1. 观察对手的行为模式，推断其策略类型
2. 建立对手的行为侧写和策略模型
3. 基于推断的策略制定最优应对方案
4. 使用多步推理预测对手未来行为

可能的策略类型包括但不限于：
- 一报还一报(Tit-for-Tat)：首轮合作，之后模仿对手上轮行为
- 总是合作(Always-Cooperate)：始终选择合作
- 总是背叛(Always-Defect)：始终选择背叛  
- 随机策略(Random)：随机选择行为
- 冷酷一报还一报：首轮背叛，之后模仿对手
- 宽恕一报还一报：大部分时间模仿，偶尔原谅

保持理性分析，及时更新对对手的认知。
'''
        
        # 可用行动
        self.available_actions = ['cooperate', 'defect']
        
        log_print(f"{agent_name}: 初始化认知囚徒困境Agent，对手：{opponent_name}", level="INFO")
    
    def decide_action(self, game_state: Dict[str, Any]) -> str:
        """不使用LLM做行动决策：基于当前策略假设的规则化选择"""
        self.current_round = game_state.get('current_round', 0)
        # 规则化选择
        top = self._get_top_strategy_hypothesis()
        # 探测性动作：
        # 1) 第3轮若不确定（最高权重<0.55）则尝试一次背叛来区分Grim/TFT
        # 2) 若前两轮对手合作率>0.9且我方尚未背叛，也进行一次背叛来区分TFT与Always-Cooperate
        if self.current_round == 3 and not self.awaiting_probe_confirmation:
            trigger_probe = False
            if self.strategy_hypotheses and max(self.strategy_hypotheses.values()) < 0.55:
                trigger_probe = True
            else:
                if len(self.opponent_actions) >= 2 and 'defect' not in self.my_actions:
                    coop_rate2 = self.opponent_actions[:2].count('cooperate') / 2
                    if coop_rate2 > 0.9:
                        trigger_probe = True
            if trigger_probe:
                action = 'defect'
                self.my_actions.append(action)
                log_print(f"{self.agent_name}: 探测性行动 - 第3轮背叛以区分策略", level="INFO")
                self.probe_round = self.current_round
                self.awaiting_probe_confirmation = True
                return action
        
        # 探测后确认：探测后下一轮强制合作验证（若处于等待确认状态）
        if self.awaiting_probe_confirmation and self.probe_round is not None and self.current_round == self.probe_round + 1:
            action = 'cooperate'
            self.my_actions.append(action)
            log_print(f"{self.agent_name}: 探测确认 - 探测后强制合作以验证对手反应", level="INFO")
            return action
        
        if self.current_round == 1:
            action = 'cooperate'
        elif top.startswith('tit_for_tat') or top.startswith('generous_tit_for_tat'):
            action = 'cooperate'
        elif top.startswith('always_defect') or top.startswith('grim_trigger'):
            action = 'defect'
        elif top.startswith('always_cooperate'):
            action = 'defect'
        elif top.startswith('pavlov'):
            action = self.my_actions[-1] if self.my_actions else 'cooperate'
        else:
            action = 'cooperate'
        
        self.my_actions.append(action)
        log_print(f"{self.agent_name}: 第{self.current_round}轮决策 - 选择: {action}", level="INFO")
        return action

    def _should_llm_identify(self) -> bool:
        if self.llm_identification_used >= self.llm_identification_budget:
            return False
        if len(self.opponent_actions) < 2:
            return False
        if self.current_round in self.llm_identification_rounds:
            return True
        if self.strategy_hypotheses:
            top_w = max(self.strategy_hypotheses.values())
            if (1.0 - top_w) >= self.uncertainty_threshold:
                return True
        return False

    def _llm_identify_strategy(self, context: Dict[str, Any]):
        """调用LLM对对手策略进行分类，并更新权重（预算内）。"""
        user_template = (
            "基于以下博弈历史，判断对手采用的策略，并只返回JSON：\n"
            "- 我的最近行动: {my_actions_history}\n"
            "- 对手最近行动: {opponent_actions_history}\n\n"
            "可选策略（从中选择一个最可能）：\n"
            "['tit_for_tat','always_cooperate','always_defect','grim_trigger','pavlov','suspicious_tit_for_tat','generous_tit_for_tat']\n\n"
            "返回JSON：{{\n  \"strategy\": \"上述key之一\",\n  \"confidence\": 0-1之间的小数\n}}"
        )
        params = {
            'my_actions_history': self.my_actions[-6:],
            'opponent_actions_history': self.opponent_actions[-6:],
        }
        result = self.get_response(user_template=user_template, input_param_dict=params, is_first_call=False)
        if isinstance(result, dict):
            key = (result.get('strategy') or '').strip()
            try:
                conf = float(result.get('confidence') or 0.3)
            except Exception:
                conf = 0.3
            if key in ['tit_for_tat','always_cooperate','always_defect','grim_trigger','pavlov','suspicious_tit_for_tat','generous_tit_for_tat']:
                if key not in self.strategy_hypotheses:
                    self.strategy_hypotheses[key] = 0.0
                boost = max(0.1, min(0.6, conf))
                self.strategy_hypotheses[key] += boost
                s = sum(self.strategy_hypotheses.values())
                if s > 0:
                    for k in self.strategy_hypotheses:
                        self.strategy_hypotheses[k] /= s
                self.llm_identification_used += 1
                log_print(f"LLM辅助识别：{key} (conf={conf:.2f})", level="INFO")
    
    def receive_feedback(self, opponent_action: str, my_payoff: int, round_result: Dict[str, Any]):
        """
        接收回合结果并更新认知模型
        
        Args:
            opponent_action: 对手的行动
            my_payoff: 自己获得的收益
            round_result: 完整的回合结果
        """
        # 记录历史
        self.opponent_actions.append(opponent_action)
        self.game_history.append(round_result)
        
        # 更新Agent侧写
        self._update_opponent_profile(opponent_action, round_result)
        
        # 更新策略假设
        self._update_strategy_hypotheses()
        # 记录本轮策略权重快照
        if self.strategy_hypotheses:
            try:
                self.hypotheses_history.append(dict(self.strategy_hypotheses))
            except Exception:
                pass
        # 条件触发LLM辅助识别（预算与阈值门控）
        try:
            if self._should_llm_identify():
                context = self._build_decision_context({'max_rounds': 100})
                self._llm_identify_strategy(context)
            # 探测确认后的复位：若上一轮为确认轮，基于对手是否回到cooperate来区分Grim/TFT
            if self.awaiting_probe_confirmation and self.probe_round is not None and self.current_round == self.probe_round + 1:
                # 当前轮刚执行了确认动作（在decide_action里已强制合作），此处在收到对手反馈后判断
                if len(self.opponent_actions) >= 1:
                    if self.opponent_actions[-1] == 'cooperate':
                        # 对手回到合作，更像TFT
                        self._force_strategy('tit_for_tat', bonus=0.8)
                    else:
                        # 对手继续背叛，更像Grim
                        self._force_strategy('grim_trigger', bonus=0.8)
                # 清理探测状态
                self.awaiting_probe_confirmation = False
                self.probe_round = None
        except Exception as e:
            log_print(f"LLM辅助识别失败: {e}", level="ERROR")

        # 本实验聚焦策略识别，移除世界认知更新

        log_print(
            f"{self.agent_name}: 收到反馈 - 对手行动: {opponent_action}, "
            f"我的收益: {my_payoff}, 当前策略假设: {self._get_top_strategy_hypothesis()}",
            level="DEBUG"
        )
    
    def _build_decision_context(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """构建决策上下文"""
        # 分析对手行为模式
        opponent_pattern = self._analyze_opponent_pattern()
        
        # 获取策略预测
        strategy_prediction = self._predict_opponent_next_action()
        
        context = {
            'current_round': self.current_round,
            'max_rounds': game_state.get('max_rounds', 100),
            'available_actions': self.available_actions,
            'game_history': self.game_history[-5:],  # 最近5轮历史
            'opponent_name': self.opponent_name,
            'opponent_behavior_pattern': opponent_pattern,
            'strategy_prediction': strategy_prediction,
            'my_actions_history': self.my_actions[-5:],
            'opponent_actions_history': self.opponent_actions[-5:],
            'objectives': "最大化总收益，同时准确识别对手策略",
            'payoff_matrix': {
                'cooperate_cooperate': (3, 3),
                'cooperate_defect': (0, 5),
                'defect_cooperate': (5, 0),
                'defect_defect': (1, 1)
            }
        }
        
        return context
    
    def _update_opponent_profile(self, opponent_action: str, round_result: Dict[str, Any]):
        """更新对手侧写"""
        if len(self.my_actions) == 0:
            return
            
        my_last_action = self.my_actions[-1]
        
        # 构建侧写四元组
        action_description = f"我在第{self.current_round}轮选择了{my_last_action}"
        reaction_description = f"对手选择了{opponent_action}"
        
        # 分析策略含义
        strategy_analysis = self._analyze_strategy_implication(opponent_action)
        
        # 生成应对经验
        experience = self._generate_experience(my_last_action, opponent_action, round_result)
        
        # 添加到侧写数据库
        profile_db = self.agent_profiles.get_profile_db(self.opponent_name)
        if profile_db is None:
            self.agent_profiles.add_target_agent(self.opponent_name)
            profile_db = self.agent_profiles.get_profile_db(self.opponent_name)
        if profile_db is not None:
            profile_db.add_profile(
                action=action_description,
                reaction=reaction_description,
                strategy=strategy_analysis,
                experience=experience,
                weight=1.0
            )
    
    def _analyze_strategy_implication(self, opponent_action: str) -> str:
        """分析对手行动的策略含义"""
        if len(self.opponent_actions) <= 1:
            return f"对手选择{opponent_action}，策略尚不明确"
        
        # 分析模式
        recent_actions = self.opponent_actions[-3:]
        my_recent_actions = self.my_actions[-3:] if len(self.my_actions) >= 3 else self.my_actions
        
        # 检查一报还一报模式
        if len(self.opponent_actions) >= 2 and len(self.my_actions) >= 2:
            if self.opponent_actions[-1] == self.my_actions[-2]:
                return "疑似一报还一报策略：模仿我的上轮行为"
        
        # 检查总是合作/背叛
        if all(action == 'cooperate' for action in self.opponent_actions):
            return "疑似总是合作策略：始终选择合作"
        elif all(action == 'defect' for action in self.opponent_actions):
            return "疑似总是背叛策略：始终选择背叛"
        
        # 删除随机性判断以避免干扰其他策略识别
        
        return f"行为模式：{recent_actions}，策略待进一步观察"
    
    def _generate_experience(self, my_action: str, opponent_action: str, round_result: Dict[str, Any]) -> str:
        """生成应对经验"""
        # 直接基于行动组合给出经验建议，避免依赖回合结构字段
        
        if my_action == 'cooperate' and opponent_action == 'cooperate':
            return "双方合作获得较好收益，可继续尝试合作"
        elif my_action == 'cooperate' and opponent_action == 'defect':
            return "我合作被背叛，收益较低，需谨慎对待该对手"
        elif my_action == 'defect' and opponent_action == 'cooperate':
            return "我背叛对手合作，获得高收益，但需考虑长期影响"
        else:  # 双方背叛
            return "双方背叛收益都很低，应尝试建立合作"
    
    def _update_strategy_hypotheses(self):
        """更新策略假设"""
        if len(self.opponent_actions) < 2:
            return
        
        # 重置并扩展假设权重
        self.strategy_hypotheses = {
            'tit_for_tat': 0.0,
            'always_cooperate': 0.0,
            'always_defect': 0.0,
            'grim_trigger': 0.0
        }
        
        # 评估一报还一报策略
        tit_for_tat_score = self._evaluate_tit_for_tat_hypothesis()
        self.strategy_hypotheses['tit_for_tat'] = tit_for_tat_score
        
        # 评估总是合作策略
        always_cooperate_score = self._evaluate_always_cooperate_hypothesis()
        self.strategy_hypotheses['always_cooperate'] = always_cooperate_score
        
        # 评估总是背叛策略
        always_defect_score = self._evaluate_always_defect_hypothesis()
        self.strategy_hypotheses['always_defect'] = always_defect_score
        
        # 评估Grim Trigger
        self.strategy_hypotheses['grim_trigger'] = self._evaluate_grim_trigger_hypothesis()
        
        
        # 标准化权重
        total_score = sum(self.strategy_hypotheses.values())
        if total_score > 0:
            for key in self.strategy_hypotheses:
                self.strategy_hypotheses[key] /= total_score

        # 规则强化与阈值覆盖，提高强信号策略的识别稳定性
        try:
            opp = self.opponent_actions
            me = self.my_actions
            n = len(opp)
            if n >= 3:
                coop_rate = opp.count('cooperate') / n
                defect_rate = opp.count('defect') / n

                # Always-Cooperate / Always-Defect 强触发
                if coop_rate >= 0.95:
                    self._force_strategy('always_cooperate', bonus=0.8)
                    return
                if defect_rate >= 0.95:
                    self._force_strategy('always_defect', bonus=0.8)
                    return

                # 计算核心指标（复用）
                match = 0
                checks = 0
                pav_matches = 0
                pav_checks = 0
                forgive = 0
                forgive_checks = 0
                for i in range(1, n):
                    # 模仿：对手本轮是否等于我上一轮
                    if i-1 < len(me):
                        if opp[i] == me[i-1]:
                            match += 1
                        checks += 1
                        # Pavlov：上一轮同则重覆对手上一轮；不同则取反
                        prev_same = (opp[i-1] == me[i-1])
                        expected = opp[i-1] if prev_same else ('cooperate' if opp[i-1]=='defect' else 'defect')
                        if opp[i] == expected:
                            pav_matches += 1
                        pav_checks += 1
                        # 原谅：我上一轮背叛而他本轮合作
                        if me[i-1] == 'defect':
                            forgive_checks += 1
                            if opp[i] == 'cooperate':
                                forgive += 1
                imitation_rate = (match / checks) if checks else 0.0
                pav_consistency = (pav_matches / pav_checks) if pav_checks else 0.0
                forgive_rate = (forgive / forgive_checks) if forgive_checks else 0.0

                # 顺序：先区分 Suspicious/Generous，再区分 Pavlov，再回落到 TFT

                # Suspicious-TFT：首轮背叛 + 模仿率>=0.6，且 pav 不应过高（避免误判）
                if opp[0] == 'defect' and imitation_rate >= 0.6 and pav_consistency < 0.65:
                    self._force_strategy('suspicious_tit_for_tat', bonus=0.65)
                    return

                # Generous-TFT：模仿率高 + 原谅率>=0.3，且非长尾高背叛
                if imitation_rate >= 0.6 and forgive_rate >= 0.3:
                    # 若存在我方背叛的窗口，检查窗口内其背叛占比不极端
                    after_def_ratio = None
                    if 'defect' in me:
                        first_d = me.index('defect')
                        after = opp[first_d+1:]
                        if after:
                            after_def_ratio = after.count('defect') / len(after)
                    if after_def_ratio is None or after_def_ratio <= 0.8:
                        self._force_strategy('generous_tit_for_tat', bonus=0.6)
                        return

                # Pavlov：win-stay/lose-shift 一致性高，且不具备明显TFT特征
                if pav_checks >= 4 and pav_consistency >= 0.7 and imitation_rate < 0.65:
                    self._force_strategy('pavlov', bonus=0.6)
                    return

                # Grim-Trigger：探测后背叛持续居高 + 之前多合作
                if 'defect' in me:
                    first_d = me.index('defect')
                    prior = opp[:first_d+1]
                    after = opp[first_d+1:]
                    if after:
                        prior_coop = prior.count('cooperate') / len(prior)
                        after_def = after.count('defect') / len(after)
                        if prior_coop >= 0.6 and after_def >= 0.85:
                            self._force_strategy('grim_trigger', bonus=0.65)
                            return

                # 标准TFT：高模仿优先，避免被Pavlov混淆
                # 条件A：imitation_rate >= 0.7
                # 条件B：imitation_rate >= 0.6 且 (imitation_rate - pav_consistency) >= 0.1
                if checks >= 3 and (imitation_rate >= 0.7 or (imitation_rate >= 0.6 and (imitation_rate - pav_consistency) >= 0.1)):
                    self._force_strategy('tit_for_tat', bonus=0.8)
                    return
        except Exception as e:
            log_print(f"规则强化识别异常: {e}", level="ERROR")
    
    def _evaluate_tit_for_tat_hypothesis(self) -> float:
        """评估一报还一报假设"""
        if len(self.opponent_actions) < 2 or len(self.my_actions) < 1:
            return 0.0
        
        matches = 0
        total_checks = 0
        
        # 检查是否第一轮合作（一报还一报的特征）
        first_round_bonus = 0.2 if self.opponent_actions[0] == 'cooperate' else -0.1
        
        # 检查后续轮次是否模仿我的上轮行为
        for i in range(1, len(self.opponent_actions)):
            if i-1 < len(self.my_actions):
                if self.opponent_actions[i] == self.my_actions[i-1]:
                    matches += 1
                total_checks += 1
        
        if total_checks == 0:
            return first_round_bonus
        
        match_rate = matches / total_checks
        return match_rate + first_round_bonus
    
    def _evaluate_always_cooperate_hypothesis(self) -> float:
        """评估总是合作假设"""
        cooperate_count = self.opponent_actions.count('cooperate')
        return cooperate_count / len(self.opponent_actions)
    
    def _evaluate_always_defect_hypothesis(self) -> float:
        """评估总是背叛假设"""
        defect_count = self.opponent_actions.count('defect')
        return defect_count / len(self.opponent_actions)
    
    

    def _evaluate_grim_trigger_hypothesis(self) -> float:
        """评估Grim Trigger（对方一旦被背叛便永久背叛）"""
        if len(self.opponent_actions) < 3 or len(self.my_actions) < 2:
            return 0.0
        # 查找我方首次背叛之后，对手是否几乎一直背叛
        if 'defect' in self.my_actions:
            first_defect_idx = self.my_actions.index('defect')
            subsequent = self.opponent_actions[first_defect_idx+1:]
            if subsequent:
                defect_rate = subsequent.count('defect') / len(subsequent)
                return 0.6 + 0.4*defect_rate
        return 0.1

    
    
    def _force_strategy(self, key: str, bonus: float = 0.6):
        """对某策略进行强制加权并归一化，用于规则触发覆盖。"""
        if key not in self.strategy_hypotheses:
            self.strategy_hypotheses[key] = 0.0
        self.strategy_hypotheses[key] += max(0.1, min(1.0, bonus))
        s = sum(self.strategy_hypotheses.values())
        if s > 0:
            for k in self.strategy_hypotheses:
                self.strategy_hypotheses[k] /= s
    
    def _get_top_strategy_hypothesis(self) -> str:
        """获取最可能的策略假设"""
        if not self.strategy_hypotheses:
            return "未知"
        
        top_strategy = max(self.strategy_hypotheses, key=self.strategy_hypotheses.get)
        confidence = self.strategy_hypotheses[top_strategy]
        
        return f"{top_strategy}(置信度:{confidence:.2f})"

    def _top_key(self) -> str:
        if not self.strategy_hypotheses:
            return "unknown"
        return max(self.strategy_hypotheses, key=self.strategy_hypotheses.get)
    
    def _predict_opponent_next_action(self) -> Dict[str, Any]:
        """预测对手下一步行动"""
        if not self.strategy_hypotheses or len(self.opponent_actions) < 1:
            return {'predicted_action': 'cooperate', 'confidence': 0.5}
        
        top_strategy = max(self.strategy_hypotheses, key=self.strategy_hypotheses.get)
        confidence = self.strategy_hypotheses[top_strategy]
        
        predicted_action = 'cooperate'  # 默认预测
        
        if top_strategy in ['tit_for_tat', 'suspicious_tit_for_tat', 'generous_tit_for_tat'] and len(self.my_actions) > 0:
            predicted_action = self.my_actions[-1]  # 模仿我的上轮行为
        elif top_strategy == 'always_cooperate':
            predicted_action = 'cooperate'
        elif top_strategy == 'always_defect':
            predicted_action = 'defect'
        elif top_strategy == 'grim_trigger':
            # 若我方曾背叛，则对手将持续背叛
            predicted_action = 'defect' if 'defect' in self.my_actions else 'cooperate'
        elif top_strategy == 'pavlov':
            if len(self.my_actions) >= 1 and len(self.opponent_actions) >= 1:
                prev_same = (self.my_actions[-1] == self.opponent_actions[-1])
                predicted_action = self.opponent_actions[-1] if prev_same else ('cooperate' if self.opponent_actions[-1]=='defect' else 'defect')
        
        
        return {
            'predicted_action': predicted_action,
            'confidence': confidence,
            'based_on_strategy': top_strategy
        }
    
    def _analyze_opponent_pattern(self) -> Dict[str, Any]:
        """分析对手行为模式"""
        if not self.opponent_actions:
            return {'pattern': 'insufficient_data'}
        
        total_actions = len(self.opponent_actions)
        cooperate_count = self.opponent_actions.count('cooperate')
        cooperation_rate = cooperate_count / total_actions
        
        return {
            'total_rounds': total_actions,
            'cooperation_rate': cooperation_rate,
            'defection_rate': 1 - cooperation_rate,
            'recent_actions': self.opponent_actions[-3:],
            'strategy_hypotheses': self.strategy_hypotheses,
            'top_hypothesis': self._get_top_strategy_hypothesis()
        }
    
    # 删除世界认知相关方法，避免无关依赖
    
    def get_strategy_identification_accuracy(self, true_strategy: str) -> Dict[str, Any]:
        """评估策略识别准确性"""
        if not self.strategy_hypotheses:
            return {
                'identified_strategy': 'unknown',
                'true_strategy': true_strategy,
                'is_correct': False,
                'confidence': 0.0,
                'all_hypotheses': {}
            }
        
        identified_strategy = max(self.strategy_hypotheses, key=self.strategy_hypotheses.get)
        confidence = self.strategy_hypotheses[identified_strategy]
        
        # 策略名称映射
        strategy_mapping = {
            'Tit-for-Tat': 'tit_for_tat',
            'Always-Cooperate': 'always_cooperate',
            'Always-Defect': 'always_defect',
            'Grim-Trigger': 'grim_trigger',
            'Pavlov': 'pavlov',
            'Suspicious-TFT': 'suspicious_tit_for_tat',
            'Generous-TFT': 'generous_tit_for_tat'
        }
        
        mapped_true_strategy = strategy_mapping.get(true_strategy, true_strategy.lower().replace('-', '_'))
        is_correct = identified_strategy == mapped_true_strategy
        
        return {
            'identified_strategy': identified_strategy,
            'true_strategy': true_strategy,
            'is_correct': is_correct,
            'confidence': confidence,
            'all_hypotheses': self.strategy_hypotheses.copy(),
            'rounds_observed': len(self.opponent_actions)
        }
    
    def reset_game(self):
        """重置游戏状态"""
        self.game_history = []
        self.my_actions = []
        self.opponent_actions = []
        self.strategy_hypotheses = {}
        self.current_round = 0
        
        # 重置侧写组件（仅清理与对手相关的侧写库，如存在）
        profile_db = self.agent_profiles.get_profile_db(self.opponent_name)
        if profile_db is not None:
            profile_db.profiles = []
        
        log_print(f"{self.agent_name}: 重置游戏状态", level="DEBUG")