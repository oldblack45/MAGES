"""
认知学习系统
实现认知库的自动更新、权重调整和持续学习机制
"""

import json
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from .world_cognition import WorldCognitionDB, WorldRecognition
from .agent_profile import MultiAgentProfileManager, AgentProfileDB
from .experiment_logger import log_print


class LearningMode(Enum):
    """学习模式"""
    CONSERVATIVE = 1    # 保守学习（慢更新）
    NORMAL = 2         # 正常学习
    AGGRESSIVE = 3     # 激进学习（快更新）
    ADAPTIVE = 4       # 自适应学习


class CognitiveLearningSystem:
    """认知学习系统：管理认知库的更新和学习过程"""
    
    def __init__(self, agent_name: str, learning_mode: LearningMode = LearningMode.NORMAL, 
                 similarity_threshold: float = 0.8):
        self.agent_name = agent_name
        self.learning_mode = learning_mode
        
        # 学习参数（根据模式调整）
        self._setup_learning_parameters()
        
        # 学习统计
        self.total_updates = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.learning_history = []
        
        # 自适应学习参数
        self.performance_window = 20  # 性能评估窗口
        self.performance_history = []
        
        
    
    def _is_prediction_correct(self, predicted_feedback: str, actual_feedback: str, llm_agent) -> bool:
        """
        智能判断预测是否正确
        先尝试规范化比较，如果失败则使用字符串匹配
        """
        prompt = f"""
        请判断以下两个反馈是否相同：
        
        预测反馈：{predicted_feedback}
        实际反馈：{actual_feedback}
        
        请判断预测反馈是否大致相同，并给出理由。
        
        回答格式：
        {{{{
            "is_correct": "是否大致相同，true或false",
            "reasoning": "理由，一句话描述"
        }}}}
        """
        response = llm_agent.get_response(prompt)
        if isinstance(response, dict):
            return response.get('is_correct', False)
        else:
            return False
    
    def _is_agent_reaction_correct(self, predicted_reaction: str, actual_reaction: str) -> bool:
        """
        判断Agent反应预测是否正确
        目前使用字符串匹配，未来可以扩展为语义匹配
        """
        # TODO: 未来可以为Agent反应也实现语义相似性比较
        log_print(f"使用字符串匹配比较Agent反应: '{predicted_reaction}' vs '{actual_reaction}'", level="DEBUG")
        return predicted_reaction == actual_reaction
    
    def _setup_learning_parameters(self):
        """根据学习模式设置参数"""
        if self.learning_mode == LearningMode.CONSERVATIVE:
            self.weight_increase = 0.1
            self.weight_decrease = 0.2
            self.min_weight = 0.2
            self.max_weight = 2.0
            self.threshold_adjustment = 0.05
        elif self.learning_mode == LearningMode.NORMAL:
            self.weight_increase = 0.2
            self.weight_decrease = 0.3
            self.min_weight = 0.1
            self.max_weight = 3.0
            self.threshold_adjustment = 0.1
        elif self.learning_mode == LearningMode.AGGRESSIVE:
            self.weight_increase = 0.3
            self.weight_decrease = 0.4
            self.min_weight = 0.05
            self.max_weight = 5.0
            self.threshold_adjustment = 0.2
        else:  # ADAPTIVE
            # 初始值，会动态调整
            self.weight_increase = 0.2
            self.weight_decrease = 0.3
            self.min_weight = 0.1
            self.max_weight = 3.0
            self.threshold_adjustment = 0.1
    
    def update_world_cognition(self, world_cognition: WorldCognitionDB, 
                             action: str, predicted_feedback: str, 
                             actual_feedback: str, llm_agent=None) -> Dict[str, Any]:
        """
        更新世界认知库并返回学习统计
        """
        log_print(f"更新世界认知: {action} -> {actual_feedback}", level="DEBUG")
        

        # 找到匹配的认知 - 只需要行为匹配
        matching_recognitions = [r for r in world_cognition.recognitions 
                               if r.action == action]
        
        # 对于同一行为应该只有一个认知记录
        if len(matching_recognitions) > 1:
            log_print(f"警告：发现{len(matching_recognitions)}个相同行为的认知记录，数据可能不一致", level="WARNING")
        elif len(matching_recognitions) == 0:
            log_print(f"未找到行为'{action}'的认知记录，创建新记录", level="DEBUG")
            # 如果没有找到匹配的认知记录，创建新的记录并立即生成经验            
            # 先创建基础记录
            world_cognition.add_recognition(
                action=action,
                feedback=actual_feedback,
                experience=f"执行{action}得到{actual_feedback}",  # 临时经验
                weight=1.0  # 新记录默认权重
            )
            
            # 立即用LLM生成经验
            if llm_agent:
                new_recognition = [r for r in world_cognition.recognitions if r.action == action][0]
                world_cognition.update_world_experiences(new_recognition, llm_agent)
                log_print(f"已为新认知生成经验: {new_recognition.experience}", level="DEBUG")
            
            # 更新结果统计
            update_result = {
                'action': action,
                'predicted_feedback': predicted_feedback,
                'actual_feedback': actual_feedback,
                'prediction_correct': False,  # 新记录视为预测错误
                'recognitions_updated': 1,
                'experience_regenerated': True,
                'timestamp': datetime.now().isoformat()
            }
            
            self.learning_history.append(update_result)
            return update_result
            
        matching_recognitions = [r for r in world_cognition.recognitions if r.action == action]
        
        # 智能判断预测是否正确
        prediction_correct = self._is_prediction_correct(predicted_feedback, actual_feedback, llm_agent)
        
        update_result = {
            'action': action,
            'predicted_feedback': predicted_feedback,
            'actual_feedback': actual_feedback,
            'prediction_correct': prediction_correct,
            'recognitions_updated': len(matching_recognitions),
            'experience_regenerated': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # 处理匹配到的认知记录
        for recognition in matching_recognitions:
            old_weight = recognition.weight
            
            if prediction_correct:
                log_print(f"世界认知预测正确，增加权重: {recognition.action} -> {recognition.feedback}", level="DEBUG")
                # 预测正确，增加权重
                self.successful_predictions += 1
                recognition.update_weight(self.weight_increase, self.min_weight, self.max_weight)
                update_result[f'weight_change_{recognition}'] = recognition.weight - old_weight
            else:
                log_print(f"世界认知预测错误，减少权重: {recognition.action} -> {recognition.feedback}", level="DEBUG")
                # 预测错误，减少权重并可能更新反馈
                self.failed_predictions += 1
                recognition.update_weight(-self.weight_decrease, self.min_weight, self.max_weight)
                
                # 如果权重过低，更新为实际反馈并重新生成经验
                if recognition.weight < world_cognition.weight_threshold:
                    log_print(f"世界认知权重过低，更新为实际反馈: {recognition.action} -> {recognition.feedback}", level="DEBUG")
                    recognition.feedback = actual_feedback
                    recognition.weight = 1.0
                    update_result['experience_regenerated'] = True
                    
                    # 重新生成经验
                    if llm_agent:
                        world_cognition.update_world_experiences(recognition, llm_agent)
                
                update_result[f'weight_change_{recognition}'] = recognition.weight - old_weight
        
        # 记录学习历史
        self.learning_history.append(update_result)
        self.total_updates += 1
        
        # 自适应学习调整
        if self.learning_mode == LearningMode.ADAPTIVE:
            self._adaptive_adjustment_world()
        
        return update_result
    
    def update_agent_profile(self, profile_db: AgentProfileDB,
                           target_agent: str, action: str, 
                           predicted_reaction: str, actual_reaction: str,
                           llm_agent=None) -> Dict[str, Any]:
        """
        更新Agent侧写认知库并返回学习统计
        """
        log_print(f"更新Agent侧写: {action} -> {actual_reaction}", level="DEBUG")
        
        # 找到匹配的侧写 - 只需要行为匹配
        matching_profiles = [p for p in profile_db.profiles 
                           if p.action == action]
        
        # 对于同一行为应该只有一个侧写记录
        if len(matching_profiles) > 1:
            log_print(f"警告：发现{len(matching_profiles)}个相同行为的侧写记录，数据可能不一致", level="WARNING")
        elif len(matching_profiles) == 0:
            # 如果没有找到匹配的侧写记录，创建新的记录并立即生成策略和经验
            log_print(f"未找到行为'{action}'的侧写记录，创建新记录", level="INFO")
            
            # 先创建基础记录
            profile_db.add_profile(
                action=action,
                reaction=actual_reaction,
                strategy=f"针对{action}采取{actual_reaction}",  # 临时策略
                experience=f"应对{target_agent}的{actual_reaction}反应",  # 临时经验
                weight=1.0  # 新记录默认权重
            )
            
            # 立即用LLM生成策略和经验
            if llm_agent:
                new_profile = [p for p in profile_db.profiles if p.action == action][0]
                profile_db.update_experience_and_strategies(new_profile, llm_agent)
                log_print(f"已为新侧写生成策略和经验: {new_profile.strategy}", level="INFO")
            
            # 更新结果统计
            update_result = {
                'target_agent': target_agent,
                'action': action,
                'predicted_reaction': predicted_reaction,
                'actual_reaction': actual_reaction,
                'prediction_correct': False,  # 新记录视为预测错误
                'profiles_updated': 1,
                'strategy_regenerated': True,
                'timestamp': datetime.now().isoformat()
            }
            self.learning_history.append(update_result)
            return update_result
        matching_profile = [p for p in profile_db.profiles if p.action == action][0]
        
        # 智能判断预测是否正确（Agent反应目前使用字符串匹配）
        prediction_correct = self._is_agent_reaction_correct(predicted_reaction, actual_reaction)
        
        update_result = {
            'target_agent': target_agent,
            'action': action,
            'predicted_reaction': predicted_reaction,
            'actual_reaction': actual_reaction,
            'prediction_correct': prediction_correct,
            'profiles_updated': 1,
            'strategy_regenerated': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # 处理匹配到的侧写记录
        old_weight = matching_profile.weight
            
        if prediction_correct:
            log_print(f"Agent侧写预测正确，增加权重: {matching_profile.action} -> {matching_profile.reaction}", level="DEBUG")
            # 预测正确，增加权重
            matching_profile.update_weight(self.weight_increase, self.min_weight, self.max_weight)
            update_result[f'weight_change_{matching_profile}'] = matching_profile.weight - old_weight
        else:
            log_print(f"Agent侧写预测错误，减少权重: {matching_profile.action} -> {matching_profile.reaction}", level="DEBUG")
            # 预测错误，减少权重并可能更新反应
            matching_profile.update_weight(-self.weight_decrease, self.min_weight, self.max_weight)
                
            # 如果权重过低，更新为实际反应并重新生成策略
            if matching_profile.weight < profile_db.weight_threshold:
                log_print(f"Agent侧写权重过低，更新为实际反应: {matching_profile.action} -> {matching_profile.reaction}", level="DEBUG")
                matching_profile.reaction = actual_reaction
                matching_profile.weight = 1.0
                update_result['strategy_regenerated'] = True
                    
                # 重新生成策略和经验
                if llm_agent:
                    profile_db.update_experience_and_strategies(matching_profile, llm_agent)
                
                update_result[f'weight_change_{matching_profile}'] = matching_profile.weight - old_weight
        
        self.learning_history.append(update_result)
        return update_result
        
    def _adaptive_adjustment_world(self):
        """自适应调整世界认知学习参数"""
        if len(self.learning_history) < self.performance_window:
            return
        
        # 计算最近性能
        recent_updates = self.learning_history[-self.performance_window:]
        correct_predictions = sum(1 for update in recent_updates 
                                if update.get('prediction_correct', False))
        accuracy = correct_predictions / len(recent_updates)
        
        self.performance_history.append(accuracy)
        
        # 根据性能调整学习参数
        if accuracy > 0.8:  # 性能很好，可以更保守
            self.weight_increase = max(0.05, self.weight_increase - 0.05)
            self.weight_decrease = max(0.1, self.weight_decrease - 0.05)
        elif accuracy < 0.4:  # 性能较差，需要更激进
            self.weight_increase = min(0.5, self.weight_increase + 0.05)
            self.weight_decrease = min(0.6, self.weight_decrease + 0.05)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        total_predictions = self.successful_predictions + self.failed_predictions
        accuracy = (self.successful_predictions / total_predictions) if total_predictions > 0 else 0
        
        stats = {
            'agent_name': self.agent_name,
            'learning_mode': self.learning_mode.name,
            'total_updates': self.total_updates,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'prediction_accuracy': accuracy,
            'learning_parameters': {
                'weight_increase': self.weight_increase,
                'weight_decrease': self.weight_decrease,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight
            }
        }
        
        # 最近性能统计
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # 最近10次
            mean_acc = sum(recent_performance) / len(recent_performance)
            std_acc = (sum((x - mean_acc) ** 2 for x in recent_performance) / len(recent_performance)) ** 0.5
            stats['recent_performance'] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'trend': self._calculate_performance_trend()
            }
        
        return stats
    
    def _calculate_performance_trend(self) -> str:
        """计算性能趋势"""
        if len(self.performance_history) < 5:
            return 'insufficient_data'
        
        recent_performance = self.performance_history[-5:]
        older_performance = self.performance_history[-10:-5] if len(self.performance_history) >= 10 else self.performance_history[:-5]
        
        if not older_performance:
            return 'insufficient_data'
        
        recent_avg = sum(recent_performance) / len(recent_performance)
        older_avg = sum(older_performance) / len(older_performance)
        
        if recent_avg > older_avg + 0.05:
            return 'improving'
        elif recent_avg < older_avg - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def export_learning_report(self, output_file: str):
        """导出学习报告"""
        stats = self.get_learning_statistics()
        
        report = f"""# {self.agent_name} 认知学习报告

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 学习概况
- 学习模式：{stats['learning_mode']}
- 总更新次数：{stats['total_updates']}
- 预测准确率：{stats['prediction_accuracy']:.2%}
- 成功预测：{stats['successful_predictions']}
- 失败预测：{stats['failed_predictions']}

## 学习参数
- 权重增加幅度：{stats['learning_parameters']['weight_increase']}
- 权重减少幅度：{stats['learning_parameters']['weight_decrease']}
- 最小权重：{stats['learning_parameters']['min_weight']}
- 最大权重：{stats['learning_parameters']['max_weight']}

"""
        
        if 'recent_performance' in stats:
            report += f"""## 最近性能
- 平均准确率：{stats['recent_performance']['mean_accuracy']:.2%}
- 准确率标准差：{stats['recent_performance']['std_accuracy']:.3f}
- 性能趋势：{stats['recent_performance']['trend']}

"""
        
        # 学习历史摘要
        if self.learning_history:
            recent_history = self.learning_history[-10:]
            report += "## 最近学习记录\n"
            for i, record in enumerate(recent_history, 1):
                status = "✓" if record.get('prediction_correct', False) else "✗"
                if 'actual_feedback' in record:
                    report += f"{i}. {status} {record['action']} -> {record['actual_feedback']}\n"
                else:
                    report += f"{i}. {status} {record['action']} -> {record['actual_reaction']}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        log_print(f"学习报告已导出到 {output_file}")
        
        return report


class CognitionMaintenance:
    """认知库维护工具"""
    
    @staticmethod
    def backup_cognition(cognition_dir: str, backup_dir: str):
        """备份认知库"""
        import shutil
        backup_path = f"{backup_dir}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(cognition_dir, backup_path)
        return backup_path
    
    @staticmethod
    def merge_cognition_dbs(source_dirs: List[str], target_dir: str):
        """合并多个认知库"""
        # 实现认知库合并逻辑
        # 这里只是框架，具体实现需要根据需求调整
        pass
    
    @staticmethod
    def analyze_cognition_conflicts(world_cognition: WorldCognitionDB) -> List[Dict]:
        """分析认知冲突"""
        conflicts = []
        
        # 检查同一行为的不同反馈
        action_feedbacks = {}
        for recognition in world_cognition.recognitions:
            action = recognition.action
            if action not in action_feedbacks:
                action_feedbacks[action] = []
            action_feedbacks[action].append(recognition)
        
        for action, recognitions in action_feedbacks.items():
            if len(recognitions) > 1:
                feedbacks = [r.feedback for r in recognitions]
                unique_feedbacks = set(feedbacks)
                if len(unique_feedbacks) > 1:
                    conflicts.append({
                        'action': action,
                        'conflicting_feedbacks': list(unique_feedbacks),
                        'weights': [r.weight for r in recognitions]
                    })
        
        return conflicts