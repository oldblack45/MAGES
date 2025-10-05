"""
囚徒困境策略识别实验
测试认知Agent的策略识别能力
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from simulation.examples.PrisonerDilemma.entity.prisoner_dilemma_world import (
    PrisonerDilemmaWorld, PrisonerDilemmaGame
)
from simulation.examples.PrisonerDilemma.entity.cognitive_prisoner_agent import CognitivePrisonerAgent
from simulation.examples.PrisonerDilemma.entity.tit_for_tat_agent import (
    TitForTatAgent, AlwaysCooperateAgent, AlwaysDefectAgent,
    GrimTriggerAgent
)
from simulation.models.cognitive.experiment_logger import ExperimentLogger, init_logger, log_print
from visiualize.strategy_identification_plots import (
    extract_convergence_frame, plot_convergence_box, plot_confidence_examples
)


class StrategyIdentificationExperiment:
    """策略识别实验类"""
    
    def __init__(self, experiment_name: str = "strategy_identification_experiment"):
        self.experiment_name = experiment_name
        self.world = PrisonerDilemmaWorld(experiment_name)
        self.results = []
        
        # 测试策略列表
        self.test_strategies = {
            'Tit-for-Tat': TitForTatAgent,
            'Always-Cooperate': AlwaysCooperateAgent,
            'Always-Defect': AlwaysDefectAgent,
            'Grim-Trigger': GrimTriggerAgent
        }
        
        log_print("策略识别实验初始化完成", level="INFO")
    
    def run_single_experiment(self, strategy_name: str, strategy_class, 
                            rounds: int = 50, games: int = 1) -> Dict[str, Any]:
        """
        运行单个策略的识别实验
        
        Args:
            strategy_name: 策略名称
            strategy_class: 策略类或函数
            rounds: 每场游戏的轮数
            games: 游戏场次
            
        Returns:
            实验结果
        """
        log_print(f"开始测试策略识别: {strategy_name}", level="INFO")
        
        experiment_results = []
        
        for game_num in range(games):
            log_print(f"运行第{game_num+1}/{games}场游戏", level="INFO")
            
            # 创建对手Agent（使用固定策略）
            opponent = strategy_class(f"Opponent_{strategy_name}")
            
            # 创建认知Agent
            cognitive_agent = CognitivePrisonerAgent(
                agent_name="CognitiveAgent",
                opponent_name=opponent.agent_name,
                experiment_logger=self.world.experiment_logger
            )
            
            # 创建游戏
            game = self.world.create_game(cognitive_agent, opponent, rounds)
            
            # 运行游戏
            game_result = self.world.run_game(game)
            
            # 评估策略识别准确性
            identification_result = cognitive_agent.get_strategy_identification_accuracy(strategy_name)
            # 计算收敛轮数（首次正确识别的轮次）。若始终不正确，记为 total_rounds
            converge_round = rounds
            for idx, snapshot in enumerate(cognitive_agent.hypotheses_history, start=1):
                top_key = max(snapshot, key=snapshot.get) if snapshot else 'unknown'
                mapping = {
                    'Tit-for-Tat': 'tit_for_tat',
                    'Always-Cooperate': 'always_cooperate',
                    'Always-Defect': 'always_defect',
                    'Grim-Trigger': 'grim_trigger'
                }
                if mapping.get(strategy_name, strategy_name.lower().replace('-', '_')) == top_key:
                    converge_round = idx
                    break
            
            # 记录结果
            experiment_result = {
                'game_number': game_num + 1,
                'strategy_name': strategy_name,
                'total_rounds': rounds,
                'game_result': game_result,
                'identification_result': identification_result,
                'cognitive_agent_stats': {
                    'cooperation_rate': game_result['cooperation_rates']['player1'],
                    'final_score': game_result['player1_final_score']
                },
                'opponent_stats': {
                    'cooperation_rate': game_result['cooperation_rates']['player2'],
                    'final_score': game_result['player2_final_score']
                },
                'converge_round': converge_round,
                'hypotheses_history': cognitive_agent.hypotheses_history
            }
            
            experiment_results.append(experiment_result)
            
            # 重置Agent状态
            cognitive_agent.reset_game()
            opponent.reset_game()
        
        # 计算策略识别统计
        identification_stats = self._calculate_identification_stats(experiment_results, strategy_name)
        
        return {
            'strategy_name': strategy_name,
            'total_games': games,
            'rounds_per_game': rounds,
            'individual_results': experiment_results,
            'identification_statistics': identification_stats
        }
    
    def run_full_experiment(self, rounds_per_game: int = 50, games_per_strategy: int = 3) -> Dict[str, Any]:
        """
        运行完整的策略识别实验
        
        Args:
            rounds_per_game: 每场游戏的轮数
            games_per_strategy: 每种策略测试的游戏场次
            
        Returns:
            完整实验结果
        """
        log_print(f"开始完整策略识别实验", level="INFO")
        log_print(f"测试参数: {rounds_per_game}轮/游戏, {games_per_strategy}场/策略", level="INFO")
        
        all_results = []
        overall_stats = {
            'correct_identifications': 0,
            'total_identifications': 0,
            'strategy_accuracies': {}
        }
        
        # 测试每种策略
        for strategy_name, strategy_class in self.test_strategies.items():
            log_print(f"测试策略: {strategy_name}", level="INFO")
            
            strategy_result = self.run_single_experiment(
                strategy_name, strategy_class, rounds_per_game, games_per_strategy
            )
            
            all_results.append(strategy_result)
            
            # 更新总体统计
            stats = strategy_result['identification_statistics']
            overall_stats['correct_identifications'] += stats['correct_count']
            overall_stats['total_identifications'] += stats['total_count']
            overall_stats['strategy_accuracies'][strategy_name] = stats['accuracy']
            
            log_print(f"{strategy_name} 识别准确率: {stats['accuracy']:.2%}", level="INFO")
        
        # 计算总体准确率
        overall_accuracy = (overall_stats['correct_identifications'] / 
                          overall_stats['total_identifications'] 
                          if overall_stats['total_identifications'] > 0 else 0)
        
        overall_stats['overall_accuracy'] = overall_accuracy
        
        # 生成实验总结
        experiment_summary = {
            'experiment_name': self.experiment_name,
            'experiment_time': datetime.now().isoformat(),
            'parameters': {
                'rounds_per_game': rounds_per_game,
                'games_per_strategy': games_per_strategy,
                'tested_strategies': list(self.test_strategies.keys())
            },
            'overall_statistics': overall_stats,
            'strategy_results': all_results,
            'conclusions': self._generate_conclusions(overall_stats, all_results)
        }
        
        # 保存结果为标准 experiment_summary.json
        # 文件位置：{experiment_dir}/summary/experiment_summary.json
        self.world.experiment_logger.save_experiment_summary(experiment_summary)

        # 自动绘图（仅两张）：收敛轮数箱线图 + 置信度收敛曲线示例
        try:
            charts_dir = os.path.join(str(self.world.experiment_logger.experiment_dir), 'evaluation_charts')
            os.makedirs(charts_dir, exist_ok=True)
            df_conv = extract_convergence_frame(experiment_summary)
            plot_convergence_box(df_conv, os.path.join(charts_dir, 'convergence_rounds.png'))
            plot_confidence_examples(experiment_summary, os.path.join(charts_dir, 'confidence_convergence_examples.png'))
            log_print(f"已生成实验图表: {charts_dir}", level="INFO")
        except Exception as e:
            log_print(f"自动绘图失败: {e}", level="ERROR")
        
        log_print(f"实验完成! 总体识别准确率: {overall_accuracy:.2%}", level="INFO")
        
        return experiment_summary
    
    def _calculate_identification_stats(self, experiment_results: List[Dict], true_strategy: str) -> Dict[str, Any]:
        """计算策略识别统计"""
        correct_count = 0
        total_count = len(experiment_results)
        confidence_scores = []
        
        for result in experiment_results:
            identification = result['identification_result']
            if identification['is_correct']:
                correct_count += 1
            confidence_scores.append(identification['confidence'])
        
        return {
            'true_strategy': true_strategy,
            'total_count': total_count,
            'correct_count': correct_count,
            'accuracy': correct_count / total_count if total_count > 0 else 0,
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'confidence_scores': confidence_scores
        }
    
    def _generate_conclusions(self, overall_stats: Dict, strategy_results: List[Dict]) -> List[str]:
        """生成实验结论"""
        conclusions = []
        
        # 总体表现评估
        overall_accuracy = overall_stats['overall_accuracy']
        if overall_accuracy >= 0.8:
            conclusions.append("认知Agent表现出色，能够高准确度识别不同策略")
        elif overall_accuracy >= 0.6:
            conclusions.append("认知Agent表现良好，具备一定的策略识别能力")
        else:
            conclusions.append("认知Agent的策略识别能力有待提升")
        
        # 各策略识别难度分析
        strategy_accuracies = overall_stats['strategy_accuracies']
        easiest_strategy = max(strategy_accuracies, key=strategy_accuracies.get)
        hardest_strategy = min(strategy_accuracies, key=strategy_accuracies.get)
        
        conclusions.append(f"最容易识别的策略: {easiest_strategy} (准确率: {strategy_accuracies[easiest_strategy]:.2%})")
        conclusions.append(f"最难识别的策略: {hardest_strategy} (准确率: {strategy_accuracies[hardest_strategy]:.2%})")
        
        # 具体策略分析
        for result in strategy_results:
            strategy_name = result['strategy_name']
            accuracy = result['identification_statistics']['accuracy']
            avg_confidence = result['identification_statistics']['average_confidence']
            
            if accuracy == 1.0:
                conclusions.append(f"{strategy_name}策略识别完美，置信度{avg_confidence:.2f}")
            elif accuracy >= 0.8:
                conclusions.append(f"{strategy_name}策略识别表现良好")
            else:
                conclusions.append(f"{strategy_name}策略识别存在困难，需要改进算法")
        
        return conclusions
    
    def save_detailed_report(self, results: Dict[str, Any], filename: str = None):
        """保存详细报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_identification_report_{timestamp}.json"
        
        # 创建报告目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        filepath = os.path.join(report_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        log_print(f"详细报告已保存: {filepath}", level="INFO")
        
        return filepath


def run_quick_test():
    """运行快速测试"""
    experiment = StrategyIdentificationExperiment("prisoner_dilemma_experiment_quick_test")
    
    # 只测试一报还一报策略
    result = experiment.run_single_experiment(
        'Tit-for-Tat', 
        TitForTatAgent, 
        rounds=20, 
        games=1
    )
    
    print(f"快速测试结果:")
    print(f"策略: {result['strategy_name']}")
    print(f"识别准确率: {result['identification_statistics']['accuracy']:.2%}")
    
    return result


def run_full_test():
    """运行完整测试"""
    experiment = StrategyIdentificationExperiment("prisoner_dilemma_experiment_full_test")
    
    results = experiment.run_full_experiment(
        rounds_per_game=20,
        games_per_strategy=10
    )
    
    # 保存详细报告
    experiment.save_detailed_report(results)
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("策略识别实验结果摘要")
    print("="*50)
    print(f"总体识别准确率: {results['overall_statistics']['overall_accuracy']:.2%}")
    print("\n各策略识别准确率:")
    for strategy, accuracy in results['overall_statistics']['strategy_accuracies'].items():
        print(f"  {strategy}: {accuracy:.2%}")
    
    print("\n实验结论:")
    for i, conclusion in enumerate(results['conclusions'], 1):
        print(f"  {i}. {conclusion}")
    
    return results


if __name__ == "__main__":
    print("囚徒困境策略识别实验")
    print("选择运行模式:")
    print("1. 快速测试 (单策略, 20轮, 1场)")
    print("2. 完整测试 (所有策略, 20轮, 2场/策略)")
    
    choice = input("请输入选择 (1或2): ").strip()
    
    if choice == "1":
        run_quick_test()
    elif choice == "2":
        run_full_test()
    else:
        print("无效选择，运行快速测试")
        run_quick_test()