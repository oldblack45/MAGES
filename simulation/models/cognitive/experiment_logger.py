"""
实验日志管理器
为每个实验创建独立的文件夹，实时记录认知数据和决策历史
包含增强日志功能：控制台输出记录和LLM调用统计
"""
import json
import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from collections import defaultdict

# 全局日志记录器实例
_global_logger = None

def get_logger() -> 'ExperimentLogger':
    """获取全局日志记录器实例"""
    global _global_logger
    if _global_logger is None:
        raise RuntimeError("ExperimentLogger not initialized. Call init_logger() first.")
    return _global_logger

def init_logger(experiment_name: str = None, base_dir: str = "./experiments") -> 'ExperimentLogger':
    """初始化全局日志记录器"""
    global _global_logger
    _global_logger = ExperimentLogger(experiment_name, base_dir)
    return _global_logger

def log_print(message: str, level: str = "INFO"):
    """全局日志打印函数"""
    logger = get_logger()
    logger.log_print(message, level)

def record_llm_call(call_details: str = ""):
    """全局LLM调用记录函数"""
    logger = get_logger()
    logger.record_llm_call(call_details)


class ExperimentLogger:
    """
    实验日志管理器
    - 为每个实验创建独立的文件夹
    - 实时记录认知数据变化
    - 统一管理决策历史和认知库更新
    - 增强功能：控制台输出记录和LLM调用统计
    """
    
    def __init__(self, experiment_name: str = None, base_dir: str = "./experiments"):
        """
        初始化实验日志管理器
        
        Args:
            experiment_name: 实验名称，如果为None则自动生成
            base_dir: 实验日志基础目录
        """
        self.base_dir = Path(base_dir)
        
        # 生成实验标识符
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.base_dir / experiment_name
        
        # 创建实验目录结构
        self._setup_experiment_directory()
        
        # 记录实验开始时间
        self.start_time = datetime.now()
        self._log_experiment_info()
        
        # 初始化增强日志功能
        self._setup_enhanced_logging()
        
        # 初始化评测数据输出
        self.init_evaluation_output()
        
        self.log_print(f"实验日志系统已初始化: {self.experiment_dir}", level="INFO")
    
    def _setup_experiment_directory(self):
        """建立实验目录结构"""
        # 创建主目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.cognition_dir = self.experiment_dir / "cognition_data"
        self.decisions_dir = self.experiment_dir / "decisions"
        self.logs_dir = self.experiment_dir / "logs"
        self.summary_dir = self.experiment_dir / "summary"
        
        for dir_path in [self.cognition_dir, self.decisions_dir, self.logs_dir, self.summary_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def _log_experiment_info(self):
        """记录实验基本信息"""
        experiment_info = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "base_dir": str(self.base_dir),
            "experiment_dir": str(self.experiment_dir),
            "status": "running"
        }
        
        info_file = self.experiment_dir / "experiment_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=2)
    
    def _setup_enhanced_logging(self):
        """设置增强日志功能"""
        # 控制台日志文件
        self.console_log_file = self.logs_dir / 'console_output.log'
        self.llm_calls_log_file = self.logs_dir / 'llm_calls.csv'
        
        # LLM调用统计
        self.llm_call_stats = defaultdict(lambda: defaultdict(int))  # {step: {country: count}}
        self.total_llm_calls = 0
        self.current_step = 0
        self.current_country = None
        
        
        # 设置Python标准日志
        self.console_logger = logging.getLogger(f'ConsoleLogger_{self.experiment_name}')
        self.console_logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别
        
        # 清除现有的处理器
        for handler in self.console_logger.handlers[:]:
            self.console_logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.console_log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 设置为DEBUG级别
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.console_logger.addHandler(file_handler)
        
        # 初始化LLM调用记录CSV
        self._init_llm_calls_csv()
    
    def _init_llm_calls_csv(self):
        """初始化LLM调用记录CSV文件"""
        if not self.llm_calls_log_file.exists():
            with open(self.llm_calls_log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'country', 'timestamp', 'total_calls_this_step', 
                               'cumulative_calls', 'call_details'])
    
    def set_step_context(self, step: int, country: str = None):
        """设置当前步骤和国家上下文"""
        self.current_step = step
        self.current_country = country
    
    def log_print(self, message: str, level: str = "INFO"):
        """
        记录print输出，同时输出到控制台和文件
        Args:
            message: 要记录的消息
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        """
        # 格式化消息
        step_info = f"[Step {self.current_step}]" if self.current_step > 0 else "[Init]"
        country_info = f"[{self.current_country}]" if self.current_country else "[System]"
        
        formatted_message = f"{step_info}{country_info} {message}"
        
        # 输出到控制台
        print(formatted_message)
        
        # 记录到文件
        log_entry = f"{step_info}{country_info} - {level} - {message}"
        
        # 根据日志级别调用相应的日志方法
        log_method = getattr(self.console_logger, level.lower(), self.console_logger.info)
        log_method(log_entry)
    
    def record_llm_call(self, call_details: str = ""):
        """
        记录LLM调用
        Args:
            country: 调用LLM的国家
            call_details: 调用详情描述
        """
        
        step = self.current_step
        # 更新统计
        self.llm_call_stats[step]["Unknown"] += 1
        self.total_llm_calls += 1
        # 记录到日志
        self.log_print(f"LLM调用 #{self.total_llm_calls} - {call_details}", level="INFO")
    
    def log_step_llm_summary(self, step: int):
        """记录每个步骤的LLM调用汇总"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        step_stats = self.llm_call_stats[step]
        
        if not step_stats:
            # 如果没有调用记录，记录零调用
            data = [step, "System", timestamp, 0, self.total_llm_calls, "No LLM calls in this step"]
            self._write_csv_row(self.llm_calls_log_file, data)
            return
        
        # 为每个国家记录LLM调用统计
        for country, calls in step_stats.items():
            call_details = f"{calls} calls made by {country} in step {step}"
            data = [
                step,
                country,
                timestamp,
                calls,
                self.total_llm_calls,
                call_details
            ]
            self._write_csv_row(self.llm_calls_log_file, data)
            
            # 同时记录到控制台日志
            self.log_print(f"Step {step} LLM统计 - {country}: {calls}次调用", level="INFO")

    def get_llm_stats(self, step: int = None) -> Dict[str, Any]:
        """
        获取LLM调用统计
        Args:
            step: 特定步骤，如果为None则返回所有统计
        Returns:
            统计信息字典
        """
        if step is not None:
            return {
                'step': step,
                'country_stats': dict(self.llm_call_stats[step]),
                'step_total': sum(self.llm_call_stats[step].values()),
                'cumulative_total': self.total_llm_calls
            }
        else:
            return {
                'all_steps': dict(self.llm_call_stats),
                'total_calls': self.total_llm_calls,
                'steps_with_calls': list(self.llm_call_stats.keys())
            }
    
    def _write_csv_row(self, file_path, row_data):
        """写入CSV行数据"""
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            error_msg = f"Error writing to log file {file_path}: {str(e)}"
            print(error_msg)
            if hasattr(self, 'console_logger'):
                self.console_logger.error(error_msg)
    
    def get_agent_cognition_dir(self, agent_name: str) -> Path:
        """获取特定agent的认知数据目录"""
        agent_dir = self.cognition_dir / agent_name
        agent_dir.mkdir(exist_ok=True)
        return agent_dir
    
    def get_agent_decisions_dir(self, agent_name: str) -> Path:
        """获取特定agent的决策记录目录"""
        agent_dir = self.decisions_dir / agent_name
        agent_dir.mkdir(exist_ok=True)
        return agent_dir
    
    def log_cognition_update(self, agent_name: str, cognition_type: str, 
                           data: Any, timestamp: datetime = None):
        """
        实时记录认知数据更新
        
        Args:
            agent_name: agent名称
            cognition_type: 认知类型 (world_cognition, agent_profile, etc.)
            data: 认知数据
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 保存到相应的认知数据文件
        agent_dir = self.get_agent_cognition_dir(agent_name)
        cognition_file = agent_dir / f"{cognition_type}.json"
        
        # 创建认知数据记录
        cognition_record = {
            "agent_name": agent_name,
            "cognition_type": cognition_type,
            "timestamp": timestamp.isoformat(),
            "data": data
        }
        
        # 保存到文件
        with open(cognition_file, 'w', encoding='utf-8') as f:
            json.dump(cognition_record, f, ensure_ascii=False, indent=2)
        
        # 同时记录到更新历史中
        self._log_cognition_history(agent_name, cognition_type, timestamp)
    
    def _log_cognition_history(self, agent_name: str, cognition_type: str, 
                              timestamp: datetime):
        """记录认知更新历史"""
        history_file = self.cognition_dir / "cognition_update_history.csv"
        
        # 如果文件不存在，创建并写入标题行
        if not history_file.exists():
            with open(history_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'agent_name', 'cognition_type', 'update_time'])
        
        # 追加更新记录
        with open(history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp.isoformat(), agent_name, cognition_type, 
                           datetime.now().isoformat()])
    
    def log_decision(self, agent_name: str, decision_data: Dict[str, Any]):
        """
        记录决策历史
        
        Args:
            agent_name: agent名称
            decision_data: 决策数据
        """
        agent_dir = self.get_agent_decisions_dir(agent_name)
        decisions_file = agent_dir / "decision_history.json"
        
        # 加载现有决策历史
        if decisions_file.exists():
            with open(decisions_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # 添加新决策
        history.append(decision_data)
        
        # 保存更新后的历史
        with open(decisions_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        # 记录到CSV便于分析
        self._log_decision_to_csv(agent_name, decision_data)
    
    def _log_decision_to_csv(self, agent_name: str, decision_data: Dict[str, Any]):
        """将决策记录到CSV文件便于分析"""
        csv_file = self.decisions_dir / "all_decisions.csv"
        
        # 如果文件不存在，创建并写入标题行
        if not csv_file.exists():
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'agent_name', 'chosen_action', 
                               'candidate_actions', 'satisfaction_score', 'reasoning_depth'])
        
        # 追加决策记录
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                decision_data.get('timestamp', ''),
                agent_name,
                decision_data.get('chosen_action', ''),
                str(decision_data.get('candidate_actions', [])),
                decision_data.get('satisfaction_score', 0),
                decision_data.get('reasoning_depth', 0)
            ])
    
    def log_game_event(self, event_type: str, event_data: Dict[str, Any]):
        """记录游戏事件"""
        event_file = self.logs_dir / "game_events.json"
        
        # 加载现有事件
        if event_file.exists():
            with open(event_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
        else:
            events = []
        
        # 添加新事件
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": event_data
        }
        events.append(event_record)
        
        # 保存
        with open(event_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
    
    def save_experiment_summary(self, summary_data: Dict[str, Any]):
        """保存实验总结"""
        summary_file = self.summary_dir / "experiment_summary.json"
        
        # 添加实验基本信息
        summary_data.update({
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": str(datetime.now() - self.start_time)
        })
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        # 更新实验信息状态
        info_file = self.experiment_dir / "experiment_info.json"
        with open(info_file, 'r', encoding='utf-8') as f:
            experiment_info = json.load(f)
        
        experiment_info.update({
            "end_time": datetime.now().isoformat(),
            "status": "completed"
        })
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=2)
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """获取实验统计信息"""
        stats = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "running_time": str(datetime.now() - self.start_time),
            "agents": [],
            "total_decisions": 0,
            "total_cognition_updates": 0
        }
        
        # 统计各agent的决策数量
        if self.decisions_dir.exists():
            for agent_dir in self.decisions_dir.iterdir():
                if agent_dir.is_dir():
                    agent_name = agent_dir.name
                    decisions_file = agent_dir / "decision_history.json"
                    
                    if decisions_file.exists():
                        with open(decisions_file, 'r', encoding='utf-8') as f:
                            decisions = json.load(f)
                        
                        agent_stats = {
                            "agent_name": agent_name,
                            "decision_count": len(decisions)
                        }
                        stats["agents"].append(agent_stats)
                        stats["total_decisions"] += len(decisions)
        
        # 统计认知更新数量
        cognition_history = self.cognition_dir / "cognition_update_history.csv"
        if cognition_history.exists():
            with open(cognition_history, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                stats["total_cognition_updates"] = sum(1 for _ in reader)
        
        return stats
    
    def finalize_experiment(self):
        """实验结束时的最终处理"""
            # 记录实验总结
        summary_data = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_llm_calls': self.total_llm_calls,
            'steps_completed': self.current_step,
            'llm_stats_by_step': dict(self.llm_call_stats),
            'experiment_directory': str(self.experiment_dir)
        }
        
        # 保存LLM调用总结
        llm_summary_file = self.summary_dir / 'llm_call_summary.json'
        with open(llm_summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        final_msg = f"实验结束 - 总共完成{self.current_step}步，LLM调用{self.total_llm_calls}次"
        self.log_print(final_msg, level="INFO")
        
        return llm_summary_file
    
    def init_evaluation_output(self):
        """初始化评测数据输出文件"""
        evaluation_dir = self.experiment_dir / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        
        self.evaluation_data_file = evaluation_dir / "evaluation_data.json"
        
        # 初始化空的评测数据列表
        with open(self.evaluation_data_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)
        
        self.evaluation_data = []
        self.log_print("评测数据输出文件已初始化", level="INFO")
    
    def log_evaluation_round(self, round_num: int, actor: str, declaration: str, 
                           action: str, world_feedback: str, timestamp: str = None):
        """
        实时记录评测格式的轮次数据
        
        Args:
            round_num: 轮次编号
            actor: 行动者 (美国/苏联)
            declaration: 声明内容
            action: 行动内容
            world_feedback: 世界反馈
            timestamp: 时间戳，如果为None则自动生成
        """
        if timestamp is None:
            timestamp = f"t+{round_num * 5}" 
        round_data = {
            "round": round_num,
            "timestamp": timestamp,
            "actor": actor,
            "declaration": declaration,
            "action": action,
            "world_feedback": world_feedback
        }
        
        # 添加到内存中的数据列表
        if not hasattr(self, 'evaluation_data'):
            self.evaluation_data = []
        self.evaluation_data.append(round_data)
        
        # 实时写入文件
        if hasattr(self, 'evaluation_data_file'):
            try:
                with open(self.evaluation_data_file, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluation_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.log_print(f"写入评测数据失败: {e}", level="ERROR")
        
        self.log_print(f"记录评测轮次 {round_num}: {actor} - {action[:20]}...", level="INFO")
    
    def export_from_structured_memory(self, structured_memory_data: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        从结构化世界记忆直接导出评测格式数据
        
        Args:
            structured_memory_data: 来自StructuredWorldMemory的数据
            
        Returns:
            评测格式的数据列表
        """
        evaluation_data = []
        
        if "rounds" not in structured_memory_data:
            self.log_print("结构化记忆中没有找到rounds数据", level="WARNING")
            return evaluation_data
        
        for round_data in structured_memory_data["rounds"]:
            round_num = round_data.get("round", 0)
            timestamp = round_data.get("timestamp", f"t+{round_num * 5}")
            world_feedback = round_data.get("world_feedback", "")
            
            # 美国行动
            if "america" in round_data:
                america_data = round_data["america"]
                evaluation_data.append({
                    "round": round_num * 2 - 1,  # 确保轮次唯一
                    "timestamp": timestamp,
                    "actor": "国家A",
                    "declaration": america_data.get("declaration", ""),
                    "action": america_data.get("action", ""),
                    "world_feedback": world_feedback
                })
            
            # 苏联行动
            if "soviet" in round_data:
                soviet_data = round_data["soviet"]
                evaluation_data.append({
                    "round": round_num * 2,  # 确保轮次唯一
                    "timestamp": timestamp,
                    "actor": "国家B",
                    "declaration": soviet_data.get("declaration", ""),
                    "action": soviet_data.get("action", ""),
                    "world_feedback": world_feedback
                })
        
        # 按轮次排序
        evaluation_data.sort(key=lambda x: x["round"])
        
        # 保存到文件
        evaluation_dir = self.experiment_dir / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        
        evaluation_file = evaluation_dir / "structured_memory_evaluation_data.json"
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        self.log_print(f"从结构化记忆导出 {len(evaluation_data)} 条评测数据", level="INFO")
        return evaluation_data
    
    def run_evaluation(self, structured_memory_data: Dict[str, Any] = None, 
                      weights: Dict[str, float] = None, save_results: bool = True) -> Any:
        """
        运行模型效果评测
        
        Args:
            structured_memory_data: 结构化世界记忆数据，如果为None则使用实时记录的数据
            weights: 各指标权重字典 {"ea": 0.25, "as": 0.25, "sr": 0.25, "om": 0.25}
            save_results: 是否保存评测结果
            
        Returns:
            评测结果对象
        """
        try:
            from .evaluation_system import ModelEvaluationSystem
            
            # 获取评测数据
            if structured_memory_data:
                evaluation_data = self.export_from_structured_memory(structured_memory_data)
            elif hasattr(self, 'evaluation_data') and self.evaluation_data:
                evaluation_data = self.evaluation_data
            else:
                self.log_print("没有找到可评测的数据", level="WARNING")
                return None
            
            if not evaluation_data:
                self.log_print("评测数据为空", level="WARNING")
                return None
            
            # 创建评测系统并运行评测
            evaluator = ModelEvaluationSystem(weights)
            result = evaluator.evaluate(evaluation_data)
            
            # 保存评测结果
            if save_results:
                evaluation_dir = self.experiment_dir / "evaluation"
                evaluation_dir.mkdir(exist_ok=True)
                
                result_file = evaluation_dir / "evaluation_results.json"
                evaluator.export_results_to_json(result, str(result_file))
                
                # 记录到实验日志
                self.log_print(f"评测完成 - 最终得分: {result.final_score:.3f}", level="INFO")
                self.log_print(f"EA: {result.ea_score:.3f}, AS: {result.as_score:.3f}, SR: {result.sr_score:.3f}, OM: {result.om_score:.3f}", level="INFO")
            
            return result
                
        except ImportError as e:
            self.log_print(f"评测系统导入失败: {e}", level="ERROR")
            return None
        except Exception as e:
            self.log_print(f"评测过程出现错误: {e}", level="ERROR")
            return None

    def __str__(self):
        return f"ExperimentLogger(name={self.experiment_name}, dir={self.experiment_dir})"