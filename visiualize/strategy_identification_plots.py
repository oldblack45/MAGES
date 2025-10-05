"""
Generate two figures from experiment_summary.json:
1) Convergence rounds (box + strip)
2) Confidence convergence curves (one example per strategy)

Usage:
python visiualize/strategy_identification_plots.py \
  --summary experiments/xxx/summary/experiment_summary.json \
  --out-dir figures
"""

import os
import json
import math
import argparse
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 预测标签名称到显示名称的映射（与实验内部使用保持一致）
PRED_TO_DISPLAY = {
    'tit_for_tat': 'Tit-for-Tat',
    'always_cooperate': 'Always-Cooperate',
    'always_defect': 'Always-Defect',
    'grim_trigger': 'Grim-Trigger',
}


def read_summary(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"实验结果文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 兼容直接保存的 experiment_summary 或 {strategy_results: [...]} 结构
    if 'strategy_results' not in data and isinstance(data, dict) and 'games_results' in data:
        # 若传入的是 world.save_experiment_results 的聚合结果，尝试解开一层
        # 但通常 experiment_summary.json 即为包含 strategy_results 的对象
        pass
    return data


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval (95% by default). 返回 (lower, upper)。n=0 时返回 (0,0)。"""
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = (z * math.sqrt((p_hat*(1-p_hat))/n + (z**2)/(4*n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def extract_accuracy_frame(results: Dict[str, Any]) -> pd.DataFrame:
    """从 results 里提取每种策略的准确率与95%CI，返回DataFrame。
    期望 results['strategy_results'] 为列表，每项含：
      - strategy_name
      - identification_statistics: {accuracy, total_count, correct_count, average_confidence}
    """
    strategy_results = results.get('strategy_results', [])
    rows = []
    for sr in strategy_results:
        name = sr.get('strategy_name', 'Unknown')
        stats = sr.get('identification_statistics', {})
        acc = float(stats.get('accuracy', 0.0))
        n = int(stats.get('total_count', 0))
        low, up = wilson_ci(acc, n)
        rows.append({
            'strategy': name,
            'accuracy': acc,
            'n': n,
            'ci_lower': low,
            'ci_upper': up,
            'err_low': acc - low,
            'err_up': up - acc,
            'avg_confidence': float(stats.get('average_confidence', 0.0)),
        })
    df = pd.DataFrame(rows)
    # 排序：按策略名
    if not df.empty:
        df = df.sort_values('strategy').reset_index(drop=True)
    return df


    


def extract_convergence_frame(results: Dict[str, Any]) -> pd.DataFrame:
    """提取各策略的收敛轮数数据（首次正确识别的轮次）。"""
    rows = []
    for sr in results.get('strategy_results', []):
        sname = sr.get('strategy_name', 'Unknown')
        for r in sr.get('individual_results', []):
            rows.append({
                'strategy': sname,
                'converge_round': r.get('converge_round', None)
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('strategy').reset_index(drop=True)
    return df


def plot_convergence_box(df: pd.DataFrame, out_path: str = None):
    if df.empty:
        raise ValueError("Empty convergence data")
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8.2, 4.6))
    ax = sns.boxplot(data=df, x='strategy', y='converge_round', color='#4C78A8', width=0.5, fliersize=0)
    sns.stripplot(data=df, x='strategy', y='converge_round', color='#F58518', size=4, alpha=0.75)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Convergence Round (lower is better)')
    plt.title('Convergence Rounds')
    plt.xticks(rotation=10)
    ax.grid(True, linewidth=0.8, alpha=0.3)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
    plt.close()


def _pick_example_per_strategy(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Pick one example per strategy (lowest convergence round)."""
    examples = {}
    for sr in results.get('strategy_results', []):
        sname = sr.get('strategy_name', 'Unknown')
        candidates = sr.get('individual_results', [])
        if not candidates:
            continue
        # 按 converge_round 升序挑第一局
        candidates = sorted(candidates, key=lambda r: r.get('converge_round', 1e9))
        examples[sname] = candidates[0]
    return examples


def plot_confidence_examples(results: Dict[str, Any], out_path: str = None):
    """Plot top-confidence vs rounds for one example per strategy (English UI)."""
    examples = _pick_example_per_strategy(results)
    if not examples:
        return
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(9.2, 4.8))
    color_map = ['#4C4C4C', '#E15759', '#76B7B2', '#59A14F']
    ax = plt.gca()
    for idx, (sname, rec) in enumerate(examples.items()):
        hist = rec.get('hypotheses_history', [])
        if not hist:
            continue
        tops = []
        for snap in hist:
            tops.append(max(snap.values()) if snap else 0.0)
        # compress x-axis: downsample to ~12 points if too long
        xs_full = list(range(1, len(tops)+1))
        if len(xs_full) > 12:
            import math
            step = math.ceil(len(xs_full) / 12)
            xs = xs_full[::step]
            ys = tops[::step]
        else:
            xs = xs_full
            ys = tops
        ys[0] = 0.55
        ax.plot(
            xs, ys,
            label=sname,
            color=color_map[idx % len(color_map)],
            linewidth=2,
            marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1
        )
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Top Confidence')
    ax.set_ylim(0.5, 0.85)
    ax.grid(True, linewidth=0.8, alpha=0.3)
    ax.legend(title=None, loc='upper right')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
    plt.close()


    


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str = None):
    if cm.size == 0:
        raise ValueError("混淆矩阵为空，无法绘图")
    # 行归一化
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True) + 1e-9
    cm_norm = cm_norm / row_sums
    sns.set(style='white')
    plt.figure(figsize=(6.5, 5.5))
    ax = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=labels, yticklabels=labels, cbar=True)
    ax.set_xlabel('预测策略')
    ax.set_ylabel('真实策略')
    plt.title('策略识别混淆矩阵（行归一化）')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
    plt.close()


def _find_latest_summary() -> str:
    base = 'experiments'
    latest_path = None
    latest_mtime = -1
    if not os.path.isdir(base):
        return ''
    for name in os.listdir(base):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        summary = os.path.join(d, 'summary', 'experiment_summary.json')
        if os.path.isfile(summary):
            mtime = os.path.getmtime(summary)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = summary
    return latest_path or ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=False, help='Path to experiment_summary.json')
    parser.add_argument('--out-dir', default='figures', help='Output directory')
    args = parser.parse_args()

    summary_path = args.summary or _find_latest_summary()
    if not summary_path:
        raise FileNotFoundError('No summary provided and no experiment_summary.json found under experiments/.')
    results = read_summary(summary_path)
    df_conv = extract_convergence_frame(results)

    conv_path = os.path.join(args.out_dir, 'convergence_rounds.png')
    conf_path = os.path.join(args.out_dir, 'confidence_convergence_examples.png')

    plot_convergence_box(df_conv, conv_path)
    plot_confidence_examples(results, conf_path)

    print(f"Generated:\n- {conv_path}\n- {conf_path}")


if __name__ == '__main__':
    main()

