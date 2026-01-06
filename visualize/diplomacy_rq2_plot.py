import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ========== 配置区域 ==========
# 指定实验文件夹名称，例如：'diplomacy_tournament_20251220_214450'
# 如果设置为 None，则自动选择最新的文件夹
EXPERIMENT_FOLDER = 'diplomacy_tournament_20251217_000945'
# =============================


def find_latest_experiment_folder(project_root: str) -> str:
    experiments_dir = os.path.join(project_root, 'experiments')
    search_pattern = os.path.join(experiments_dir, 'diplomacy_tournament_*')
    folders = glob.glob(search_pattern)

    if not folders:
        raise FileNotFoundError(f"No experiment folders found matching {search_pattern}")

    latest_folder = max(folders)
    print(f"Automatically selected latest experiment: {latest_folder}")
    return latest_folder


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    if EXPERIMENT_FOLDER:
        # 用户指定了文件夹名称
        latest_folder = os.path.join(project_root, 'experiments', EXPERIMENT_FOLDER)
        if not os.path.exists(latest_folder):
            raise FileNotFoundError(f"指定的文件夹不存在: {latest_folder}")
        print(f"使用指定的实验文件夹: {latest_folder}")
    else:
        # 自动选择最新的文件夹
        latest_folder = find_latest_experiment_folder(project_root)

    df = pd.read_csv(os.path.join(latest_folder, 'RQ2_Evolution.csv'))
    # 2. 设置学术风格 (Seaborn + Matplotlib)
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "serif" # 使用衬线字体 (Times New Roman风格)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. 绘制随机猜测基准线 (Random Guess Baseline)
    ax.axhline(y=0.33, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='Random Baseline (33%)')

    # 4. 绘制不同对手的细线 (Fade out)
    # 这一步是为了展示 Robustness，但颜色要淡，不要抢戏
    personas = df['Target_Persona'].unique()
    palette = sns.color_palette("husl", len(personas))

    for i, persona in enumerate(personas):
        subset = df[df['Target_Persona'] == persona]
        # 计算每轮的平均值（因为可能有多个 GameID）
        mean_line = subset.groupby('Round')['Prediction_Accuracy'].mean()
        ax.plot(mean_line.index, mean_line.values, 
                color=palette[i], alpha=0.3, linewidth=1, linestyle='-', label=f'vs {persona}')

    # 5. 绘制全局平均线 (The Hero Line) - 带置信区间
    # Seaborn 的 lineplot 默认会画出 95% 置信区间 (阴影部分)
    sns.lineplot(data=df, x='Round', y='Prediction_Accuracy', 
                color='#C0392B', linewidth=3, errorbar=('ci', 95), label='RISE (Global Avg.)', ax=ax)

    # 6. 添加阶段标注 (Phase Annotation) - 最关键的叙事部分
    # Phase 1: Exploration
    ax.axvspan(1, 4, color='gray', alpha=0.1)
    ax.text(2.5, 0.95, 'Exploration', ha='center', va='top', fontsize=12, fontweight='bold', color='#555')

    # Phase 2: Rapid Learning
    ax.axvspan(4, 12, color='#F39C12', alpha=0.1)
    ax.text(8, 0.95, 'Rapid Learning', ha='center', va='top', fontsize=12, fontweight='bold', color='#D35400')

    # Phase 3: Convergence
    ax.axvspan(12, 20, color='#27AE60', alpha=0.1)
    ax.text(16, 0.95, 'Convergence', ha='center', va='top', fontsize=12, fontweight='bold', color='#1E8449')

    # 7. 美化坐标轴与图例
    ax.set_xlabel('Game Rounds', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, 20)

    # 优化图例：把 RISE 放在最前面
    handles, labels = ax.get_legend_handles_labels()
    # 重排图例顺序，让 Global Avg 在第一位
    order = [-1] + list(range(len(labels)-2)) + [-2] # Adjust index based on labels
    # 简单起见，直接重新定位
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, shadow=True)

    # 去掉上方和右侧的边框 (Spines)
    sns.despine()

    plt.tight_layout()
    plt.savefig('rq2_academic_style.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()