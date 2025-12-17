import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    latest_folder = find_latest_experiment_folder(project_root)

    rq3_df = pd.read_csv(os.path.join(latest_folder, 'RQ3_Performance.csv'))

    win_counts = rq3_df['Winner_Architecture'].value_counts()
    total_games = len(rq3_df)
    win_rates = (win_counts / total_games * 100).reset_index()
    win_rates.columns = ['Architecture', 'Win_Rate']

    order = ['MAGES', 'ReAct', 'Reflexion', 'EvoAgent']
    win_rates['Architecture'] = pd.Categorical(win_rates['Architecture'], categories=order, ordered=True)
    win_rates = win_rates.sort_values('Architecture')

    sns.set(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = ['#d62728' if x == 'MAGES' else '#95a5a6' for x in win_rates['Architecture']]
    bp = sns.barplot(ax=ax, data=win_rates, x='Architecture', y='Win_Rate', palette=colors)

    for p in bp.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 1,
            '{:.1f}%'.format(height),
            ha="center",
            fontsize=12,
            fontweight='bold',
        )

    ax.set_title('(b) Strategic Outcome: Tournament Win Rates', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Agent Architecture', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('rq3_win_rates.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
