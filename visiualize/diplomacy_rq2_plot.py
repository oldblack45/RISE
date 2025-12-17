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

    df = pd.read_csv(os.path.join(latest_folder, 'RQ2_Evolution.csv'))
    # Calculate average accuracy per round (Global)
    global_acc = df.groupby('Round')['Prediction_Accuracy'].mean().reset_index()
    global_acc['Target_Persona'] = 'Global Average'

    # Calculate average accuracy per round per Persona
    persona_acc = df.groupby(['Round', 'Target_Persona'])['Prediction_Accuracy'].mean().reset_index()

    # Combine for plotting if needed, or just plot separately
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot Global Average with a thicker line
    sns.lineplot(data=global_acc, x='Round', y='Prediction_Accuracy', 
                color='red', linewidth=3, label='Global Average')

    # Plot individual Personas with thinner lines / different styles
    sns.lineplot(data=persona_acc, x='Round', y='Prediction_Accuracy', hue='Target_Persona', 
                style='Target_Persona', markers=True, dashes=False, linewidth=1.5, alpha=0.7)

    plt.title('Cognitive Evolution: Prediction Accuracy over Time', fontsize=14)
    plt.xlabel('Game Rounds', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title='Opponent Architecture')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig('rq2_evolution_chart.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()