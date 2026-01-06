import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hardcoded data provided by user
# All data starts from x=1
data_source = {
    'Hardline': [55, 70, 72, 78, 82, 80, 80, 88, 85, 80, 90, 95, 96, 100],
    'Tit-for-Tat': [50, 68, 74, 78, 82, 80, 76, 70, 74, 72, 68, 65, 60, 45, 40, 35],
    'Flexible': [50, 65, 70, 75, 75, 72, 74, 60, 63, 47, 45, 40, 35],
    'Historical': [50, 60, 65, 62, 58, 55, 50, 45, 40, 38, 35, 33],
    'Concession': [50, 60, 64, 60, 50, 45, 40, 35, 20]
}

# Prepare data for DataFrame
plot_data = []
for persona, values in data_source.items():
    for i, val in enumerate(values):
        plot_data.append({
            'Round': i + 1,
            'Prediction_Accuracy': val / 100.0, # Convert to 0-1 scale
            'Target_Persona': persona
        })

df = pd.DataFrame(plot_data)

# Plotting
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Define colors mapping using 'tab:' prefix for nicer shades (Matplotlib/Seaborn defaults)
palette = {
    'Hardline': 'tab:red',
    'Tit-for-Tat': 'gold',
    'Flexible': 'tab:blue',
    'Historical': 'black',
    'Concession': 'tab:green'
}

# Plot lines
markers = {p: 'o' for p in palette}
ax = sns.lineplot(data=df, x='Round', y='Prediction_Accuracy', hue='Target_Persona',
             style='Target_Persona', markers=markers, dashes=False, linewidth=2, palette=palette)

# Customize markers to be white with colored borders
for line in ax.lines:
    line.set_markerfacecolor('white')
    line.set_markeredgecolor(line.get_color())
    line.set_markeredgewidth(1.5)
    line.set_markersize(4)

plt.xlabel('Action Sequence', fontsize=14, fontweight='bold')
plt.ylabel('World Situation', fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9), prop={'weight': 'bold', 'size': 12})
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot to a file
plt.savefig('rq2_evolution_chart_history.png')
plt.show()