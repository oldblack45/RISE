import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataframe
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'RQ2_Evolution.csv')
df = pd.read_csv(csv_path)

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
             color='red', linewidth=3, label='RISE (Global Average)')

# Plot individual Personas with thinner lines / different styles
sns.lineplot(data=persona_acc, x='Round', y='Prediction_Accuracy', hue='Target_Persona',
             style='Target_Persona', markers=True, dashes=False, linewidth=1.5, alpha=0.7)

plt.xlabel('Game Rounds', fontsize=14, fontweight='bold')
plt.ylabel('Prediction Accuracy', fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 12})
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot to a file
plt.savefig('rq2_evolution_chart.png')
plt.show()

# Display the first few rows of the aggregated data to help with writing
print(global_acc.head())
print(global_acc.tail())
print(persona_acc.groupby('Target_Persona')['Prediction_Accuracy'].mean())