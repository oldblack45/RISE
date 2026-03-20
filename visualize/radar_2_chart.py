import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import pi

# ==========================================
# 1. Raw Data
# ==========================================
# Macro: Diplomacy  (Win%, Surv%, SCs)
dip_raw = {
    'Full_Model':      [0.7,  1.0,  5.6],
    'w/o_WorldModel':  [0.2,  1.0,  3.5],
    'w/o_Reasoning':   [0.3,  0.8,  4.9],
    'w/o_Utility':     [0.6,  0.5,  4.8],
    'w/o_All':         [0.1,  0.3,  2.5],
}

# Micro: Delivery  (Prof, P/D, Ful%)
del_raw = {
    'Full_Model':      [315.4, 65.5, 98.2],
    'w/o_WorldModel':  [250.2, 52.1, 88.5],
    'w/o_Reasoning':   [275.6, 48.3, 85.0],
    'w/o_Utility':     [290.1, 58.4, 60.5],
    'w/o_All':         [210.5, 38.0, 68.5],
}

# ==========================================
# 2. Normalization
# ==========================================
dip_base = np.array(dip_raw['Full_Model'])
dip_norm = {}
for k, v in dip_raw.items():
    dip_norm[k] = (np.array(v) / (dip_base + 1e-9)).tolist()

del_base = np.array(del_raw['Full_Model'])
del_norm = {}
for k, v in del_raw.items():
    del_norm[k] = (np.array(v) / (del_base + 1e-9)).tolist()

# ==========================================
# 3. Plotting Configuration
# ==========================================
plot_mapping = {
    'w/o World Model':  ('w/o_WorldModel', 'w/o_WorldModel'),
    'w/o Reasoning':    ('w/o_Reasoning',  'w/o_Reasoning'),
    'w/o Utility':      ('w/o_Utility',    'w/o_Utility'),
    'w/o All': ('w/o_All',     'w/o_All'),
}

colors = ['#3498DB', '#F1C40F', '#9B59B6', '#34495E']
metrics_dip = ['Succ', 'Surv', 'Res']
metrics_del = ['Prof', 'P/D', 'Ful']
categories = metrics_dip + metrics_del
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Set global font size
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': 'polar'})
plt.subplots_adjust(wspace=0.35, hspace=0.45)  # Increase hspace for titles


def create_hex_radar(ax, title, dip_key, del_key, color):
    # Rotate 240 degrees (4*pi/3) so Left=Dip, Right=Del
    ax.set_theta_offset(4 * pi / 3)
    ax.set_theta_direction(-1)

    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_yticklabels([])

    c_data = dip_norm[dip_key]
    d_data = del_norm[del_key]
    max_val = max(max(c_data), max(d_data))

    ticks = [0.25, 0.50, 0.75, 1.00]
    if max_val > 1.05:
        ticks.append(1.25)
        if max_val > 1.3:
            ticks.append(1.50)

    # Draw Hexagonal Grid
    for t in ticks:
        ax.plot(angles, [t] * (N + 1), color='grey', linewidth=1.0, linestyle=':', alpha=0.6)

    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, ticks[-1]], color='grey', linewidth=1.0, alpha=0.6)

    ax.plot(angles, [ticks[-1]] * (N + 1), color='black', linewidth=1.5)

    # Left Zone (Diplomacy - Macro)
    ax.fill_between(np.linspace(-pi / 6, 5 * pi / 6, 100), 0, ticks[-1] * 1.1, color='#3498DB', alpha=0.1)

    # Right Zone (Delivery - Micro)
    ax.fill_between(np.linspace(5 * pi / 6, 11 * pi / 6, 100), 0, ticks[-1] * 1.1, color='#E74C3C', alpha=0.1)

    # Plot Full Model
    ax.plot(angles, [1.0] * (N + 1), color='gray', linewidth=2.0, linestyle='--', alpha=0.6, label='Full Model')

    # Plot Variant
    values = c_data + d_data
    values += values[:1]

    ax.plot(angles, values, color=color, linewidth=3.5, marker='o', markersize=6)  # Thicker line
    ax.fill(angles, values, color=color, alpha=0.25)

    # Axis Labels - BOLD and LARGER
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=16, weight='bold')
    ax.tick_params(pad=8)  # Move labels out further

    # Percentage Labels
    for t in ticks:
        if t <= 1.5:
            # Place on 'Win' axis (Index 3)
            ax.text(angles[3], t, f"{int(t * 100)}%", ha='center', va='bottom', fontsize=16, fontweight='bold',
                    color='gray')

    ax.set_title(title, size=20, weight='bold', y=1.12, color='#333')
    ax.set_ylim(0, ticks[-1])


for i, (title, (dip_key, del_key)) in enumerate(plot_mapping.items()):
    ax = axes.flat[i]
    create_hex_radar(ax, title, dip_key, del_key, colors[i])

# ==========================================
# 4. Improved Legend
# ==========================================
# Create custom legend handles
blue_patch = mpatches.Patch(color='#3498DB', alpha=0.2, label='Macro: Diplomacy')
red_patch = mpatches.Patch(color='#E74C3C', alpha=0.2, label='Micro: Delivery')
# gray_line = plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Full RISE Model')
# variant_line = plt.Line2D([0], [0], color='black', linewidth=3, marker='o',
#                           label='Ablated Variant')  # Generic black for legend

# Place legend at the top or bottom, centered
legend = fig.legend(handles=[blue_patch, red_patch],
           loc='lower center', ncol=4, fontsize=20, frameon=False, bbox_to_anchor=(0.5, 0.02))
for text in legend.get_texts():
    text.set_fontweight('bold')

# plt.suptitle("Figure 7: Hexagonal Ablation Analysis (Bold & Enhanced)", fontsize=18, weight='bold', y=0.98)

plt.savefig('rq4_hex_radar_bold.png', dpi=300, bbox_inches='tight')
plt.show()