import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 定义指标名称
metrics = ['AS', 'SR', 'OM', 'EA']
num_metrics = len(metrics)

# 四种消融及完整模型数据
ablation_data = {
    'w/o world model': [0.63, 1, 0.75, 0.73],
    'w/o agent profiling': [0.64, 0.75, 0.5, 0.51],
    'w/o reasoning': [0.58, 0.75, 0.5, 0.49],
    'w/o all': [0.51, 0.5, 0.4, 0.45],
    'RISE': [0.65, 1.00, 1.00, 0.81]
}

# 定义颜色（完整模型浅色，消融模型深色）
colors = {
    'w/o world model': '#FF7F0E',    # 深橙色
    'w/o agent profiling': '#1F77B4',  # 深蓝色
    'w/o reasoning': '#2CA02C',        # 深绿色
    'w/o all': '#9467BD',            # 深紫色
    'RISE': '#FFD1DC'     # 浅粉色
}

# 对比组合及标题（全部小写，与数据一致）
comparisons = [
    ('w/o world model', 'w/o world model'),
    ('w/o agent profiling', 'w/o agent profiling'),
    ('w/o reasoning', 'w/o reasoning'),
    ('w/o all', 'w/o all')
]

# 雷达图参数
angles = (np.linspace(0, 2 * np.pi, num_metrics, endpoint=False) + np.pi/2).tolist()
ring_values = [0.2, 0.4, 0.6, 0.8, 1.0]
max_radius = 1.0
label_distance = 1.1
limit_xy = 1.2

# 绘制雷达网格
def draw_radar_grid(ax, angles, ring_values, max_radius):
    for radius in ring_values:
        x_points = []
        y_points = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            x_points.append(x)
            y_points.append(y)
        x_points.append(x_points[0])
        y_points.append(y_points[0])
        ax.plot(x_points, y_points, '--', color='grey', alpha=0.3, linewidth=0.8)
    for angle in angles:
        x_end = max_radius * np.cos(angle)
        y_end = max_radius * np.sin(angle)
        ax.plot([0, x_end], [0, y_end], '-', color='grey', alpha=0.3, linewidth=0.8)

# 绘制雷达数据
def draw_radar_data(ax, model1_name, model1_data, model2_name, model2_data, angles, colors):
    for model_name, scores in [(model1_name, model1_data), (model2_name, model2_data)]:
        x_points = []
        y_points = []
        for i, score in enumerate(scores):
            angle = angles[i]
            x = score * np.cos(angle)
            y = score * np.sin(angle)
            x_points.append(x)
            y_points.append(y)
        x_points.append(x_points[0])
        y_points.append(y_points[0])
        linewidth = 3 if model_name == 'RISE' else 2
        alpha = 0.9 if model_name == 'RISE' else 0.8
        ax.plot(x_points, y_points, 'o-', linewidth=linewidth, markersize=6,
                color=colors[model_name], label=model_name, alpha=alpha)
        ax.fill(x_points, y_points, alpha=0.15, color=colors[model_name])

# 设置坐标轴和标签
def setup_radar_axes(ax, title, metrics, angles, ring_values, label_distance, max_radius, limit_xy):
    for i, metric in enumerate(metrics):
        angle = angles[i]
        x_label = label_distance * np.cos(angle)
        y_label = label_distance * np.sin(angle)
        ax.text(x_label, y_label, metric, ha='center', va='center', fontsize=12, fontweight='bold')
    for r in ring_values:
        ax.text(0, r + max_radius * 0.05, f"{r:.2f}", ha='center', va='center',
                fontsize=10, color='grey', alpha=0.8, fontweight='bold')
    ax.set_xlim(-limit_xy, limit_xy)
    ax.set_ylim(-limit_xy, limit_xy)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, (ablation_model, title) in enumerate(comparisons):
    ax = axes[i]
    draw_radar_grid(ax, angles, ring_values, max_radius)
    draw_radar_data(ax, ablation_model, ablation_data[ablation_model],
                   'RISE', ablation_data['RISE'], angles, colors)
    setup_radar_axes(ax, '', metrics, angles, ring_values, label_distance, max_radius, limit_xy)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
                      frameon=True, fancybox=True, shadow=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontweight('bold')

plt.tight_layout(pad=2.0)
plt.savefig('radar_chart_ablation.png', dpi=300, bbox_inches='tight')
print("消融研究雷达图已保存为 'radar_chart_ablation.png'")
plt.show()

# 打印数据表格
print("\n消融研究数据表格:")
print("模型\t\t\t" + "\t".join(metrics))
print("-" * 60)
for model_name, scores in ablation_data.items():
    values_str = "\t".join([f"{val:.2f}" for val in scores])
    print(f"{model_name}\t\t{values_str}")
