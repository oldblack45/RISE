import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义模型名称和指标 - 删除 ReAct，添加 WarAgent
models = ['Werewolf', 'CoT', 'WarAgent', 'Ours']
metrics = ['EA', 'AS', 'SR', 'OM', 'FS']

# 生成示例数据 (二维数组：模型 × 指标) - 更新为 WarAgent 数据
data = np.array([
    [0.17, 0.24, 0.47, 0.71, 0.47],  # Werewolf
    [0.10, 0.18, 0.48, 0.73, 0.47],  # CoT
    [0.20, 0.28, 0.52, 0.68, 0.50],  # WarAgent (宏观历史模拟)
    [0.35, 0.37, 0.54, 0.72, 0.54],  # Ours (RISE)
])

# 设置颜色（与原图一致）
colors = ['#fce6dd', '#728597', '#9be0fe', '#ff8f95']  # 浅橙、灰、浅蓝、浅红

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(13, 3))

# 设置柱状图的位置
x = np.arange(len(metrics))
width = 0.18  # 柱子的宽度
gap = 0.03    # 相邻柱子之间的缝隙
step = width + gap
multiplier = 0

# 绘制柱状图（无边框）
for i, (model, color) in enumerate(zip(models, colors)):
    offset = step * multiplier
    rects = ax.bar(
        x + offset, data[i], width, label=model, color=color, alpha=0.9
    )
    # 数值标签
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    multiplier += 1

# 生成垂直紧凑的分离图例（样式与原图一致）
legend_handles = [Patch(facecolor=colors[i], edgecolor='none', label=models[i]) for i in range(len(models))]
fig_legend = plt.figure(figsize=(1.6, 2.2))
legend = fig_legend.legend(handles=legend_handles, loc='center', ncol=1, frameon=True,
                  borderpad=0.3, handlelength=1.0, handletextpad=0.4, labelspacing=0.3, fancybox=False)
# legend.get_frame().set_edgecolor('black')
# legend.get_frame().set_linewidth(1.2)
fig_legend.tight_layout()
fig_legend.savefig('bar_legend.png', dpi=300, bbox_inches='tight')

# 设置图表属性（仅加粗纵坐标与指标）
# ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x + step * (len(models) - 1) / 2)
ax.set_xticklabels(metrics)
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
ax.set_ylim(0, 1.0)

# 收紧左右留白：设置到柱边缘略微外扩
left_edge = x[0] - width/2 - 0.02
right_edge = x[-1] + step * (len(models) - 1) + width/2 + 0.02
ax.set_xlim(left_edge, right_edge)
ax.margins(x=0)

# 不在主图显示图例
# ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

# 去掉边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 设置网格
ax.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print("柱状图已保存为 'model_comparison_chart.png'，分离图例已保存为 'bar_legend.png'")

# 显示图表
plt.show()

# 打印数据表格
print("\n数据表格:")
print("模型\t\t" + "\t".join(metrics))
print("-" * 50)
for i, model in enumerate(models):
    values_str = "\t".join([f"{val:.2f}" for val in data[i]])
    print(f"{model}\t\t{values_str}")
