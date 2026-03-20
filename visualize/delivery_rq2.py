import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件
df = pd.read_csv('delivery_cognitive_curve.csv')

# 2. 创建画布
plt.figure(figsize=(10, 6))

# 3. 绘制各基线的折线图 (自定义颜色、线型和标记以符合学术规范)
plt.plot(df['Day'], df['Ours'], 
         marker='o', color='#d62728', linewidth=2, markersize=5, label='Ours')

plt.plot(df['Day'], df['LATS'], 
         marker='s', color='#1f77b4', linewidth=1.5, markersize=4, label='LATS')

plt.plot(df['Day'], df['ReAct'], 
         marker='^', color='#ff7f0e', linewidth=1.5, markersize=4, label='ReAct')

plt.plot(df['Day'], df['Hypothetical Minds'], 
         marker='d', color='#2ca02c', linewidth=1.5, markersize=4, label='Hypothetical Minds')

plt.plot(df['Day'], df['Greedy Heuristic'], 
         marker='x', color='#7f7f7f', linestyle='--', linewidth=1.5, markersize=4, label='Greedy Heuristic')

# 4. 设置图表标题和坐标轴标签
plt.xlabel('Simulation Day', fontsize=18, fontweight='bold')
plt.ylabel('Profit per Distance', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16,fontweight='bold')
plt.yticks(fontsize=16,fontweight='bold')

# 5. 去掉边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# 6. 设置图例和网格
plt.legend(fontsize=16, loc='lower right',prop={'weight': 'bold', 'size': 16}) # 您可以根据需要调整 legend 的位置
plt.grid(True, linestyle='--', alpha=0.6)

# 6. 自动调整布局并保存为高清图片
plt.tight_layout()
plt.show()
plt.savefig('delivery_curve.png', dpi=300)
