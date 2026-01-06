import plotly.graph_objects as go

# 定义节点
actions = ["公开声明", "外交谈判", "军事演习", "区域封锁", "武器部署"]
feedbacks = ["局势缓和", "紧张升级", "冲突爆发"]

labels = actions + feedbacks

# 构建流向数据 (action -> feedback)
# 这里的数值是假设的频次/概率
source = [
    0, 0,   # 公开声明 -> 局势缓和/紧张升级
    1, 1,   # 外交谈判 -> 局势缓和/紧张升级
    2, 2,   # 军事演习 -> 紧张升级/冲突爆发
    3, 3,   # 区域封锁 -> 紧张升级/冲突爆发
    4, 4    # 武器部署 -> 紧张升级/冲突爆发
]

target = [
    5, 6,   # 公开声明 -> 局势缓和/紧张升级
    5, 6,   # 外交谈判 -> 局势缓和/紧张升级
    6, 7,   # 军事演习 -> 紧张升级/冲突爆发
    6, 7,   # 区域封锁 -> 紧张升级/冲突爆发
    6, 7    # 武器部署 -> 紧张升级/冲突爆发
]

values = [
    10, 5,   # 公开声明
    15, 5,   # 外交谈判
    8, 6,    # 军事演习
    4, 10,   # 区域封锁
    3, 12    # 武器部署
]

# 绘制桑基图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=["#4CAF50","#2196F3","#FFC107","#FF5722","#9C27B0",
               "#8BC34A","#FF9800","#F44336"]
    ),
    link=dict(
        source=source,
        target=target,
        value=values,
        color=["rgba(76,175,80,0.4)", "rgba(76,175,80,0.4)",
               "rgba(33,150,243,0.4)", "rgba(33,150,243,0.4)",
               "rgba(255,193,7,0.4)", "rgba(255,193,7,0.4)",
               "rgba(255,87,34,0.4)", "rgba(255,87,34,0.4)",
               "rgba(156,39,176,0.4)", "rgba(156,39,176,0.4)"]
    )
)])

fig.update_layout(title_text="行动选择 → 世界反馈 桑基图", font_size=14)
fig.show()
