import numpy as np
import random
import os
import sys


# all_orders_list 主要实现函数
def all_orders_list(step_len, repeat_len,  mer_list, rest_list, n:float, rand=True):
    """_summary_

    Args:
        step_len (_type_): 模拟的订单生成的时间步数
        repeat_len (_type_): 订单规律的作用步数长度
        mer_list (_type_): 商家位置坐标
        rest_list (_type_): 目的地位置坐标
        n(float):系数调整订单数量
        rand (bool, optional): 是否随机生成. Defaults to True.

    Returns:
        _type_: 返回订单的数组
    """
    # 目前生成的订单数量是针对两家企业产生的，如果订单的数量觉得不够的话可以修改n的大小，让订单成倍增长
    if not rand:
        random.seed(0)
    all_list = []
    for i in range(1, step_len):
        i = i % repeat_len  # 调节波峰出现速率 | 取余，每repeat_len循环一遍
        order_num = int(n * int(fitting_dist(i)))
        orders = orders_list(order_num, mer_list, rest_list)

        for order in orders:
            order.append(i)
        all_list.append(orders)

    return all_list


# 订单产生的函数
def fitting_dist(x):
    a = [314.2, 188.3, 95.56, 22.9, 48.67]
    b = [172.5, 281.5, 315.5, 228.9, 267.1]
    c = [4.645, 1.559, 10.69, 167.7, 13.1]
    fitting_model = 0
    for i in range(0, 5):
        fitting_model = fitting_model + a[i] * np.exp(-((x - b[i]) / c[i]) ** 2)
    return fitting_model


# 订单类型函数，订单的类型有A, B, C, A正常，B高峰期，C配送费用高
def get_order_type():
    weight = {"A": 0.2, "B": 0.4, "C": 0.4}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


# 订单难度，1/2/3
def order_difficulty():
    return random.randint(1, 3)


# 订单是否支持合作,0/1
def order_cooperation():
    return random.randint(0, 1)


# 订单金额
def order_money(order_type):
    print(order_type, "order_type------------")
    if order_type == 'A':
        return random.uniform(3, 4)
    elif order_type == 'B':
        return random.uniform(4, 8)
    elif order_type == 'C':
        return random.uniform(10, 12)


# 订单处理成本
def order_process():
    return random.randint(20, 50)


# !!!修改：订单位置-商家
def merchant_position(merchant_list):
    random_node = random.choice(merchant_list)
    return random_node


#  !!!修改：订单位置-目的地
def rest_position(rest_list):
    random_node = random.choice(rest_list)
    return random_node



# 生成单日订单序列,单日订单序列是由列表组成，列表的每一个元素为一个订单list，
# 例：[['AC', 76, 17, 2, pos], ['C', 30, 17, 1,pos], ['AB', 48, 17, 2, pos]]
def orders_list(order_num, mer_pos, rest_pos):
    daily_order = []
    print(mer_pos, rest_pos)
    for i in range(0, order_num):
        daily_order.append([get_order_type(),
                            order_money(get_order_type()),
                            merchant_position(mer_pos),
                            rest_position(rest_pos),
                            order_difficulty()])

    return daily_order


if __name__ == "__main__":
    orders = all_orders_list(360, 120, [1,2,3,4], [1,2,3,4], 0.05)
    print(orders)
