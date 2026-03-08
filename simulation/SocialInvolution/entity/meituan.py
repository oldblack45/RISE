import random
import csv
import datetime
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from models.algorithm.utilityCalMethods.sys_cal import SysCal
from repast4py.space import DiscretePoint as dpt
from simulation.SocialInvolution.algorithm.generate_orders import all_orders_list
from simulation.SocialInvolution.entity.user import User
from simulation.SocialInvolution.entity.order import Order
from simulation.SocialInvolution.entity.merchant import Merchant


class Platform:
# 实现功能 1.订单生成（一天订单生成 每个step订单生成） 2. 订单池子存储（更新池子 返回池子信息） 3. 骑手排行榜（排行榜信息返回 排行榜信息更新）
    def __init__(self, order_BF: int, riders, start_time):
        """
            order_BF: 订单波峰设置
        """
        self.riders = riders
        self.order_BF = order_BF
        self.order_id = [0]  # 记录生成的所有的订单数目
        self.all_normal_orders_info = [] # 存储所有普通订单信息
        self.normal_orders_dict = {}
        self.now_orders_info = {}

        self.dis_day_rank = []
        self.money_day_rank = []
        self.order_day_rank = []

        self.target = SysCal(len(self.riders))
        self.start_time = start_time
        self.init_info()
    
    def update_target(self):
        """
        更新效能 多样性 熵增速率 公平性四个指标
        """
        self.target.update_fairness(list(self.riders.values()))
        self.target.update_variety(list(self.riders.values()))
        self.target.update_entropy_increase(list(self.riders.values()))
        self.target.update_utility()


    def init_info(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = f"agent_log/{self.start_time}"
        folder_path = os.path.join(script_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = f"finish_order_record.csv"
        filepath = os.path.join(folder_path, path)

        with open(filepath, mode='w', newline='') as file:
            file.seek(0)
            file.truncate()
            writer = csv.writer(file)
            writer.writerow(["order_id", "order_money", "rider", "finish_time"])

        path = f"platform_record.csv"
        filepath = os.path.join(folder_path, path)

        with open(filepath, mode='w', newline='') as file:
            file.seek(0)
            file.truncate()
            writer = csv.writer(file)
            writer.writerow(["runner_step", "before_order_num", "after_order_num", "profit", "utility", "fairness", "variety", "entropy"])


    def update_info_platform(self, runner_step, order_num):
        #["runner_step", "before_order_num", "after_order_num", "profit", "utility", "fairness", "variety", "entropy"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = f"agent_log/{self.start_time}"
        folder_path = os.path.join(script_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = f"platform_record.csv"
        filepath = os.path.join(folder_path, path)

        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([runner_step, order_num, len(self.now_orders_info), self.target.profit, self.target.utility, self.target.fairness, self.target.variety, self.target.entropy_increase])


    def info_log(self, order, rider_id, runner_step):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = f"agent_log/{self.start_time}"
        folder_path = os.path.join(script_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = f"finish_order_record.csv"
        filepath = os.path.join(folder_path, path)

        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([order.id, order.money, rider_id, runner_step])


    # 订单生成
    def generate_order_num_all_step(self, step_len, repeat_len,  mer_list, rest_list, n:float):
        """ 生成每个时间点生成订单的信息，再调用平台生成订单的方式

        Args:
            runner_step (_type_): 当前时间点
        """
        self.all_normal_orders_info = all_orders_list(step_len, repeat_len,  mer_list, rest_list, n)
        cnt = 0
        for order in self.all_normal_orders_info:
                cnt += len(order)
        print(cnt)
        print('test', self.all_normal_orders_info)
        

    def generate_order_now_step(self, runner_step):
        """
            普通订单生成平台调度
        """
        if runner_step >= len(self.all_normal_orders_info):
            return
        for order in self.all_normal_orders_info[runner_step]:
            new_order = Order(self.order_id[0], self.order_id[0], order[2], order[3],
                              (runner_step, runner_step + 15), (runner_step, runner_step + 15), order[1], runner_step)
            self.order_id[0] += 1
            self.normal_orders_dict[new_order.id_num] = new_order
            self.now_orders_info[new_order.id_num] = {
                "order_id": new_order.id_num,
                "pickup_location": new_order.pickup_location,
                "delivery_location":new_order.delivery_location,
                "money": round(new_order.money,2)
            }


    # 2. 订单池子
    def choose_order_update(self, choose_list: list):
        """
            骑手选择订单，平台更新订单信息
        """

        for i in choose_list:
            try:
                self.target.update_profit(self.normal_orders_dict[i].platform_money) # 更新平台的收益
                try:
                    del self.now_orders_info[i]
                except KeyError:
                    print(f"Order ID {i} not found in the dictionary.")
            except Exception as e:
                print('删除订单错误')
                print(e)


    def check_orders_rider(self):
        """
            订单信息返回
        """
        return self.now_orders_info
            

    # 3. 骑手排行榜
    def return_rank(self):
        """
            返回排名（ 一天收益 一天移动距离 一天的订单数量）
        """
        return self.money_day_rank, self.dis_day_rank, self.order_day_rank


    def update_rank(self):
        """
            更新骑手的排行榜, 新的一天更新订单信息
        """
        self.dis_day_rank.clear()
        self.order_day_rank.clear()
        self.money_day_rank.clear()
        for rider in self.riders.values():
            self.dis_day_rank.append((rider.dis_day,rider))
            self.money_day_rank.append((rider.money_day,rider))
            self.order_day_rank.append((rider.order_day,rider))
        self.dis_day_rank.sort(key=lambda i: (i[0]),reverse=True)
        self.money_day_rank.sort(key=lambda i: (i[0]),reverse=True)
        self.order_day_rank.sort(key=lambda i: (i[0]),reverse=True)
        for i in range(len(self.dis_day_rank)):
            self.dis_day_rank[i][1].rank_day['dis_rank'] = i + 1
            self.money_day_rank[i][1].rank_day['money_rank'] = i + 1
            self.order_day_rank[i][1].rank_day['order_rank'] = i + 1
        for rider in self.riders.values():
            rider.dis_day = 0
            rider.money_day = 0
            rider.order_day = 0

            
    def get_order(self, order_id):
        return self.normal_orders_dict[order_id]


    def del_orders(self, orders, rider_id, runner_step):
        # 骑手完成订单时候调用这个函数以清除order
        for order_id, order in orders:
            try:
                self.info_log(self.normal_orders_dict[order_id], rider_id, runner_step)
                del self.normal_orders_dict[order_id]    
            except KeyError:
                print(f"Order ID {order_id} not found in the dictionary.")      

