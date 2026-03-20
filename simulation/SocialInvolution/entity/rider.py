import csv
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))

from models.algorithm.utilityCalMethods.individual_cal import IndividualCal
from models.agents.repast_agent import Agent
from models.agents.SociologyAgent import (
    RiderGreedyHeuristic,
    RiderHypotheticalMinds,
    RiderLATSAgent,
    RiderLLMAgent,
    RiderReActAgent,
)
from models.env.map.city_with_islands.read_map import AStarPlanner
from simulation.SocialInvolution.algorithm.order_sequence import order_sequence_cal
from repast4py.space import DiscretePoint as dpt

class Rider(Agent,RiderLLMAgent):
    TYPE = 0

    def __init__(self, id, rank, t, pt, max_orders, one_day, step_move_distance=30, role_param_dict = None, start_time=None):
        super().__init__(id, rank, t, pt)
        RiderLLMAgent.__init__(self,role_param_dict)
        self._bind_decision_baseline(role_param_dict)

        # RiderLLMAgent.__init__(self)
        self.route = [] # 派单路线
        self.will_move_positions = [] # 存储所有的移动路径

        self.finish_orders_time = {} # 完成订单时间存储字典 key:step
        self.order_count = 0 # 当前手中订单数目
        self.total_order = 0
        self.max_orders = max_orders # 可以完成的最大订单数目
        self.move_step = step_move_distance #一个step移动的距离
        self.labor = 0 # 骑手已经移动的距离成本
        self.dis = 0
        self.location = (self.pt.x, self.pt.y)  # 已经规划好的位置
        self.no_choose_order_step = 0
        self.money = 0

        self.target = IndividualCal() # 需要计算的指标

        self.go_work_time = 8
        self.get_off_work_time = 18

        # day
        self.rank_day = {
            'order_rank': 0,
            'dis_rank': 0,
            'money_rank': 0,
        }
        self.order_day = 0
        self.dis_day = 0
        self.money_day = 0
        # 设计属性
        self.one_day = one_day
        self.rider_num = 100
        self.choose_order_step_interval = 15
        self.start_time = start_time
        self.write_info_init()

    def _bind_decision_baseline(self, role_param_dict):
        """根据 role_param_dict['baseline_type'] 绑定决策实现。"""
        cfg = role_param_dict or {}
        baseline = str(cfg.get("baseline_type", "rise")).strip().lower()
        baseline_map = {
            "rise": RiderLLMAgent,
            "react": RiderReActAgent,
            "lats": RiderLATSAgent,
            "hypothetical_minds": RiderHypotheticalMinds,
            "hm": RiderHypotheticalMinds,
            "greedy": RiderGreedyHeuristic,
        }
        mixin_cls = baseline_map.get(baseline, RiderLLMAgent)
        if mixin_cls is RiderLLMAgent:
            return

        mixin_cls.__init__(self, cfg)
        self.decide_time = mixin_cls.decide_time.__get__(self, Rider)
        self.take_order = mixin_cls.take_order.__get__(self, Rider)


    def step(self, meituan, runner_step):
        # 每天开始决定今日工作时间
        self.decide_work_time(runner_step)
        self.orders_status_update(runner_step, meituan)
        self.route_to_walk(meituan, runner_step)
        self.walk_to_move()
        self.target_update()
        self.info_log(runner_step)
        #不在今日工作时间内直接return
        day_step = runner_step % self.one_day
        if day_step < self.go_work_time*(self.one_day/24) or day_step > self.get_off_work_time*(self.one_day/24):
            return
        self.choose_order(meituan,runner_step)

    def target_update(self):
        self.target.profit_present_time = self.target.income_present_time - self.target.cost_present_time
        if self.target.profit_present_time < 0:
            self.target.profit_present_time = 0 

        self.target.cost_list_present_time.append(self.target.cost_present_time)
        self.target.income_list_present_time.append(self.target.income_present_time)
        self.target.profit_list_present_time.append(self.target.profit_present_time)
        self.target.update_stability()
        self.target.update_robustness()
        self.target.update_inv()
        self.target.update_utility()    


    def decide_work_time(self,runner_step):
        if runner_step % self.one_day == 0:
            time_info ={
                'before_go_work_time':self.go_work_time,
                'before_get_off_work_time':self.get_off_work_time,
                'rider_num': self.rider_num
            }
            print(f"{self.id}开始决定工作时间")
            self.go_work_time,self.get_off_work_time = self.decide_time(runner_step=runner_step,info={**time_info,**self.rank_day})
    
    
    def choose_order(self,meituan,runner_step):
        chosen_order_list = [] #列表中元素类型为Order类
        chosen_order_id_list = []#元素类型为int，表示order_id
        order_list = meituan.now_orders_info
        if len(order_list) > 0 and self.max_orders - self.order_count > 0 and self.no_choose_order_step >= self.choose_order_step_interval:
            print(f'{self.id}开始选择订单')
            if len(order_list)>10:
                short_order_list = list(order_list.items())[:10]
            else:
                short_order_list = order_list
            info = {
                'order_list':short_order_list,
                'now_location':self.location,
                'accept_count':self.max_orders-self.order_count
            }

            chosen_order_id_list = self.take_order(runner_step,info)
            if len(chosen_order_list) > self.max_orders-self.order_count:
                chosen_order_list = chosen_order_list[:self.max_orders-self.order_count]
            for uid in chosen_order_id_list:
                if type(uid) is int and uid in meituan.normal_orders_dict.keys():
                    chosen_order_list.append(meituan.normal_orders_dict[uid])
            # keys = order_list.keys()
            # first_val = order_list[list(keys)[0]]
            # chosen_order_list.append(meituan.normal_orders_dict[first_val['order_id']])
            # chosen_order_id_list.append(first_val['order_id'])
            order_sequence_cal(chosen_order_list,self)
            self.order_count += len(chosen_order_list)
            self.total_order += len(chosen_order_list)
            self.order_day += len(chosen_order_list)
            meituan.choose_order_update(chosen_order_id_list)
            self.no_choose_order_step = 0
        else:
            self.no_choose_order_step += 1
        return chosen_order_list


    def write_info_init(self):
        # 记录信息
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = f"agent_log/{self.start_time}"
        folder_path = os.path.join(script_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = f"deliver_{self.id}_record.csv"
        filepath_agent = os.path.join(folder_path, path)

        with open(filepath_agent, mode='w', newline='') as file:
            file.seek(0)
            file.truncate()
            writer = csv.writer(file)
            writer.writerow(["step", "id", "x", "y", "time", "money","start_work_time", "end_work_time", "day","order_count","total_order","dis","learn_type", "stability", "robustness", "inv", "utility"])



    def route_to_walk(self, meituan, runner_step):
        """将派单算法中的路线进行路径规划，转为路线

        Args:
            meituan (_type_): 平台
            runner_step (_type_): 当前step 
        """
        # print(self.location)
        move_step = 0
        for pos in self.route:
            #获得所有路径
            order = meituan.get_order(pos[1])
            a_star_walk = None
            now_loc = self.now_plan_pos()
            if pos[0] == 'pickup':
                a_star_walk = AStarPlanner(now_loc[0], now_loc[1], order.pickup_location[0], order.pickup_location[1])
                walks = a_star_walk.planning()
                combine_walks = list(zip(walks[0], walks[1]))
                move_step += len(combine_walks)
                self.will_move_positions += combine_walks
            else:
                a_star_walk = AStarPlanner(now_loc[0], now_loc[1], order.delivery_location[0], order.delivery_location[1])
                walks = a_star_walk.planning()
                combine_walks = list(zip(walks[0], walks[1]))
                move_step += len(combine_walks)
                self.will_move_positions += combine_walks
                order.finish_time = runner_step + int(move_step / self.move_step)
                if order.finish_time in self.finish_orders_time:
                    self.finish_orders_time[order.finish_time].append((pos[1],order))
                else:
                    self.finish_orders_time[order.finish_time] = [(pos[1],order)]
        self.route = []  #清空所有的完成的规划路径


    def now_plan_pos(self):
        if len(self.will_move_positions):
            return self.will_move_positions[-1]
        else:
            return self.location



    def walk_to_move(self):
        """
            按照路径规划的路径，每一个step进行移动，并更新没有完成的路径
        """
        move_step = self.move_step
        i = 0
        self.target.cost_present_time = 0
        while i < move_step:
            if len(self.will_move_positions):
                self.pt = dpt(self.will_move_positions[0][0], self.will_move_positions[0][1], 0)
                self.location = (self.pt.x, self.pt.y)
                self.will_move_positions.pop(0)
                self.labor += 1
                self.dis_day += 1
                self.target.cost_present_time += 0.1
            else:
                break
            i += 1


    def orders_status_update(self, runner_step, meituan):
        """更新当前手中订单的数目

        Args:
            runner_step (_type_): 当前运行的时间
        """
        if runner_step in self.finish_orders_time:
            self.order_count -= len(self.finish_orders_time[runner_step]) # 订单当前数量更新
            now_money = sum(order[1].money for order in self.finish_orders_time[runner_step]) #收益更新
            self.money += now_money
            self.target.income_present_time = self.money
            self.money_day += now_money

            meituan.del_orders(self.finish_orders_time[runner_step], self.id, runner_step)

            del self.finish_orders_time[runner_step] #存储字典更新]

    def info_log(self, runner_step):
        path = f"agent_log/{self.start_time}/deliver_{self.id}_record.csv"
        with open(path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([runner_step, self.id, self.location[0], self.location[1], runner_step, int(self.money),
                            self.go_work_time, self.get_off_work_time,
                             int(runner_step) // self.one_day + 1, self.order_count, self.total_order, self.labor,"llm", self.target.stability, self.target.robustness, self.target.inv, self.target.utility])
    

