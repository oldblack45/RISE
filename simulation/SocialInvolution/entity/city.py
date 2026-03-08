import datetime
import json
import sys
import time
import repast4py.space as space
from repast4py.core import Agent
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousSpace
from repast4py.space import ContinuousPoint as cpt
from repast4py import context as ctx
from repast4py import schedule, logging
from mpi4py import MPI
import random
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../../'))
from rider import Rider
from merchant import Merchant
from user import User
from meituan import Platform
from models.env.map.city_with_islands.read_map import ReadMap, AStarPlanner
class City:
    def __init__(self,
            comm: MPI.Intracomm, 
            one_day, 
            one_hour, 
            run_len,
            rider_num: int, 
            order_bf,
            init_riders_json='../config/rider_config.json',
            order_weight=0.3):
        super().__init__()
        #引擎初始化
        self.rank = comm.Get_rank()
        # 环境描述
        box = space.BoundingBox(0, 1000, 0, 1000, 0, 0)  # 设置边界
        self.grid = space.SharedGrid(name="grid", bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
        self.context = ctx.SharedContext(comm)
        self.context.add_projection(self.grid)
        self.read_map = ReadMap() # 初始化读取地图的类
        # 时间描述
        self.run_len = run_len
        self.one_day = one_day
        self.one_hour = one_hour
        self.runner_step = 0
        #agent初始化
        #骑手初始化
        self.riders = {}
        home_list =[]
        home_list = self.read_map.get_all_list()
        riders_config = self.read_riders_config(init_riders_json)
        starts_time = datetime.datetime.now()
        for i in range(rider_num):
            init_pos = random.choice(home_list)[0]
            role_param_dict = riders_config[i]
            rider = Rider(i, self.rank, 3, dpt(init_pos[0], init_pos[1], 0), max_orders=5, one_day=one_day, role_param_dict=role_param_dict, start_time=starts_time)
            self.riders[rider.id] = rider
            self.context.add(rider)
            self.grid.move(rider, rider.pt)
        # 用户初始化
        self.users = {}
        user_list_random = self.read_map.get_door("公司")
        i = 0
        for points in user_list_random:  # 用户初始化
            for point in points:
                user = User(dpt(point[0], point[1], 0), i, self.rank)
                i += 1
                self.users[i] = user
                self.context.add(user)
                self.grid.move(user, user.location)
        # 商家初始化
        self.merchants = {}
        merchant_list_random = self.read_map.get_door("饭店") + self.read_map.get_door("咖啡店") + self.read_map.get_door("面包店") + self.read_map.get_door("水果店")
        i = 0
        for points in merchant_list_random:  # 用户初始化
            for point in points:
                mer = Merchant(dpt(point[0], point[1], 0), i, self.rank)
                i += 1
                self.merchants[i] = mer
                self.context.add(mer)
                self.grid.move(mer, mer.pt)
        # 平台初始化
        self.meituan = Platform(order_bf, self.riders, starts_time)
        # 订单初始化
        mer_list = [sublist[0] for sublist in merchant_list_random]
        user_list = [sublist[0] for sublist in user_list_random]
        self.meituan.generate_order_num_all_step(self.run_len, int(self.one_day/2), mer_list, user_list, order_weight)
    
    
    def step(self):
        #每天开始时更新排行榜
        tmp_order_num = len(self.meituan.now_orders_info)
        if self.runner_step % self.one_day==0:
            self.meituan.update_rank()
        self.meituan.generate_order_now_step(self.runner_step)  # 生成当前时刻的订单
        agent_list = list(self.context.agents())
        random.shuffle(agent_list)
        for agent in agent_list:
            # 遍历骑手
            if agent.TYPE == 0:
                agent.step(self.meituan, self.runner_step)
        self.meituan.update_target() # 系统指标更新
        self.meituan.update_info_platform(self.runner_step, tmp_order_num) #平台信息存储
        self.runner_step += 1    
        pass

    def read_riders_config(self, init_riders_info):
        """"读取配置"""
        try:
            with open(init_riders_info) as json_file:
                riders_config = json.load(json_file)
        except Exception as e:
            print(e)
        return riders_config
        

def run_city(run_step_len, rider_num, init_riders_info, order_weight):
    start_time = time.time()
    # 程序代码
    # run_step_len = 360
    # rider_num = 5
    city = City(MPI.COMM_WORLD, 120, 5, run_step_len, rider_num, 3, init_riders_json=init_riders_info, order_weight=order_weight)
    for i in range(run_step_len):
        print("-" * 10 + str(i) + "-" * 10)
        city.step()

    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间为：", run_time, "秒")        


if __name__ == "__main__":
    # # 修改 一个step移动距离 订单生成
    # start_time = time.time()
    # # 程序代码
    # run_step_len = 360
    # rider_num = 5
    # city = City(MPI.COMM_WORLD, 120, 5, run_step_len, rider_num, 3, init_riders_json='../config/rider_config.json')
    # for i in range(run_step_len):
    #     print("-" * 10 + str(i) + "-" * 10)
    #     city.step()

    # end_time = time.time()
    # run_time = end_time -    start_time
    # print("程序运行时间为：", run_time, "秒")

    # run_city(3600, 100, '../config/rider_config_all.json',0.15)
    run_city(1800, 50, '../config/rider_config_hardworking.json', 0.3)
    run_city(1800, 50, '../config/rider_config_hardworking.json', 0.3)
    run_city(1800, 50, '../config/rider_config_hardworking.json', 0.3)
    run_city(1800, 50, '../config/rider_config_hardworking.json', 0.3)
    run_city(1800, 50, '../config/rider_config_hardworking.json', 0.3)