from repast4py.core import Agent
from repast4py.space import DiscretePoint as dpt


class Merchant(Agent):
    TYPE = 100
    def __init__(self, pt: dpt, id: int, rank: int):
        super().__init__(id=id, type=Merchant.TYPE, rank=rank)
        self.pt = pt
        self.action_list = {}
        self.status = 0 # 0休息 1开始工作 2准备订单
        self.money = 0 # 收入   

