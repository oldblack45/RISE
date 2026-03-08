from repast4py.core import Agent
from repast4py.space import DiscretePoint as dpt


class User(Agent):
    TYPE = 10

    def __init__(self, pt: dpt, id_user: int, rank: int):
        super().__init__(id=id_user, type=User.TYPE, rank=rank)
        self.location = pt
        self.wish_time = float('inf')
        self.review = True
        self.false_review_time = 0

