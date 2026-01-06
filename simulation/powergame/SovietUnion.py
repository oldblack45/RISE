import json

from simulation.models.agents.GameAgent import GameAgent
class SovietUnionAgent(GameAgent):
    def __init__(self):


        init_config = {
            'name': '苏联',
            'game_attributes':{
                "军事实力":85,
                '核武器力量':80,
                '民众士气':85,
                '领导力':99,
                '资源':70,
                '经济':65,
            }
        }
        super().__init__(init_config=init_config)

