import json

from simulation.models.agents.GameAgent import GameAgent
class AmericaAgent(GameAgent):
    def __init__(self):

        init_config =  {
            'name':'美国',
            'game_attributes':{
                "军事实力":90,
                '核武器力量':91,
                '民众士气':85,
                '领导力':79,
                '资源':90,
                '经济':85,
            }
        }
        super().__init__(init_config = init_config)

