# 世界模拟器命令行运行版本
import time
from simulation.powergame.SovietUnion import SovietUnionAgent
from simulation.powergame.America import AmericaAgent
from simulation.models.agents.SecretaryAgent import WorldSecretaryAgent
from simulation.powergame.logger import GameLogger
class World:
    def __init__(self):
        self.soviet_union = SovietUnionAgent()
        self.america = AmericaAgent()
        self.world_secretary = WorldSecretaryAgent()
        self.world_memory = None
        self.step = 1
        self.exit_game = False
        self.logger = GameLogger()
        self.last_scores = (None, None, None)
    def attributes_adjust_world(self,america_attr_change,soviet_attr_change):
        self.america.game_attributes['军事实力'] += america_attr_change['军事实力']
        self.america.game_attributes['核武器力量'] += america_attr_change['核武器力量']
        self.america.game_attributes['民众士气'] += america_attr_change['民众士气']
        self.america.game_attributes['领导力'] += america_attr_change['领导力']
        self.america.game_attributes['资源'] += america_attr_change['资源']
        self.america.game_attributes['经济'] += america_attr_change['经济']
        self.soviet_union.game_attributes['军事实力'] += soviet_attr_change['军事实力']
        self.soviet_union.game_attributes['核武器力量'] += soviet_attr_change['核武器力量']
        self.soviet_union.game_attributes['民众士气'] += soviet_attr_change['民众士气']
        self.soviet_union.game_attributes['领导力'] += soviet_attr_change['领导力']
        self.soviet_union.game_attributes['资源'] += soviet_attr_change['资源']
        self.soviet_union.game_attributes['经济'] += soviet_attr_change['经济']
    def america_run(self):
        self.america.run(self.world_memory)
        america_attr_change, soviet_attr_change = self.world_secretary.attributes_adjust(self.world_memory,self.america, self.soviet_union)
        self.attributes_adjust_world(america_attr_change,soviet_attr_change)
        exit_game, score, america_score, soviet_score = self.world_secretary.cal_score(self.world_memory)
        if exit_game:
            self.exit_game = True
        self.world_memory += '美国回复:' + self.america.action[-1] + '\n'
        self.world_memory += '美国宣言:' + self.america.declaration[-1] + '\n'
        self.logger.log_country_action('america', self.step, self.america, america_attr_change, soviet_attr_change)
        # 保存分数用于后续记录
        self.last_scores = (score, soviet_score, america_score)
        

    def soviet_run(self):
        self.soviet_union.run(self.world_memory)
        america_attr_change, soviet_attr_change = self.world_secretary.attributes_adjust(self.world_memory,self.soviet_union, self.america)
        self.attributes_adjust_world(america_attr_change,soviet_attr_change)
        exit_game, score, america_score, soviet_score = self.world_secretary.cal_score(self.world_memory)
        if exit_game:
            self.exit_game = True
        self.world_memory += '苏联回复:' + self.soviet_union.action[-1] + '\n'
        self.world_memory += '苏联宣言:' + self.soviet_union.declaration[-1] + '\n'
        self.logger.log_country_action('soviet', self.step, self.soviet_union, america_attr_change, soviet_attr_change)
        # 更新分数用于后续记录
        self.last_scores = (score, soviet_score, america_score)
        
    def run_one_step(self):
        print("-" * 10 + str(self.step) + "-" * 10)
        if self.world_memory is None:
            self.world_memory = '1962年10月16日，一架飞越古巴上空的美军U-2侦察机拍摄到了苏联在古巴的核导弹基地。\n'

        self.america_run()
        self.logger.log_world_state(
            self.step,
            None,
            None,
            self.last_scores
        )
        self.soviet_run()
        self.logger.log_world_state(
            self.step,
            None,
            None,
            self.last_scores
        )
        sum_world_info, situation = self.world_secretary.scenario_summary(
            self.world_memory)
        # 记录世界状态
        self.logger.log_world_state(
            self.step,
            sum_world_info,
            situation,
            None
        )
        self.step += 1
        if self.exit_game:
            return

    def start_sim(self):
        self.step = 1
        for i in range(10):
            self.run_one_step()
            if self.exit_game:
                print(f"已结束step{self.step}")
                return

if __name__ == '__main__':
    for i in range(5):
        start_time = time.time()
        world =World()
        world.start_sim()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"耗时: {elapsed_time:.6f} 秒")