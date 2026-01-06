import os
import csv
import json
from datetime import datetime

class GameLogger:
    def __init__(self, base_dir=None):
        """
        初始化日志管理器
        Args:
            base_dir: 日志基础目录，如果为None则使用当前文件所在目录
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建日志目录结构
        self.log_dir = os.path.join(base_dir, 'logs')
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.log_dir, f"session_{self.current_time}")
        
        # 确保目录存在
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 初始化日志文件
        self.world_log_file = os.path.join(self.session_dir, 'world_events.csv')
        self.america_log_file = os.path.join(self.session_dir, 'america_actions.csv')
        self.soviet_log_file = os.path.join(self.session_dir, 'soviet_actions.csv')
        
        # 初始化CSV头部
        self._init_log_files()
    
    def _init_log_files(self):
        """初始化所有日志文件的表头"""
        world_header = ['step', 'timestamp', 'world_info', 'situation', 'total_score', 'soviet_score', 'america_score']
        country_header = ['step', 'timestamp', 'declaration', 'action', 'attributes', 'america_attr_change', 'soviet_attr_change']
        
        self._init_csv_file(self.world_log_file, world_header)
        self._init_csv_file(self.america_log_file, country_header)
        self._init_csv_file(self.soviet_log_file, country_header)
    
    def _init_csv_file(self, file_path, header):
        """初始化单个CSV文件"""
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def log_country_action(self, country_type, step, country_agent, america_attr_change=None, soviet_attr_change=None):
        """
        记录国家行动
        Args:
            country_type: 'america' 或 'soviet'
            step: 当前步骤
            country_agent: 国家代理对象
            america_attr_change: 美国属性变化
            soviet_attr_change: 苏联属性变化
        """
        log_file = self.america_log_file if country_type == 'america' else self.soviet_log_file
        
        data = [
            step,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            country_agent.declaration[-1] if country_agent.declaration else '',
            country_agent.action[-1] if country_agent.action else '',
            json.dumps(country_agent.game_attributes, ensure_ascii=False) if hasattr(country_agent, 'game_attributes') else '{}',
            json.dumps(america_attr_change, ensure_ascii=False) if america_attr_change else '{}',
            json.dumps(soviet_attr_change, ensure_ascii=False) if soviet_attr_change else '{}'
        ]
        
        self._write_csv_row(log_file, data)
    
    def log_world_state(self, step, world_info, situation, scores=None):
        """
        记录世界状态
        Args:
            step: 当前步骤
            world_info: 世界信息摘要
            situation: 当前局势
            scores: (total_score, soviet_score, america_score)
        """
        if scores:
            total_score, soviet_score, america_score = scores
        else:
            total_score, soviet_score, america_score = None, None, None
            
        data = [
            step,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            world_info,
            situation,
            total_score,
            soviet_score,
            america_score
        ]
        
        self._write_csv_row(self.world_log_file, data)
    
    def _write_csv_row(self, file_path, row_data):
        """写入CSV行数据"""
        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            print(f"Error writing to log file {file_path}: {str(e)}")
            # 在实际应用中，这里可以添加更多的错误处理逻辑