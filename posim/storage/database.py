"""
数据库管理 - SQLite存储仿真数据
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class SimulationDatabase:
    """仿真数据库"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_tables()
    
    def _init_tables(self):
        """初始化数据表"""
        cursor = self.conn.cursor()
        
        # 行为记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                time TEXT,
                user_id TEXT,
                username TEXT,
                agent_type TEXT,
                action_type TEXT,
                target_post_id TEXT,
                target_author TEXT,
                content TEXT,
                emotion TEXT,
                stance TEXT,
                topics TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 状态快照表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                time TEXT,
                data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 统计数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                time TEXT,
                intensity REAL,
                active_count INTEGER,
                action_count INTEGER,
                hot_topics TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def save_action(self, action: Dict[str, Any], step: int):
        """保存行为记录"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO actions (step, time, user_id, username, agent_type, action_type,
                               target_post_id, target_author, content, emotion, stance, topics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            step,
            action.get('time', ''),
            action.get('user_id', ''),
            action.get('username', ''),
            action.get('agent_type', ''),
            action.get('action_type', ''),
            action.get('target_post_id', ''),
            action.get('target_author', ''),
            action.get('content', ''),
            action.get('emotion', ''),
            action.get('stance', ''),
            json.dumps(action.get('topics', []), ensure_ascii=False)
        ))
        self.conn.commit()
    
    def save_actions_batch(self, actions: List[Dict], step: int):
        """批量保存行为"""
        for action in actions:
            self.save_action(action, step)
    
    def save_snapshot(self, step: int, time: str, data: Dict):
        """保存状态快照"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO snapshots (step, time, data)
            VALUES (?, ?, ?)
        ''', (step, time, json.dumps(data, ensure_ascii=False)))
        self.conn.commit()
    
    def save_statistics(self, step: int, time: str, intensity: float, 
                       active_count: int, action_count: int, hot_topics: List):
        """保存统计数据"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO statistics (step, time, intensity, active_count, action_count, hot_topics)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (step, time, intensity, active_count, action_count, 
              json.dumps(hot_topics, ensure_ascii=False)))
        self.conn.commit()
    
    def get_actions(self, start_step: int = None, end_step: int = None) -> List[Dict]:
        """获取行为记录"""
        cursor = self.conn.cursor()
        query = "SELECT * FROM actions"
        params = []
        
        if start_step is not None or end_step is not None:
            conditions = []
            if start_step is not None:
                conditions.append("step >= ?")
                params.append(start_step)
            if end_step is not None:
                conditions.append("step <= ?")
                params.append(end_step)
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_snapshot(self, step: int) -> Optional[Dict]:
        """获取状态快照"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM snapshots WHERE step = ?", (step,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None
    
    def get_statistics(self) -> List[Dict]:
        """获取统计数据"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM statistics ORDER BY step")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()
