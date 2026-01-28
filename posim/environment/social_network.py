"""
社交网络管理 - 基于Neo4j图数据库
生成具有社交媒体性质的兴趣网络
支持粉丝关注网络、实时转发网络、实时评论网络
"""
from typing import List, Dict, Any, Set
from dataclasses import dataclass
import random
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkNode:
    """网络节点"""
    user_id: str
    username: str
    followers_count: int
    following_count: int
    influence_score: float


class SocialNetwork:
    """社交网络管理器 - 支持粉丝、转发、评论网络"""
    
    def __init__(self, neo4j_config=None):
        self.neo4j_config = neo4j_config
        self.driver = None
        self._nodes: Dict[str, NetworkNode] = {}
        self._edges: Dict[str, Set[str]] = {}  # user_id -> set of following user_ids
        self._followers: Dict[str, Set[str]] = {}  # user_id -> set of follower user_ids
        self._repost_network: Dict[str, List[Dict]] = {}  # 转发网络: post_id -> [{user_id, time}]
        self._comment_network: Dict[str, List[Dict]] = {}  # 评论网络: post_id -> [{user_id, time, content}]
        
        self.neo4j_enabled = neo4j_config and getattr(neo4j_config, 'enabled', False)
        if self.neo4j_enabled:
            self._init_neo4j()
    
    def _init_neo4j(self):
        """初始化Neo4j连接"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.neo4j_config.uri,
                auth=(self.neo4j_config.user, self.neo4j_config.password)
            )
            self._clear_database()
            logger.info(f"Neo4j connected: {self.neo4j_config.uri}")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Running without Neo4j.")
            self.driver = None
            self.neo4j_enabled = False
    
    def _clear_database(self):
        """清空数据库以开始新的仿真"""
        if self.driver:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
    
    def add_user(self, user_id: str, username: str, followers_count: int = 0, 
                following_count: int = 0, influence_score: float = 0.0):
        """添加用户节点"""
        node = NetworkNode(
            user_id=user_id,
            username=username,
            followers_count=followers_count,
            following_count=following_count,
            influence_score=influence_score
        )
        self._nodes[user_id] = node
        if user_id not in self._edges:
            self._edges[user_id] = set()
        if user_id not in self._followers:
            self._followers[user_id] = set()
        
        if self.driver:
            with self.driver.session() as session:
                session.run(
                    "MERGE (u:User {user_id: $uid}) "
                    "SET u.username = $name, u.followers_count = $fc, "
                    "u.following_count = $foc, u.influence_score = $inf",
                    uid=user_id, name=username, fc=followers_count,
                    foc=following_count, inf=influence_score
                )
    
    def add_follow(self, follower_id: str, following_id: str):
        """添加关注关系"""
        if follower_id not in self._edges:
            self._edges[follower_id] = set()
        if following_id not in self._followers:
            self._followers[following_id] = set()
        
        self._edges[follower_id].add(following_id)
        self._followers[following_id].add(follower_id)
        
        if self.driver:
            with self.driver.session() as session:
                # 使用单一MERGE语句避免笛卡尔积警告
                session.run(
                    "MERGE (a:User {user_id: $fid}) "
                    "MERGE (b:User {user_id: $tid}) "
                    "MERGE (a)-[:FOLLOWS]->(b)",
                    fid=follower_id, tid=following_id
                )
    
    def add_follows_batch(self, relations: list):
        """批量添加关注关系"""
        for rel in relations:
            follower_id = rel.get('follower_id') or rel.get('from')
            following_id = rel.get('following_id') or rel.get('to')
            if follower_id and following_id:
                if follower_id not in self._edges:
                    self._edges[follower_id] = set()
                if following_id not in self._followers:
                    self._followers[following_id] = set()
                self._edges[follower_id].add(following_id)
                self._followers[following_id].add(follower_id)
    
    def add_repost(self, user_id: str, post_id: str, original_author_id: str, time: str):
        """记录转发关系"""
        if post_id not in self._repost_network:
            self._repost_network[post_id] = []
        self._repost_network[post_id].append({'user_id': user_id, 'time': time})
        
        if self.driver:
            with self.driver.session() as session:
                session.run(
                    "MERGE (u:User {user_id: $uid}) "
                    "MERGE (p:Post {post_id: $pid}) "
                    "MERGE (u)-[:REPOSTED {time: $time}]->(p)",
                    uid=user_id, pid=post_id, time=time
                )
                # 记录用户间的转发关系
                if original_author_id:
                    session.run(
                        "MERGE (u1:User {user_id: $uid}) "
                        "MERGE (u2:User {user_id: $oid}) "
                        "MERGE (u1)-[:REPOSTED_FROM {post_id: $pid, time: $time}]->(u2)",
                        uid=user_id, oid=original_author_id, pid=post_id, time=time
                    )
    
    def add_comment(self, user_id: str, post_id: str, original_author_id: str, content: str, time: str):
        """记录评论关系"""
        if post_id not in self._comment_network:
            self._comment_network[post_id] = []
        self._comment_network[post_id].append({'user_id': user_id, 'time': time, 'content': content[:100]})
        
        if self.driver:
            with self.driver.session() as session:
                session.run(
                    "MERGE (u:User {user_id: $uid}) "
                    "MERGE (p:Post {post_id: $pid}) "
                    "MERGE (u)-[:COMMENTED {time: $time, content: $content}]->(p)",
                    uid=user_id, pid=post_id, time=time, content=content[:100]
                )
                # 记录用户间的评论关系
                if original_author_id:
                    session.run(
                        "MERGE (u1:User {user_id: $uid}) "
                        "MERGE (u2:User {user_id: $oid}) "
                        "MERGE (u1)-[:COMMENTED_ON {post_id: $pid, time: $time}]->(u2)",
                        uid=user_id, oid=original_author_id, pid=post_id, time=time
                    )
    
    def get_repost_network(self) -> Dict[str, List[Dict]]:
        """获取转发网络用于可视化"""
        return self._repost_network
    
    def get_comment_network(self) -> Dict[str, List[Dict]]:
        """获取评论网络用于可视化"""
        return self._comment_network
    
    def get_following(self, user_id: str) -> List[str]:
        """获取关注列表"""
        return list(self._edges.get(user_id, set()))
    
    def get_followers(self, user_id: str) -> List[str]:
        """获取粉丝列表"""
        return list(self._followers.get(user_id, set()))
    
    def get_influence_score(self, user_id: str) -> float:
        """获取影响力得分"""
        node = self._nodes.get(user_id)
        return node.influence_score if node else 0.0
    
    def generate_ba_network(self, users: List[Dict], m: int = 3):
        """
        基于BA无标度网络模型生成社交网络
        考虑用户影响力的优先连接
        """
        # 按影响力排序
        sorted_users = sorted(users, key=lambda x: x.get('followers_count', 0), reverse=True)
        
        for user in sorted_users:
            self.add_user(
                user_id=user['user_id'],
                username=user['username'],
                followers_count=user.get('followers_count', 0),
                following_count=user.get('following_count', 0),
                influence_score=self._calc_influence(user)
            )
        
        # 使用BA模型建立连接
        user_ids = [u['user_id'] for u in sorted_users]
        if len(user_ids) <= m:
            for i, uid in enumerate(user_ids):
                for other in user_ids[:i]:
                    self.add_follow(uid, other)
            return
        
        # 初始化完全连接的种子网络
        for i in range(m):
            for j in range(i):
                self.add_follow(user_ids[i], user_ids[j])
                self.add_follow(user_ids[j], user_ids[i])
        
        # 优先连接
        degrees = {uid: len(self._followers.get(uid, set())) + 1 for uid in user_ids[:m]}
        
        for uid in user_ids[m:]:
            # 计算连接概率
            total = sum(degrees.values())
            probs = [(other, d / total) for other, d in degrees.items()]
            
            # 选择m个节点连接
            targets = set()
            attempts = 0
            while len(targets) < m and attempts < 100:
                r = random.random()
                cumsum = 0
                for other, p in probs:
                    cumsum += p
                    if r < cumsum and other != uid:
                        targets.add(other)
                        break
                attempts += 1
            
            for target in targets:
                self.add_follow(uid, target)
                degrees[target] = degrees.get(target, 0) + 1
            
            degrees[uid] = 1
    
    def _calc_influence(self, user: Dict) -> float:
        """计算用户影响力"""
        followers = user.get('followers_count', 0)
        posts = user.get('posts_count', 0)
        return math.log(followers + 1) * 0.7 + math.log(posts + 1) * 0.3
    
    def get_neighbor_emotions(self, user_id: str, agents: Dict) -> List:
        """获取邻居的情绪状态"""
        following = self.get_following(user_id)
        emotions = []
        for fid in following:
            agent = agents.get(fid)
            if agent and hasattr(agent, 'belief_system'):
                emotions.append(agent.belief_system.emotion.emotion_vector)
        return emotions
