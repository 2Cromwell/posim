"""
社交媒体推荐系统 - 模拟微博热门推荐机制
S_exp = α·Homophily + β·Popularity + γ·Recency
"""
import numpy as np
import random
from datetime import datetime
from typing import List, Dict, Any, Optional


class RecommendationSystem:
    """推荐系统
    S_exp = α·Homophily + β·Popularity + γ·Recency + δ·Relation
    """
    
    def __init__(self, api_pool, config):
        self.api_pool = api_pool
        self.homophily_weight = config.homophily_weight  # α
        self.popularity_weight = config.popularity_weight  # β
        self.recency_weight = config.recency_weight  # γ
        self.relation_weight = getattr(config, 'relation_weight', 0.5)  # δ
        self.recommend_count = config.recommend_count
        self.comment_count = config.comment_count
        
        # 从api_pool获取embedding维度配置
        self.embedding_dimension = api_pool.embedding_dimension
        
        # 内容池（存储所有博文）
        self.content_pool: List[Dict[str, Any]] = []
        self._post_id_counter = 0
        
        # 关注关系网络
        self._following_map: Dict[str, set] = {}  # user_id -> set of following_ids
    
    def add_post(self, post: Dict[str, Any], current_time: str = None) -> str:
        """添加博文到内容池"""
        self._post_id_counter += 1
        post_id = f"post_{self._post_id_counter}"
        post['id'] = post_id
        post['likes'] = post.get('likes', 0)
        post['reposts'] = post.get('reposts', 0)
        post['comments_count'] = post.get('comments_count', 0)
        post['comments'] = post.get('comments', [])
        if 'time' not in post:
            post['time'] = current_time if current_time else datetime.now().isoformat()
        if 'embedding' not in post and self.api_pool:
            post['embedding'] = self.api_pool.encode([post.get('content', '')])[0]
        self.content_pool.append(post)
        return post_id
    
    def add_posts_batch(self, posts: List[Dict[str, Any]], current_time: str = None) -> List[str]:
        """批量添加博文到内容池 - 优化版本"""
        if not posts:
            return []
        
        # 准备批量数据
        post_ids = []
        posts_to_encode = []
        posts_without_embedding = []
        
        for post in posts:
            self._post_id_counter += 1
            post_id = f"post_{self._post_id_counter}"
            post['id'] = post_id
            post['likes'] = post.get('likes', 0)
            post['reposts'] = post.get('reposts', 0)
            post['comments_count'] = post.get('comments_count', 0)
            post['comments'] = post.get('comments', [])
            if 'time' not in post:
                post['time'] = current_time if current_time else datetime.now().isoformat()
            
            post_ids.append(post_id)
            
            # 收集需要编码的文本
            if 'embedding' not in post and self.api_pool:
                content = post.get('content', '')
                if content.strip():
                    posts_to_encode.append(content)
                    posts_without_embedding.append(post)
                else:
                    post['embedding'] = np.zeros(self.embedding_dimension)
        
        # 批量编码embedding
        if posts_to_encode and self.api_pool:
            embeddings = self.api_pool.encode(posts_to_encode)
            for post, embedding in zip(posts_without_embedding, embeddings):
                post['embedding'] = embedding
        
        # 批量添加到内容池
        self.content_pool.extend(posts)
        
        return post_ids
    
    def set_relations(self, relations: List[Dict]):
        """设置关注关系数据"""
        self._following_map.clear()
        for rel in relations:
            follower_id = rel.get('follower_id', '')
            following_id = rel.get('following_id', '')
            if follower_id and following_id:
                if follower_id not in self._following_map:
                    self._following_map[follower_id] = set()
                self._following_map[follower_id].add(following_id)
    
    def get_following(self, user_id: str) -> set:
        """获取用户关注列表"""
        return self._following_map.get(user_id, set())
    
    def get_recommendations(self, user_profile: Dict, user_recent_posts: List[str],
                           current_time: str, count: int = None) -> List[Dict]:
        """获取推荐博文"""
        count = count or self.recommend_count
        if not self.content_pool:
            return []
        
        # 计算用户画像embedding
        user_emb = self._get_user_embedding(user_profile, user_recent_posts)
        user_id = user_profile.get('user_id', '')
        user_following = self.get_following(user_id)
        
        # 计算所有博文的得分
        scored_posts = []
        for post in self.content_pool:
            if post.get('author_id') == user_id:
                continue  # 不推荐自己的博文
            score = self._calculate_score(post, user_emb, current_time, user_following)
            scored_posts.append((post, score))
        
        # 排序并选取top-k
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        top_posts = [p for p, _ in scored_posts[:count * 2]]  # 取2倍再随机选
        
        # 添加随机性
        if len(top_posts) > count:
            selected = random.sample(top_posts, count)
        else:
            selected = top_posts[:count]
        
        # 为每条博文添加随机评论
        result = []
        for post in selected:
            post_copy = post.copy()
            post_copy['comments'] = self._get_random_comments(post, self.comment_count)
            result.append(post_copy)
        
        return result
    
    def _get_user_embedding(self, profile: Dict, recent_posts: List[str]) -> np.ndarray:
        """获取用户画像embedding"""
        texts = [profile.get('description', '')]
        texts.extend(recent_posts[:5])
        combined = " ".join(texts)
        if self.api_pool and combined.strip():
            return self.api_pool.encode([combined])[0]
        return np.zeros(self.embedding_dimension)
    
    def _calculate_score(self, post: Dict, user_emb: np.ndarray, current_time: str, 
                         user_following: set = None) -> float:
        """计算博文推荐得分
        S_exp = α·Homophily + β·Popularity + γ·Recency + δ·Relation
        """
        # Homophily（同质性）
        post_emb = post.get('embedding')
        if post_emb is not None and user_emb is not None:
            homophily = float(np.dot(user_emb, post_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(post_emb) + 1e-8))
        else:
            homophily = 0.5
        
        # Popularity（热度）
        likes = post.get('likes', 0)
        reposts = post.get('reposts', 0)
        popularity = min(1.0, (likes + reposts * 2) / 10000)
        
        # Recency（新鲜度）
        recency = self._calculate_recency(post.get('time', ''), current_time)
        
        # Relation（关系）- 如果作者是用户关注的人，提高得分
        relation = 0.0
        if user_following:
            author_id = post.get('author_id', '')
            if author_id in user_following:
                relation = 1.0
        
        # 归一化权重
        total_weight = (self.homophily_weight + self.popularity_weight + 
                       self.recency_weight + self.relation_weight)
        
        return (self.homophily_weight * homophily + 
                self.popularity_weight * popularity + 
                self.recency_weight * recency +
                self.relation_weight * relation) / total_weight
    
    def _calculate_recency(self, post_time: str, current_time: str) -> float:
        """计算新鲜度得分"""
        if not post_time or not current_time:
            return 0.5
        post_dt = datetime.fromisoformat(post_time)
        current_dt = datetime.fromisoformat(current_time)
        hours_ago = (current_dt - post_dt).total_seconds() / 3600
        return np.exp(-hours_ago / 24)
    
    def _get_random_comments(self, post: Dict, count: int) -> List[str]:
        """获取博文的随机评论"""
        all_comments = post.get('comments', [])
        if len(all_comments) >= count:
            return random.sample(all_comments, count)
        return all_comments
    
    def update_post_stats(self, post_id: str, action_type: str):
        """更新博文统计数据"""
        for post in self.content_pool:
            if post['id'] == post_id:
                if action_type == 'like':
                    post['likes'] = post.get('likes', 0) + 1
                elif action_type in ['repost', 'repost_comment']:
                    post['reposts'] = post.get('reposts', 0) + 1
                elif action_type in ['short_comment', 'long_comment']:
                    post['comments_count'] = post.get('comments_count', 0) + 1
                break
    
    def add_comment(self, post_id: str, comment: str):
        """添加评论到博文"""
        for post in self.content_pool:
            if post['id'] == post_id:
                if 'comments' not in post:
                    post['comments'] = []
                post['comments'].append(comment)
                break
    
    def clear_old_posts(self, current_time: str, max_age_hours: int = 72):
        """清理过期博文"""
        if current_time:
            current_dt = datetime.fromisoformat(current_time)
            self.content_pool = [
                p for p in self.content_pool
                if p.get('time') and (current_dt - datetime.fromisoformat(p['time'])).total_seconds() / 3600 < max_age_hours
            ]
