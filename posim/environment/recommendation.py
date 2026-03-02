"""
推荐系统 - S_exp = α·Homophily + β·Popularity + γ·Recency
候选集: 50% 关系博文 + 50% 随机博文
exploration_rate 比例的推荐位用纯随机填充，抑制信息茧房
新增语义去重：与最近内容余弦相似度 > dedup_threshold 的博文被拒绝入池
"""
import numpy as np
import random
from datetime import datetime
from typing import List, Dict, Any

MAX_CANDIDATES = 100
DEDUP_THRESHOLD = 0.92
DEDUP_WINDOW = 200


class RecommendationSystem:

    def __init__(self, api_pool, config):
        self.api_pool = api_pool
        self.homophily_weight = config.homophily_weight
        self.popularity_weight = config.popularity_weight
        self.recency_weight = config.recency_weight
        self.exploration_rate = getattr(config, 'exploration_rate', 0.2)
        self.recommend_count = config.recommend_count
        self.comment_count = config.comment_count
        self.embedding_dimension = api_pool.embedding_dimension

        self.content_pool: List[Dict[str, Any]] = []
        self._post_id_counter = 0
        self._post_id_index: Dict[str, int] = {}     # post_id → content_pool index
        self._author_index: Dict[str, List[int]] = {} # author_id → indices
        self._following_map: Dict[str, set] = {}
        self._recommendation_history: Dict[str, set] = {}
        self._add_count_since_cleanup = 0
        self._recent_embeddings: List[np.ndarray] = []
    
    def _is_duplicate(self, embedding: np.ndarray) -> bool:
        """检查新博文是否与最近内容语义重复"""
        if embedding is None or not self._recent_embeddings:
            return False
        new_norm = np.linalg.norm(embedding)
        if new_norm == 0:
            return False
        for old_emb in self._recent_embeddings:
            old_norm = np.linalg.norm(old_emb)
            if old_norm > 0 and float(np.dot(embedding, old_emb) / (new_norm * old_norm)) > DEDUP_THRESHOLD:
                return True
        return False

    def _track_embedding(self, embedding: np.ndarray):
        if embedding is not None:
            self._recent_embeddings.append(embedding)
            if len(self._recent_embeddings) > DEDUP_WINDOW:
                self._recent_embeddings = self._recent_embeddings[-DEDUP_WINDOW:]

    def add_post(self, post: Dict[str, Any], current_time: str = None) -> str | None:
        self._post_id_counter += 1
        post_id = f"post_{self._post_id_counter}"
        post.setdefault('likes', 0)
        post.setdefault('reposts', 0)
        post.setdefault('comments_count', 0)
        post.setdefault('comments', [])
        post['id'] = post_id
        if 'time' not in post:
            post['time'] = current_time or datetime.now().isoformat()
        if 'embedding' not in post and self.api_pool:
            post['embedding'] = self.api_pool.encode([post.get('content', '')])[0]

        if self._is_duplicate(post.get('embedding')):
            return None

        self._track_embedding(post.get('embedding'))

        idx = len(self.content_pool)
        self.content_pool.append(post)
        self._post_id_index[post_id] = idx

        author_id = post.get('author_id', '')
        if author_id:
            self._author_index.setdefault(author_id, []).append(idx)

        self._add_count_since_cleanup += 1
        if self._add_count_since_cleanup >= 500:
            self._trim_recommendation_history()
            self._add_count_since_cleanup = 0
        return post_id
    
    def add_posts_batch(self, posts: List[Dict[str, Any]], current_time: str = None) -> List[str]:
        if not posts:
            return []

        post_ids = []
        to_encode, to_encode_posts = [], []

        for post in posts:
            self._post_id_counter += 1
            post_id = f"post_{self._post_id_counter}"
            post.setdefault('likes', 0)
            post.setdefault('reposts', 0)
            post.setdefault('comments_count', 0)
            post.setdefault('comments', [])
            post['id'] = post_id
            if 'time' not in post:
                post['time'] = current_time or datetime.now().isoformat()
            post_ids.append(post_id)

            if 'embedding' not in post and self.api_pool:
                content = post.get('content', '')
                if content.strip():
                    to_encode.append(content)
                    to_encode_posts.append(post)
                else:
                    post['embedding'] = np.zeros(self.embedding_dimension)

        if to_encode and self.api_pool:
            embeddings = self.api_pool.encode(to_encode)
            for p, emb in zip(to_encode_posts, embeddings):
                p['embedding'] = emb

        base_idx = len(self.content_pool)
        self.content_pool.extend(posts)
        for i, post in enumerate(posts):
            idx = base_idx + i
            self._post_id_index[post['id']] = idx
            author_id = post.get('author_id', '')
            if author_id:
                self._author_index.setdefault(author_id, []).append(idx)
            self._track_embedding(post.get('embedding'))

        return post_ids
    
    def set_relations(self, relations: List[Dict]):
        self._following_map.clear()
        for rel in relations:
            fid, tid = rel.get('follower_id', ''), rel.get('following_id', '')
            if fid and tid:
                self._following_map.setdefault(fid, set()).add(tid)

    def get_following(self, user_id: str) -> set:
        return self._following_map.get(user_id, set())
    
    def get_recommendations(self, user_profile: Dict, user_recent_posts: List[str],
                           current_time: str, count: int = None) -> List[Dict]:
        count = count or self.recommend_count
        if not self.content_pool:
            return []

        user_emb = self._get_user_embedding(user_profile, user_recent_posts)
        user_id = user_profile.get('user_id', '')
        following = self.get_following(user_id)
        history = self._recommendation_history.get(user_id, set())

        # 按类型分桶：原创 vs 转发，保证推荐列表中原创博文不被淹没
        original_cands, repost_cands = [], []
        for post in self.content_pool:
            if post.get('author_id') == user_id or post.get('type') == 'comment':
                continue
            if post['id'] in history:
                continue
            bucket = original_cands if post.get('type', 'original') == 'original' else repost_cands
            bucket.append(post)

        # 按 exploration_rate 划分：scored 推荐位 + 纯随机探索位
        n_explore = max(1, int(count * self.exploration_rate))
        n_scored = count - n_explore

        # 至少 30% 推荐位给原创博文
        n_original = max(1, int(n_scored * 0.5))
        n_repost = n_scored - n_original

        half = MAX_CANDIDATES // 2

        def _select(pool, n):
            if n <= 0 or not pool:
                return []
            cands = random.sample(pool, min(len(pool), half)) if len(pool) > half else pool
            scored = sorted(cands, key=lambda p: self._calculate_score(p, user_emb, current_time), reverse=True)
            top = scored[:n * 2]
            return random.sample(top, min(len(top), n))

        selected = _select(original_cands, n_original) + _select(repost_cands, n_repost)
        # 如果某个桶不足，从另一个桶补齐
        if len(selected) < n_scored:
            selected_ids = {p['id'] for p in selected}
            remaining = [p for p in original_cands + repost_cands if p['id'] not in selected_ids]
            selected += _select(remaining, n_scored - len(selected))

        # 探索位：从全部候选中纯随机选取（不经过打分排序）
        all_cands = original_cands + repost_cands
        selected_ids = {p['id'] for p in selected}
        explore_pool = [p for p in all_cands if p['id'] not in selected_ids]
        if explore_pool:
            selected += random.sample(explore_pool, min(len(explore_pool), n_explore))

        random.shuffle(selected)

        self._recommendation_history.setdefault(user_id, set()).update(p['id'] for p in selected)

        result = []
        for post in selected:
            pc = post.copy()
            comments = post.get('comments', [])
            pc['comments'] = random.sample(comments, min(len(comments), self.comment_count))
            result.append(pc)
        return result
    
    def _get_user_embedding(self, profile: Dict, recent_posts: List[str]) -> np.ndarray:
        combined = " ".join([profile.get('description', '')] + recent_posts[:5])
        if self.api_pool and combined.strip():
            return self.api_pool.encode([combined])[0]
        return np.zeros(self.embedding_dimension)

    def _calculate_score(self, post: Dict, user_emb: np.ndarray, current_time: str) -> float:
        """S_exp = α·Homophily + β·Popularity + γ·Recency"""
        # Homophily
        post_emb = post.get('embedding')
        if post_emb is not None and user_emb is not None:
            nu, np_ = np.linalg.norm(user_emb), np.linalg.norm(post_emb)
            homophily = float(np.dot(user_emb, post_emb) / (nu * np_)) if nu > 0 and np_ > 0 else 0.5
        else:
            homophily = 0.5

        # Popularity (log-scaled)
        eng = post.get('likes', 0) + post.get('reposts', 0) * 2 + post.get('comments_count', 0)
        popularity = min(1.0, np.log10(max(eng, 1) + 1) / 5)

        # Recency (exponential decay)
        pt, ct = post.get('time', ''), current_time
        if pt and ct:
            hours = (datetime.fromisoformat(ct) - datetime.fromisoformat(pt)).total_seconds() / 3600
            recency = np.exp(-hours / 24)
        else:
            recency = 0.5

        w = self.homophily_weight + self.popularity_weight + self.recency_weight or 1.0
        return (self.homophily_weight * homophily +
                self.popularity_weight * popularity +
                self.recency_weight * recency) / w
    
    def _get_post(self, post_id: str) -> Dict[str, Any] | None:
        idx = self._post_id_index.get(post_id)
        return self.content_pool[idx] if idx is not None else None

    def update_post_stats(self, post_id: str, action_type: str):
        post = self._get_post(post_id)
        if not post:
            return
        if action_type == 'like':
            post['likes'] += 1
        elif action_type in ('repost', 'repost_comment'):
            post['reposts'] += 1
        elif action_type in ('short_comment', 'long_comment'):
            post['comments_count'] += 1

    def add_comment(self, post_id: str, comment: str):
        post = self._get_post(post_id)
        if post:
            post['comments'].append(comment)
    
    def _trim_recommendation_history(self, max_per_user: int = 200):
        valid_ids = {p['id'] for p in self.content_pool}
        for uid in list(self._recommendation_history):
            h = self._recommendation_history[uid] & valid_ids
            self._recommendation_history[uid] = set() if len(h) > max_per_user else h

    def clear_old_posts(self, current_time: str, max_age_hours: int = 72):
        if not current_time:
            return
        cur = datetime.fromisoformat(current_time)
        self.content_pool = [
            p for p in self.content_pool
            if p.get('time') and (cur - datetime.fromisoformat(p['time'])).total_seconds() / 3600 < max_age_hours
        ]
        # 重建索引
        self._post_id_index.clear()
        self._author_index.clear()
        for i, p in enumerate(self.content_pool):
            self._post_id_index[p['id']] = i
            aid = p.get('author_id', '')
            if aid:
                self._author_index.setdefault(aid, []).append(i)
