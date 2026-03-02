"""
Tianjiaerhuan Event Data Annotation System
Academic research project for public opinion analysis
"""

import json
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from snownlp import SnowNLP
import re
import random
import aiohttp

class DataAnnotator:
    def __init__(self, config_path: str = "data_process/tianjaierhuan/config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.api_configs = [c for c in config['llm']['api_configs'] if c.get('enabled', True)]
        self.concurrency = config['llm'].get('concurrency', 50)
        self.event_background = "2025年5月，演员黄杨钿甜发布成人礼照片，佩戴的绿色耳饰被网友扒出疑似价值230万的奢侈品牌Graff耳环。"
        self.stance_target = "黄杨钿甜佩戴耳环的行为"
        
        # Time filtering
        self.start_time = datetime.strptime("2025-05-15 00:00:00", "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.strptime("2025-05-18 23:59:59", "%Y-%m-%d %H:%M:%S")
        
        # Retry and debug settings
        self.max_retries = 3
        self.debug_probability = 0.0005  # 0.05%
        
        self.output_dir = Path("data_process/tianjaierhuan/output")
        self.labels_file = self.output_dir / "labels.json"
        self.failed_file = self.output_dir / "failed_annotations.json"
        self.existing_labels = self._load_existing_labels()
        self.failed_annotations = []
        
        # Initialize AsyncOpenAI clients for each API config
        self.clients = []
        for config in self.api_configs:
            client = AsyncOpenAI(
                api_key=config['api_key'],
                base_url=config['base_url']
            )
            self.clients.append({
                'client': client,
                'config': config
            })
        
    def _load_existing_labels(self) -> Dict:
        """Load existing annotations to skip already processed posts"""
        if self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                return {item['post_id']: item for item in json.load(f)}
        return {}
    
    def _is_in_time_range(self, post: Dict) -> bool:
        """Check if post is within the specified time range"""
        time_str = post.get('time', '')
        if not time_str:
            return False
        
        try:
            # Try multiple time formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    post_time = datetime.strptime(time_str, fmt)
                    in_range = self.start_time <= post_time <= self.end_time
                    return in_range
                except ValueError:
                    continue
            return False
        except Exception as e:
            print(f"[WARNING] Failed to parse time '{time_str}': {e}")
            return False
    
    def _should_debug_print(self) -> bool:
        """Randomly decide if should print debug info (0.05% probability)"""
        return random.random() < self.debug_probability
    
    def _is_valid_post(self, post: Dict) -> bool:
        """Check if post should be annotated (filter out short/empty content)"""
        post_type = post.get('type', '')
        
        # For reposts: skip if no user content (direct repost)
        if post_type == 'repost':
            user_content = post.get('user_content', '').strip()
            if not user_content or user_content == '转发微博':
                return False
            # Filter very short reposts (likely just emojis)
            if len(user_content) < 3:
                return False
        
        # For comments: filter very short comments (likely just emojis)
        elif post_type == 'comment':
            content = post.get('content', '').strip()
            if not content or len(content) < 3:
                return False
        
        # For original posts: filter very short posts
        elif post_type == 'original':
            content = post.get('content', '').strip()
            if not content or len(content) < 5:
                return False
        
        return True
    
    def _generate_post_id(self, post: Dict) -> str:
        """Generate unique ID from time and URL"""
        time_str = post.get('time', '')
        url = post.get('url', '')
        return hashlib.md5(f"{time_str}_{url}".encode()).hexdigest()
    
    def _extract_context(self, post: Dict) -> str:
        """Extract context chain for reposts and comments"""
        context_parts = []
        
        # For reposts: include root content
        if post.get('type') == 'repost':
            if post.get('root_content'):
                context_parts.append(f"原博文: {post['root_content'][:200]}")
            content = post.get('user_content', '')
            if content:
                context_parts.append(f"转发内容: {content}")
        
        # For comments: include original post and reply chain
        elif post.get('type') == 'comment':
            if post.get('original_post_content'):
                context_parts.append(f"原博文: {post['original_post_content'][:200]}")
            if post.get('replied_to_content'):
                context_parts.append(f"回复对象: {post['replied_to_content'][:100]}")
            content = post.get('content', '')
            context_parts.append(f"评论内容: {content}")
        
        # For original posts
        else:
            content = post.get('content', '')
            context_parts.append(f"博文内容: {content}")
        
        return "\n".join(context_parts) if context_parts else post.get('content', '') or post.get('user_content', '')
    
    def _determine_behavior_type(self, post: Dict) -> str:
        """Rule-based behavior type classification"""
        post_type = post.get('type', '')
        
        if post_type == 'original':
            content_len = len(post.get('content', ''))
            return '长博文' if content_len > 50 else '短博文'
        
        elif post_type == 'repost':
            user_content = post.get('user_content', '').strip()
            # Check if it's just a repost without comment
            if not user_content or user_content == '转发微博':
                return '仅转发'
            # Check comment length
            content_len = len(user_content)
            return '转发并长评论' if content_len > 20 else '转发并短评论'
        
        elif post_type == 'comment':
            content = post.get('content', '')
            content_len = len(content)
            return '长评论' if content_len > 20 else '短评论'
        
        return '未知'
    
    def _nlp_sentiment_analysis(self, text: str) -> Dict:
        """Traditional NLP sentiment analysis using SnowNLP"""
        try:
            s = SnowNLP(text)
            score = s.sentiments
            
            # Classify sentiment
            if score > 0.6:
                polarity = 'positive'
            elif score < 0.4:
                polarity = 'negative'
            else:
                polarity = 'neutral'
            
            return {
                'method': 'snownlp',
                'score': round(score, 3),
                'polarity': polarity
            }
        except:
            return {'method': 'snownlp', 'score': 0.5, 'polarity': 'neutral'}
    
    def _build_annotation_prompt(self, post: Dict, context: str) -> str:
        """Build COT prompt for LLM annotation"""
        behavior_type = self._determine_behavior_type(post)
        
        prompt = f"""你是一位专业的舆情分析专家。请对以下微博内容进行多维度标注。

【事件背景】
{self.event_background}

【博文信息】
行为类型: {behavior_type}
发布时间: {post.get('time', 'unknown')}

【博文内容及上下文】
{context}

【标注任务】
请使用思维链(Chain of Thought)方式，逐步分析并标注以下维度：

1. **情绪类型**: 从以下选择一个或多个：愤怒/厌恶/焦虑/悲伤/幸灾乐祸/兴奋/中性
2. **情绪强度**: 低/中等/高/极高
3. **立场**: 针对"{self.stance_target}"，选择：支持/反对/中立
4. **立场强度**: 低/中等/高/极高
5. **表达风格**: 从以下选择一个或多个：阴阳怪气/讽刺/激进/嘲讽/情绪宣泄/质疑/共情/冷漠/理性/客观
6. **叙事策略**: 从以下选择一个或多个：贴标签/道德绑架/阴谋论/转移话题/号召行动/人身攻击/质疑事实/陈述事实/提供证据
7. **情绪极性**: 负面/中性/正面
8. **态度**: 负面/中立/正面
9. **信念**: 相信此事件/怀疑此事件/不确定
10. **礼貌度**: 礼貌/一般/不礼貌/粗鲁
11. **情绪性**: 低/中/高 (衡量情绪化程度)

请按以下JSON格式输出(不要包含markdown代码块标记):
{{
  "reasoning": "你的分析推理过程",
  "emotion_type": ["情绪类型"],
  "emotion_intensity": "情绪强度",
  "stance": "立场",
  "stance_intensity": "立场强度",
  "expression_style": ["表达风格"],
  "narrative_strategy": ["叙事策略"],
  "sentiment_polarity": "情绪极性",
  "attitude": "态度",
  "belief": "信念",
  "politeness": "礼貌度",
  "emotionality": "情绪性"
}}"""
        return prompt
    
    async def _call_llm_api(self, prompt: str, client_info: Dict, retry_count: int = 0) -> Optional[Dict]:
        """Async LLM API call with retry logic using AsyncOpenAI"""
        try:
            client = client_info['client']
            config = client_info['config']
            
            # Debug print with 0.05% probability
            if self._should_debug_print():
                print("\n" + "="*80)
                print(f"[DEBUG] API: {config['name']} | Model: {config['model']}")
                print(f"[DEBUG] Prompt (first 500 chars):\n{prompt[:500]}...")
                print("="*80)
            
            # Call API using AsyncOpenAI
            response = await client.chat.completions.create(
                model=config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get('temperature', 0.7),
                top_p=config.get('top_p', 0.9)
            )
            
            content = response.choices[0].message.content
            
            # Debug print response
            if self._should_debug_print():
                print(f"[DEBUG] Response (first 500 chars):\n{content[:500]}...")
                print("="*80 + "\n")
            
            # Extract JSON from response
            content = content.strip()
            if content.startswith('```'):
                content = re.sub(r'^```(?:json)?\s*\n', '', content)
                content = re.sub(r'\n```\s*$', '', content)
            
            return json.loads(content)
                    
        except Exception as e:
            if retry_count < self.max_retries - 1:
                # Retry with exponential backoff
                await asyncio.sleep(2 ** retry_count)
                return await self._call_llm_api(prompt, client_info, retry_count + 1)
            else:
                # Max retries reached
                print(f"\n[ERROR] API call failed after {self.max_retries} attempts: {str(e)[:200]}")
                return None
    
    async def _annotate_single_post(self, post: Dict, api_idx: int) -> Optional[Dict]:
        """Annotate a single post"""
        post_id = self._generate_post_id(post)
        
        # Skip if already annotated
        if post_id in self.existing_labels:
            return self.existing_labels[post_id]
        
        try:
            # Extract context and build prompt
            context = self._extract_context(post)
            prompt = self._build_annotation_prompt(post, context)
            
            # Get client (round-robin)
            client_info = self.clients[api_idx % len(self.clients)]
            
            # LLM annotation with retry
            llm_result = await self._call_llm_api(prompt, client_info)
            
            if llm_result is None:
                # Record failure
                failure_info = {
                    'post_id': post_id,
                    'time': post.get('time'),
                    'url': post.get('url'),
                    'type': post.get('type'),
                    'reason': 'LLM API call failed after retries',
                    'failed_at': datetime.now().isoformat()
                }
                self.failed_annotations.append(failure_info)
                print(f"\n[FAILED] Post {post_id[:8]}... | Time: {post.get('time')} | Reason: LLM API failed")
                return None
            
            # NLP sentiment analysis - get the actual content
            content = ''
            if post.get('type') == 'repost':
                content = post.get('user_content', '')
            elif post.get('type') == 'comment':
                content = post.get('content', '')
            else:  # original
                content = post.get('content', '')
            
            nlp_sentiment = self._nlp_sentiment_analysis(content) if content else {'method': 'snownlp', 'score': 0.5, 'polarity': 'neutral'}
            
            # Combine results
            annotation = {
                'post_id': post_id,
                'time': post.get('time'),
                'url': post.get('url'),
                'type': post.get('type'),
                'content': content[:500],  # Truncate for storage
                'behavior_type': self._determine_behavior_type(post),
                'llm_annotation': llm_result,
                'nlp_sentiment': nlp_sentiment,
                'annotated_at': datetime.now().isoformat()
            }
            
            return annotation
            
        except Exception as e:
            # Record failure
            failure_info = {
                'post_id': post_id,
                'time': post.get('time'),
                'url': post.get('url'),
                'type': post.get('type'),
                'reason': f'Exception: {str(e)[:200]}',
                'failed_at': datetime.now().isoformat()
            }
            self.failed_annotations.append(failure_info)
            print(f"\n[FAILED] Post {post_id[:8]}... | Time: {post.get('time')} | Reason: {str(e)[:100]}")
            return None
    
    async def _save_batch(self, annotations: List[Dict]):
        """Save batch of annotations"""
        # Filter out None values (failed annotations)
        valid_annotations = [ann for ann in annotations if ann is not None]
        
        # Update existing labels
        for ann in valid_annotations:
            self.existing_labels[ann['post_id']] = ann
        
        # Write to file
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.existing_labels.values()), f, ensure_ascii=False, indent=2)
        
        # Save failed annotations
        if self.failed_annotations:
            with open(self.failed_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_annotations, f, ensure_ascii=False, indent=2)
    
    async def annotate_dataset(self, data_path: str = "data_process/tianjaierhuan/output/base_data.json", batch_size: int = 500):
        """Main annotation pipeline"""
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Flatten all posts from nested structure
        all_posts = []
        
        def extract_posts_recursive(obj, parent_type=None):
            """Recursively extract all posts from nested structure"""
            if isinstance(obj, dict):
                # Check if this is a post
                if 'type' in obj and obj['type'] in ['original', 'repost', 'comment']:
                    all_posts.append(obj)
                
                # Recursively check nested structures
                for key in ['original_posts', 'repost_posts', 'comments']:
                    if key in obj and isinstance(obj[key], list):
                        for item in obj[key]:
                            extract_posts_recursive(item, key)
            
            elif isinstance(obj, list):
                for item in obj:
                    extract_posts_recursive(item, parent_type)
        
        # Extract from all users
        for user_data in data:
            extract_posts_recursive(user_data)
        
        print(f"Total posts extracted: {len(all_posts)}")
        
        # Filter by time range
        time_filtered = [p for p in all_posts if self._is_in_time_range(p)]
        print(f"Posts in time range (2025-05-15 to 2025-05-19): {len(time_filtered)}")
        
        # Filter valid posts (exclude short/empty content and direct reposts)
        valid_posts = [p for p in time_filtered if self._is_valid_post(p)]
        print(f"Valid posts (after filtering short/empty content): {len(valid_posts)}")
        print(f"  - Filtered out: {len(time_filtered) - len(valid_posts)} posts")
        
        # Show sample times for debugging
        if valid_posts:
            sample_times = [p.get('time', 'N/A') for p in valid_posts[:5]]
            print(f"Sample times from valid posts: {sample_times}")
        
        print(f"Already annotated: {len(self.existing_labels)}")
        
        # Filter unannotated posts
        unannotated = [p for p in valid_posts if self._generate_post_id(p) not in self.existing_labels]
        print(f"To annotate: {len(unannotated)}")
        print(f"Time range: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} to {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Max retries per post: {self.max_retries}")
        print(f"Debug print probability: {self.debug_probability*100}%")
        print(f"Using {len(self.clients)} API clients")
        
        if not unannotated:
            print("All posts already annotated!")
            return
        
        # Process in batches
        for i in range(0, len(unannotated), batch_size):
            batch = unannotated[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(unannotated)-1)//batch_size + 1}")
            
            # Annotate batch with progress bar
            tasks = [self._annotate_single_post(post, idx) for idx, post in enumerate(batch)]
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Annotating"):
                result = await coro
                results.append(result)
            
            # Save batch
            await self._save_batch(results)
            valid_count = sum(1 for r in results if r is not None)
            failed_count = len(results) - valid_count
            print(f"Saved {valid_count} annotations, {failed_count} failed")
        
        print(f"\nAnnotation complete!")
        print(f"Total successful: {len(self.existing_labels)}")
        print(f"Total failed: {len(self.failed_annotations)}")
        if self.failed_annotations:
            print(f"Failed annotations saved to: {self.failed_file}")


async def main():
    annotator = DataAnnotator()
    await annotator.annotate_dataset()


if __name__ == "__main__":
    asyncio.run(main())
