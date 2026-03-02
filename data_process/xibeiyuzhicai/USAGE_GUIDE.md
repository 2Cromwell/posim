# 西贝预制菜事件数据标注系统 - 使用指南

## Overview
This system provides comprehensive multi-dimensional annotation for social media posts related to the 西贝预制菜 (Western Restaurant Pre-made Dishes) event using both LLM-based and traditional NLP methods.

## Key Features

### 1. Time Range Filtering
- **Range**: 2025-09-12 05:00:00 to 2025-09-16 05:00:00 (configured in config.json)
- Automatically filters posts outside this simulation period
- Supports multiple time formats for robustness

### 2. Content Filtering
The system automatically filters out:
- **Direct reposts** without comments (仅转发)
- **Very short comments** (< 3 characters, likely just emojis)
- **Very short original posts** (< 5 characters)
- Empty or whitespace-only content

### 3. Retry Mechanism
- **Max retries**: 3 attempts per post
- **Exponential backoff**: 2^retry_count seconds between retries
- Failed annotations are logged with detailed error messages

### 4. Debug Printing
- **Probability**: 0.05% (1 in 2000 posts)
- Prints prompt and response for random samples
- Helps monitor annotation quality

### 5. LLM API Integration
- Uses AsyncOpenAI client (compatible with OpenAI API format)
- Supports multiple API endpoints with round-robin load balancing
- Configured via `config.json`

### 6. Annotation Dimensions

#### Behavior Type (Rule-based)
- 长博文 (Long Post): > 50 characters
- 短博文 (Short Post): ≤ 50 characters
- 转发并长评论 (Repost with Long Comment): > 20 characters
- 转发并短评论 (Repost with Short Comment): ≤ 20 characters
- 长评论 (Long Comment): > 20 characters
- 短评论 (Short Comment): ≤ 20 characters

#### LLM-based Annotations
1. **Emotion Type**: 愤怒/厌恶/焦虑/悲伤/幸灾乐祸/兴奋/中性
2. **Emotion Intensity**: 低/中等/高/极高
3. **Stance**: 支持/反对/中立 (towards "西贝预制菜事件")
4. **Stance Intensity**: 低/中等/高/极高
5. **Expression Style**: 阴阳怪气/讽刺/激进/嘲讽/情绪宣泄/质疑/共情/冷漠/理性/客观
6. **Narrative Strategy**: 贴标签/道德绑架/阴谋论/转移话题/号召行动/人身攻击/质疑事实/陈述事实/提供证据
7. **Sentiment Polarity**: 负面/中性/正面
8. **Attitude**: 负面/中立/正面
9. **Belief**: 相信此事件/怀疑此事件/不确定
10. **Politeness**: 礼貌/一般/不礼貌/粗鲁
11. **Emotionality**: 低/中/高

#### NLP-based Annotations
- **Method**: SnowNLP
- **Score**: 0.0 to 1.0 (sentiment score)
- **Polarity**: negative/neutral/positive

## Installation

```bash
# Install required packages
pip install -r data_process/xibeiyuzhicai/requirements_annotation.txt

# Required packages:
# - openai>=1.12.0
# - snownlp>=0.12.3
# - tqdm>=4.66.0
# - matplotlib>=3.8.0
# - seaborn>=0.13.0
# - pandas>=2.1.0
# - numpy>=1.26.0
```

## Usage

### Step 1: Configure API Keys
Edit `data_process/xibeiyuzhicai/config.json`:
```json
{
  "llm": {
    "api_configs": [
      {
        "name": "api1",
        "enabled": true,
        "base_url": "https://api.siliconflow.cn/v1/",
        "api_key": "your-api-key",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0.7,
        "top_p": 0.9
      }
    ]
  }
}
```

### Step 2: Run Annotation
```bash
# Run annotation only
python data_process/xibeiyuzhicai/annotator.py

# Or run full pipeline (annotation + visualization)
python data_process/xibeiyuzhicai/run_annotation.py
```

### Step 3: Generate Visualizations
```bash
python data_process/xibeiyuzhicai/visualizer.py
```

## Output Files

### 1. labels.json
Main annotation results file:
```json
[
  {
    "post_id": "abc123...",
    "time": "2025-09-14 10:08:15",
    "url": "http://weibo.com/...",
    "type": "repost",
    "content": "post content...",
    "behavior_type": "Repost with Short Comment",
    "llm_annotation": {
      "reasoning": "...",
      "emotion_type": ["Anger"],
      "emotion_intensity": "High",
      ...
    },
    "nlp_sentiment": {
      "method": "snownlp",
      "score": 0.234,
      "polarity": "negative"
    },
    "annotated_at": "2026-02-04T10:30:00"
  }
]
```

### 2. failed_annotations.json
Records of failed annotations:
```json
[
  {
    "post_id": "def456...",
    "time": "2025-09-14 12:00:00",
    "url": "http://weibo.com/...",
    "type": "comment",
    "reason": "LLM API call failed after retries",
    "failed_at": "2026-02-04T10:35:00"
  }
]
```

### 3. Visualizations (output/visualizations/)
- `behavior_type_distribution.png`
- `emotion_distribution.png`
- `stance_distribution.png`
- `intensity_heatmap.png`
- `sentiment_comparison.png`
- `expression_style.png`
- `narrative_strategy.png`
- `belief_distribution.png`
- `confrontation_metrics.png`
- `temporal_sentiment_trend.png`

## Progress Monitoring

The system provides real-time progress information:

```
Loading data from data_process/xibeiyuzhicai/output/base_data.json...
Total posts extracted: 105974
Posts in time range (2025-09-12 to 2025-09-16): 45230
Valid posts (after filtering short/empty content): 42150
  - Filtered out: 3080 posts
Already annotated: 0
To annotate: 42150
Time range: 2025-09-12 05:00:00 to 2025-09-16 05:00:00
Max retries per post: 3
Debug print probability: 0.05%
Using 8 API clients

Processing batch 1/844
Annotating: 100%|████████████████| 50/50 [00:45<00:00,  1.11it/s]
Saved 48 annotations, 2 failed

[FAILED] Post abc12345... | Time: 2025-09-14 10:30:00 | Reason: LLM API failed
```

## Troubleshooting

### Issue: Time filtering not working
**Solution**: Check that post times are in format "YYYY-MM-DD HH:MM:SS". The system now supports multiple formats.

### Issue: API calls failing
**Solution**: 
1. Verify API keys in config.json
2. Check base_url is correct
3. Ensure model name matches API provider
4. Check network connectivity

### Issue: Too many short posts filtered
**Solution**: Adjust minimum length thresholds in `_is_valid_post()` method:
- Reposts: currently 3 characters
- Comments: currently 3 characters  
- Original posts: currently 5 characters

### Issue: Chinese characters not displaying in visualizations
**Solution**: The system automatically configures fonts based on OS. If issues persist:
- Windows: Install Microsoft YaHei font
- macOS: Use default system fonts
- Linux: Install WenQuanYi Micro Hei font

## Performance Tips

1. **Batch Size**: Default is 50. Increase for faster processing (if API rate limits allow)
2. **Concurrency**: Configured in config.json (default: 50)
3. **API Load Balancing**: Add more API endpoints to distribute load
4. **Resume Capability**: System automatically skips already-annotated posts

## Alignment with Simulation

The annotation dimensions are aligned with simulation behavior dimensions in:
`posim/prompts/citizen_prompts/intention_prompts.py`

This ensures consistency between simulated and real data for evaluation purposes.

## Academic Use

This system is designed for academic research:
- Concise, focused code
- Reproducible results
- Comprehensive logging
- All visualizations in English for publication
- Proper error handling and validation

## Citation

If you use this annotation system in your research, please cite:
```
[Your paper citation here]
```

## Support

For issues or questions:
1. Check this guide first
2. Review error messages in console output
3. Check failed_annotations.json for detailed failure reasons
4. Verify configuration in config.json
