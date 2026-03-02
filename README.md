# POSIM: 基于元认知智能体的大规模社交媒体舆情仿真系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**POSIM** (Public Opinion SIMulator) 是一个由元认知智能体驱动的大规模社交媒体舆情仿真系统。与传统的反应式智能体不同,POSIM采用了增强情感的BDI (Belief-Desire-Intention) 架构进行基于信念的决策制定,并使用高精度的霍克斯点过程时间引擎来捕捉突发互动行为。

## 🌟 核心特性

### 1. **元认知智能体架构**
- **完整认知流程**: 感知 → 信念 → 欲望 → 意图 → 行为
- **EBDI认知模型**: 整合情感的信念-欲望-意图模型
- **四类信念系统**:
  - **身份信念**: 自然语言描述的角色身份
  - **心理信念**: 心理认知信念列表
  - **事件信念**: 对涉事主体的观点和原因
  - **情绪信念**: 6维情绪向量 (愤怒、厌恶、恐惧、快乐、悲伤、惊讶)

### 2. **异质性智能体**
支持四类不同特征的智能体:

| 智能体类型 | 特征 | 行为模式 |
|-----------|------|---------|
| **普通网民** (Citizen) | 行为轻量化、情绪化程度高、易受热点影响 | 单次最多5个行为 |
| **意见领袖** (KOL) | 高影响力、注重个人品牌、倾向发布原创内容 | 单次最多3个行为 |
| **媒体** (Media) | 专业客观、以发布新闻为主、注重公信力 | 单次最多2个行为 |
| **政府** (Government) | 语言正式规范、发布官方通报、引导舆论 | 单次最多1个行为 |

### 3. **霍克斯点过程时间引擎**
- **公式**: λ(t) = circadian_factor(t) × [μ + Σ α × w_i × exp(-β × (t - t_i))]
- **特性**:
  - 时间驱动的活跃度模拟
  - 支持分钟级仿真精度
  - 昼夜节律调节
  - 事件自激励机制
  - 指数衰减影响力

### 4. **社交媒体环境模拟**
- **推荐系统**: S_exp = α·Homophily + β·Popularity + γ·Recency + δ·Relation
- **热搜榜单**: 基于热度的话题排名机制
- **社交网络**: 支持粉丝关注网络、转发网络、评论网络
- **事件队列**: 外部事件注入和管理

### 5. **可扩展架构**
- 模块化设计,各组件独立可替换
- 支持自定义智能体类型
- 支持自定义提示词模板
- 支持多种LLM API
- 可选Neo4j图数据库支持

---

## 📁 项目结构

```
posim/
├── posim/                          # 核心算法包 (49个Python文件)
│   ├── agents/                     # 智能体模块
│   │   ├── base_agent.py           # 基础智能体抽象类 (元认知架构)
│   │   ├── citizen_agent.py        # 普通网民智能体
│   │   ├── kol_agent.py            # 意见领袖智能体
│   │   ├── media_agent.py          # 媒体智能体
│   │   ├── government_agent.py     # 政府智能体
│   │   └── ebdi/                   # EBDI认知模型
│   │       ├── belief/             # 信念系统
│   │       │   ├── belief_system.py        # 信念系统主类
│   │       │   ├── belief_updater.py       # 信念更新器
│   │       │   ├── identity_belief.py      # 角色身份信念
│   │       │   ├── psychological_belief.py # 心理认知信念
│   │       │   ├── event_belief.py         # 事件观点信念
│   │       │   └── emotion_belief.py       # 情绪激发信念
│   │       ├── desire/             # 欲望系统
│   │       │   ├── desire_system.py        # 欲望系统
│   │       │   └── desire_types.py         # 欲望类型定义
│   │       ├── intention/          # 意图系统
│   │       │   ├── intention_system.py     # 意图系统 (三级COT决策)
│   │       │   ├── action_selector.py      # 行为选择器
│   │       │   └── strategy_selector.py    # 策略选择器
│   │       └── memory/             # 记忆模块
│   │           ├── stream_memory.py        # 流式记忆
│   │           └── memory_retrieval.py     # 记忆检索
│   │
│   ├── environment/                # 环境模块
│   │   ├── recommendation.py       # 推荐系统
│   │   ├── hot_search.py           # 热搜榜单
│   │   ├── social_network.py       # 社交网络
│   │   └── event_queue.py          # 事件队列
│   │
│   ├── engine/                     # 仿真引擎
│   │   ├── hawkes_process.py       # 霍克斯点过程
│   │   ├── time_engine.py          # 时间引擎
│   │   └── simulator.py            # 仿真核心
│   │
│   ├── llm/                        # 大模型接口
│   │   ├── api_pool.py             # API池管理
│   │   └── llm_client.py           # LLM客户端
│   │
│   ├── prompts/                    # 提示词模板
│   │   ├── citizen_prompts/        # 普通网民提示词
│   │   ├── kol_prompts/            # KOL提示词
│   │   ├── media_prompts/          # 媒体提示词
│   │   ├── government_prompts/     # 政府提示词
│   │   └── prompt_loader.py        # 提示词加载器
│   │
│   ├── data/                       # 数据处理
│   ├── storage/                    # 存储模块
│   ├── config/                     # 配置管理
│   ├── evaluation/                 # 评估模块
│   └── utils/                      # 工具函数
│
├── scripts/                        # 仿真脚本
│   └── tianjiaerhuan/              # 天价耳环事件示例
│       ├── config.json             # 仿真配置文件
│       └── data/                   # 数据目录
│           ├── users.json          # 用户数据
│           ├── events.json         # 事件队列
│           ├── initial_posts.json  # 初始博文
│           └── relations.json      # 关系数据
│
├── evaluation/                     # 评估模块
├── models/                         # 本地模型
├── output/                         # 输出目录
├── 参数调整说明.md                 # 参数调整文档
└── 霍克斯过程计算公式说明.md       # 霍克斯过程公式文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖**:
- `numpy>=1.24.0` - 数值计算
- `openai>=1.0.0` - LLM API调用
- `pydantic>=2.0.0` - 数据验证
- `sentence-transformers>=2.2.0` - 文本嵌入
- `neo4j>=5.0.0` - 图数据库 (可选)
- `matplotlib>=3.7.0` - 可视化
- `torch>=2.0.0` - 深度学习框架

### 2. 配置大模型API

编辑 `scripts/tianjiaerhuan/config.json`,填入你的API密钥:

```json
{
  "llm": {
    "llm_api_configs": [
      {
        "name": "default",
        "enabled": true,
        "base_url": "https://api.example.com/v1/",
        "api_key": "your-api-key",
        "model": "model-name",
        "temperature": 0.7,
        "top_p": 0.9,
        "weight": 1.0
      }
    ]
  }
}
```

### 3. 配置仿真参数

编辑 `scripts/tianjiaerhuan/config.json`:

```json
{
  "simulation": {
    "event_name": "your_event_name",
    "start_time": "2025-05-14T10:00",
    "end_time": "2025-05-21T22:00",
    "time_granularity": 600,
    "participant_scale": 1000,
    "hawkes_mu": 0.05,
    "hawkes_alpha": 0.01,
    "hawkes_beta": 0.01,
    "hawkes_activation_scale": 6.0,
    "use_llm": true
  }
}
```

### 4. 准备数据

在 `scripts/tianjiaerhuan/data/` 目录下准备以下文件:

#### users.json - 用户数据
```json
{
  "user_id": "citizen_001",
  "username": "用户名",
  "agent_type": "citizen",
  "followers_count": 1000,
  "identity_description": "自然语言的角色身份描述...",
  "psychological_beliefs": ["心理认知信念1", "心理认知信念2"],
  "event_opinions": [
    {
      "time": "2025-05-14T10:00",
      "subject": "涉事主体",
      "opinion": "观点",
      "reason": "原因"
    }
  ],
  "emotion_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "history_posts": [
    {"time": "2025-05-01T12:00", "content": "历史博文内容"}
  ]
}
```

#### events.json - 事件队列
```json
{
  "time": "2025-05-14T10:00",
  "type": "global_broadcast",
  "source": ["external"],
  "content": "事件内容描述",
  "influence": 10.0
}
```

#### initial_posts.json - 初始博文
```json
{
  "post_id": "post_001",
  "user_id": "citizen_001",
  "content": "博文内容",
  "time": "2025-05-14T10:00",
  "likes": 0,
  "reposts": 0,
  "comments": []
}
```

### 5. 运行仿真

```bash
cd scripts/tianjiaerhuan
python run_simulation.py
```

### 6. 查看结果

仿真结果将保存在 `output/` 目录下,包括:
- 仿真日志
- 智能体行为记录
- 博文数据
- 热度演变曲线
- 评估报告

---

## 📊 霍克斯过程详解

### 核心公式

```
λ(t) = circadian_factor(t) × [μ + Σ α × w_i × exp(-β × (t - t_i))]
```

**物理意义**: t时刻系统的瞬时活跃强度,决定了该时刻有多少智能体会被激活。

### 参数说明

| 参数 | 符号 | 默认值 | 含义 | 调参建议 |
|------|------|--------|------|----------|
| `hawkes_mu` | μ | 0.05 | **基础强度**: 背景活跃率 | 增大→整体更活跃 |
| `hawkes_alpha` | α | 0.01 | **激励强度**: 事件对活跃度的影响 | 增大→事件影响更强 |
| `hawkes_beta` | β | 0.01 | **衰减率**: 影响力衰减速度 (1/分钟) | 增大→影响快速消退 |
| `hawkes_activation_scale` | scale | 6.0 | **激活系数**: 控制激活数量 | 增大→更多智能体激活 |
| `event_influence_scale` | - | 1.0 | **事件影响力缩放**: 放大外部事件影响 | 建议保持1.0 |

### 量纲统一说明

**重要**: 所有时间相关的计算都统一为**分钟**单位:

- `time_granularity`: 600秒 = **10分钟** (每步仿真的时间间隔)
- `beta`: 单位为 **1/分钟** (衰减率)
- 时间差 `delta`: **分钟** (当前时间 - 事件发生时间)

### 衰减特性

以 β = 0.01 为例:

| 时间差 | exp(-0.01×t) | 剩余影响 |
|--------|--------------|----------|
| 10分钟 | 0.905 | 90.5% |
| 30分钟 | 0.741 | 74.1% |
| 60分钟 | 0.549 | 54.9% |
| 120分钟 | 0.301 | 30.1% |
| 180分钟 | 0.165 | 16.5% |
| 300分钟 | 0.050 | 5.0% |

**结论**: β = 0.01 时,事件影响可持续3-5小时,符合真实社交媒体传播规律。

### 调参指南

#### 增加整体活跃度
```json
{
  "hawkes_mu": 0.1,
  "hawkes_activation_scale": 8.0
}
```

#### 增强事件影响力
```json
{
  "hawkes_alpha": 0.02,
  "event_influence_scale": 1.5
}
```

#### 延长事件影响时间
```json
{
  "hawkes_beta": 0.005
}
```

#### 缩短事件影响时间
```json
{
  "hawkes_beta": 0.02
}
```

---

## 🎯 EBDI认知模型

### 信念系统 (Belief)

#### 1. 身份信念 (Identity Belief)
自然语言描述的角色身份:
```
"我是一名关注社会公平的大学生,对社会不公现象比较敏感..."
```

#### 2. 心理信念 (Psychological Belief)
心理认知信念列表:
```python
[
  "我倾向于相信官方媒体的报道",
  "我对网络谣言保持警惕",
  "我认为应该理性看待社会事件"
]
```

#### 3. 事件信念 (Event Belief)
对涉事主体的观点:
```python
{
  "time": "2025-05-14T10:00",
  "subject": "黄杨钿甜",
  "opinion": "质疑",
  "reason": "成人礼佩戴天价耳环,与其家庭背景不符"
}
```

#### 4. 情绪信念 (Emotion Belief)
6维情绪向量 [愤怒, 厌恶, 恐惧, 快乐, 悲伤, 惊讶]:
```python
[0.7, 0.3, 0.0, 0.0, 0.2, 0.5]
```

### 欲望系统 (Desire)

基于信念和环境感知生成欲望集:

| 欲望类型 | 强度等级 | 数值映射 |
|---------|---------|---------|
| 表达观点 | 极低/低/中等/高/极高 | 0.1/0.3/0.5/0.7/0.9 |
| 获取信息 | 极低/低/中等/高/极高 | 0.1/0.3/0.5/0.7/0.9 |
| 社交互动 | 极低/低/中等/高/极高 | 0.1/0.3/0.5/0.7/0.9 |

### 意图系统 (Intention)

通过**单次LLM调用**使用**三级COT**生成行为决策列表:

#### 行为类型
- `like`: 点赞
- `repost`: 转发
- `repost_comment`: 转发评论
- `short_comment`: 短评论 (<50字)
- `long_comment`: 长评论 (≥50字)
- `short_post`: 短博文 (<100字)
- `long_post`: 长博文 (≥100字)

#### 策略维度
- **情感**: 正面/中性/负面
- **立场**: 支持/中立/反对
- **强度**: 温和/适中/激烈

---

## 🌐 环境模块

### 推荐系统

**公式**:
```
S_exp = α·Homophily + β·Popularity + γ·Recency + δ·Relation
```

**参数**:
- `homophily_weight` (α): 同质性权重 (默认0.4)
- `popularity_weight` (β): 热度权重 (默认0.3)
- `recency_weight` (γ): 时效性权重 (默认0.3)
- `relation_weight` (δ): 关系权重 (默认0.5)

### 热搜榜单

基于热度的话题排名机制:
- `hot_search_update_interval`: 更新间隔 (默认15分钟)
- `hot_search_count`: 榜单数量 (默认50)

### 社交网络

支持三种网络类型:
1. **粉丝关注网络**: 静态关注关系
2. **实时转发网络**: 动态转发传播
3. **实时评论网络**: 动态评论互动

可选Neo4j图数据库支持,用于大规模网络分析。

---

## 🔧 干预接口

仿真器提供以下干预接口,用于模拟平台管理和政策干预:

```python
# 禁言用户
simulator.ban_user(user_id)

# 解禁用户
simulator.unban_user(user_id)

# 删除博文
simulator.delete_post(post_id)

# 注入外部事件
simulator.inject_event(
    time="2025-05-15T10:00",
    content="事件内容",
    influence=50.0,
    event_type="global_broadcast"
)
```

---

## 📈 评估指标

### 宏观指标
- 热度演变曲线
- 参与度分布
- 话题传播速度
- 舆情极性变化

### 微观指标
- 用户行为统计
- 观点分布
- 情绪演变
- 网络结构特征

---

## 🔍 霍克斯过程设计问题与优化

### 已识别的问题

#### 1. ~~量纲不统一~~ (已解决)
- **问题**: 早期版本中 `time_granularity` 单位混乱
- **解决方案**: 统一为秒数,代码中自动转换为分钟
- **当前状态**: ✅ 已统一,`time_granularity=600秒=10分钟`

#### 2. ~~参数冗余~~ (已优化)
- **问题**: `event_influence_scale` 参数冗余
- **解决方案**: 设置为1.0,直接在 `events.json` 中设置合理的 `influence` 值
- **当前状态**: ✅ 已优化,建议 `influence` 范围为 5-80

#### 3. ~~衰减过快~~ (已修复)
- **问题**: 早期 `beta=0.5` 导致事件影响30分钟后消失
- **解决方案**: 降低为 `beta=0.01`,延长影响时间至3-5小时
- **当前状态**: ✅ 已修复

### 当前推荐参数

```json
{
  "hawkes_mu": 0.05,
  "hawkes_alpha": 0.01,
  "hawkes_beta": 0.01,
  "hawkes_activation_scale": 6.0,
  "event_influence_scale": 1.0,
  "time_granularity": 600
}
```

### events.json 中的 influence 建议值

| 事件类型 | influence 值 | 说明 |
|---------|-------------|------|
| 初始事件 | 10-15 | 引发关注 |
| 网友爆料 | 40-60 | 第一波高峰 |
| 官方回应 | 25-35 | 中等热度 |
| 政府介入 | 60-80 | 最高峰 |
| 持续发酵 | 15-25 | 持续关注 |
| 次要事件 | 5-10 | 小规模讨论 |

---

## 🛠️ 高级配置

### 昼夜节律曲线

模拟真实用户作息规律:

```json
{
  "circadian_curve": {
    "0": 0.4, "1": 0.3, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.3,
    "6": 0.5, "7": 0.7, "8": 0.9, "9": 1.0, "10": 1.0, "11": 0.95,
    "12": 0.85, "13": 0.8, "14": 0.85, "15": 0.9, "16": 0.95, "17": 1.0,
    "18": 1.1, "19": 1.15, "20": 1.2, "21": 1.2, "22": 1.1, "23": 0.8
  }
}
```

**特点**:
- 凌晨 (0-5时): 0.2-0.4 (低活跃)
- 早晨 (6-9时): 0.5-1.0 (逐渐上升)
- 上午 (10-11时): 0.95-1.0 (高峰)
- 中午 (12-13时): 0.8-0.85 (午休下降)
- 下午 (14-17时): 0.85-1.0 (恢复上升)
- 晚上 (18-22时): 1.0-1.2 (最高峰)
- 深夜 (23时): 0.8 (开始下降)

### 行为权重

不同行为类型的影响力权重:

```json
{
  "action_weights": {
    "like": 0.1,
    "short_comment": 0.3,
    "long_comment": 0.5,
    "short_post": 0.6,
    "repost_comment": 0.8,
    "long_post": 0.8,
    "repost": 1.0
  }
}
```

### 本地Embedding模型

支持本地embedding模型,避免API调用:

```json
{
  "llm": {
    "use_local_embedding_model": true,
    "local_embedding_model_path": "models/bge-small-zh-v1.5",
    "embedding_dimension": 512,
    "embedding_device": "cpu"
  }
}
```

---

## 📚 文档

- [参数调整说明](参数调整说明.md) - 详细的参数调整指南
- [霍克斯过程计算公式说明](霍克斯过程计算公式说明.md) - 霍克斯过程数学推导

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议!

### 开发指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📖 引用

如果您使用了本项目,请引用:

```bibtex
@article{posim2025,
  title={POSIM: A Large-Scale Social Media Public Opinion Simulator with Meta-Cognitive Agents},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和研究者!

---

## 📧 联系方式

如有问题或建议,请通过以下方式联系:

- 提交 Issue: [GitHub Issues](https://github.com/yourusername/posim/issues)
- 邮箱: your.email@example.com

---

**注意**: 本项目仅用于学术研究和教育目的,请勿用于非法用途。
