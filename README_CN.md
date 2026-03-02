# POSIM: 基于元认知智能体的大规模社交媒体舆情仿真器

POSIM (Public Opinion SIMulator) 是一个由元认知智能体驱动的大规模社交媒体舆情仿真系统。与传统的反应式智能体不同，POSIM采用了增强情感的BDI架构进行基于信念的决策制定，并使用高精度的霍克斯点过程时间引擎来捕捉突发互动行为。

## 特性

- **元认知智能体架构**：感知 → 信念 → 欲望 → 意图 → 行为
- **EBDI认知模型**：整合情感的信念-欲望-意图模型
- **异质性智能体**：支持普通网民、意见领袖、媒体、政府四类智能体
- **霍克斯点过程**：时间驱动的活跃度模拟，支持分钟级仿真
- **社交媒体推荐机制**：模拟微博热门推荐算法
- **热搜榜单机制**：基于热度的话题排名
- **可扩展架构**：模块化设计，支持自定义扩展

## 项目结构

```
posim/
├── posim/                          # 核心算法包
│   ├── agents/                     # 智能体模块
│   │   ├── base_agent.py           # 基础智能体抽象类
│   │   ├── citizen_agent.py        # 普通网民智能体
│   │   ├── kol_agent.py            # 意见领袖智能体
│   │   ├── media_agent.py          # 媒体智能体
│   │   ├── government_agent.py     # 政府智能体
│   │   └── ebdi/                   # EBDI认知模型
│   │       ├── belief/             # 信念系统
│   │       ├── desire/             # 欲望系统
│   │       ├── intention/          # 意图系统
│   │       └── memory/             # 记忆模块
│   ├── environment/                # 环境模块
│   │   ├── recommendation.py       # 推荐系统
│   │   ├── hot_search.py           # 热搜榜单
│   │   ├── social_network.py       # 社交网络
│   │   └── event_queue.py          # 事件队列
│   ├── engine/                     # 仿真引擎
│   │   ├── hawkes_process.py       # 霍克斯点过程
│   │   ├── time_engine.py          # 时间引擎
│   │   └── simulator.py            # 仿真核心
│   ├── llm/                        # 大模型接口
│   ├── prompts/                    # 提示词模板
│   ├── data/                       # 数据处理
│   ├── storage/                    # 存储模块
│   ├── config/                     # 配置管理
│   └── utils/                      # 工具函数
├── scripts/                        # 仿真脚本
│   └── tianjiaerhuan/              # 天价耳环事件示例
├── evaluation/                     # 评估模块
├── output/                         # 输出目录
└── models/                         # 本地模型
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

1. **配置大模型API**

编辑 `scripts/tianjiaerhuan/llm_config.json`，填入你的API密钥：

```json
{
  "llm_api_configs": [
    {
      "name": "你的模型名称",
      "base_url": "https://api.example.com/v1/",
      "api_key": "your-api-key",
      "model": "model-name",
      "enabled": true
    }
  ]
}
```

2. **配置仿真参数**

编辑 `scripts/tianjiaerhuan/config.json` 设置仿真时间范围和参数。

3. **准备数据**

在 `scripts/tianjiaerhuan/data/` 目录下准备：

- `users.json`: 用户数据（包含信念系统）
- `events.json`: 事件队列
- `initial_posts.json`: 初始博文

4. **运行仿真**

```bash
cd scripts/tianjiaerhuan
python run_simulation.py
```

5. **评估结果**

```bash
python -m evaluation.evaluator -e tianjiaerhuan
```

## 数据格式

### 用户数据 (users.json)

```json
{
  "user_id": "citizen_001",
  "username": "用户名",
  "agent_type": "citizen",
  "identity_description": "自然语言的角色身份描述...",
  "psychological_beliefs": ["心理认知信念1", "心理认知信念2"],
  "event_opinions": [
    {"time": "...", "subject": "涉事主体", "opinion": "观点", "reason": "原因"}
  ],
  "emotion_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "history_posts": [{"time": "...", "content": "历史博文内容"}]
}
```

### 事件数据 (events.json)

```json
{
  "time": "2024-09-10T10:00:00",
  "type": "global_broadcast",
  "source": ["external"],
  "content": "事件内容描述",
  "influence": 2.0
}
```

## 干预接口

仿真器提供以下干预接口：

```python
simulator.ban_user(user_id)      # 禁言用户
simulator.unban_user(user_id)    # 解禁用户
simulator.delete_post(post_id)   # 删除博文
simulator.inject_event(...)      # 注入事件
```

## 引用

如果您使用了本项目，请引用：

```
@article{posim2024,
  title={POSIM: A Large-Scale Social Media Public Opinion Simulator with Meta-Cognitive Agents},
  author={...},
  year={2024}
}
```

## 许可证

MIT License
