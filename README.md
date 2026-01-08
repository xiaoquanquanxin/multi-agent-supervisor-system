# 多智能体图像处理系统

基于 LangGraph 的系统，使用智能体-监督者模式协调多个 AI 智能体执行图像处理任务。

## 概述

本项目实现了基于 LangGraph 智能体-监督者模式的多智能体系统，其中监督者智能体协调多个专门的图像处理智能体。该系统展示了 LangGraph 的无边图架构和 Command 构造在智能体协调中的使用。

### 系统架构

![工作流图](workflow_graph.png)

系统包含：

1. **监督者智能体**
   - 协调工作流程
   - 对任务排序做出智能决策
   - 使用 LangGraph 的 Command 构造将请求路由到适当的智能体

2. **任务智能体**
   - 图像生成智能体：处理图像创建请求
   - 文本叠加智能体：在图像上添加文本
   - 背景移除智能体：移除图像背景

上面的图形可视化显示了：
- 连接到监督者的初始入口点（START）
- 协调所有任务智能体的监督者节点
- 用于特定图像处理操作的任务智能体节点
- 基于用户请求通过系统的潜在路径

### 使用的关键 LangGraph 特性

1. **无边图架构**
   - 不使用节点间的显式边，而是通过智能体命令处理路由
   - 每个智能体返回一个指定下一个要运行的智能体的命令
   - 简化图结构并使其更加灵活

2. **Command 构造**
   ```python
   Command(
       goto="next_agent",
       update={
           "next_agent": "next_agent",
           "current_task": "current_task",
           "messages": [...],
       }
   )
   ```
   - `goto`：指定要执行的下一个智能体
   - `update`：更新在智能体之间传递的状态

3. **StateGraph**
   ```python
   builder = StateGraph(AgentState)
   builder.add_node("supervisor", create_supervisor_agent())
   builder.add_edge(START, "supervisor")
   ```
   - 管理智能体之间的状态转换
   - 只需要从 START 到监督者的初始边

## 安装

1. 创建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # Windows 系统：venv\Scripts\activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 创建包含 OpenAI API 密钥的 `.env` 文件：

```
OPENAI_API_KEY=your-key-here
```

## 运行系统

在项目根目录下运行：

```bash
python -m src.main
```

示例输入：
- "生成一张日落图片并在上面添加'美丽的夜晚'文字"
- "创建一张山景图片并移除背景"
- "生成一张猫的图片并添加'你好'文字"

系统将：
1. 接收您的输入
2. 使用监督者确定任务序列
3. 通过适当的智能体路由请求
4. 显示执行路径和最终结果

## 评估框架

系统包含一个评估框架，用于评估多智能体工作流的性能和正确性。

有关评估框架的详细信息，请参阅[评估文档](src/evaluation/README.md)。

## 项目结构

```
image_processing_agents/
├── src/
│   ├── agents/
│   │   ├── supervisor.py      # 监督者智能体实现
│   │   ├── image_generation.py
│   │   ├── text_overlay.py
│   │   └── background_removal.py
│   ├── evaluation/          # 评估框架
│   │   ├── evaluators.py    # 评估函数
│   │   ├── create_dataset.py # 测试数据集创建
│   │   └── run_evaluation.py # 主评估脚本
│   ├── agent_types/
│   │   └── state.py          # 状态类型定义
│   ├── config/
│   │   └── settings.py       # 配置设置
│   └── main.py              # 主执行脚本
├── .env                     # 环境变量
├── .gitignore
└── requirements.txt
```

## 实现细节

1. **状态管理**
   - 使用 TypedDict 进行类型安全的状态管理
   - 跟踪消息、当前任务和图像 URL
   - 维护执行历史

2. **智能体通信**
   - 智能体通过状态更新进行通信
   - 每个智能体将其操作添加到消息历史中
   - 监督者基于完整上下文做出决策

3. **路由逻辑**
   - 监督者分析原始请求和当前状态
   - 对任务执行做出顺序决策
   - 使用 LLM 理解复杂请求

## 基于
本实现遵循 LangGraph 智能体-监督者教程：
[LangGraph 多智能体教程](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)

### 评估组件

1. **测试数据集**
   - 预定义的测试用例和预期结果
   - 存储在 LangSmith 中用于跟踪和分析

2. **LLM 评判器 (GPT-4)**
   - 评估任务完成准确性
   - 分析智能体执行模式
   - 为分数提供详细推理

3. **指标**
   - 任务完成分数 (0.0 - 1.0)
   - 节点执行分数 (0.0 - 1.0)
   - 执行时间

4. **结果存储**
   - 评估结果存储在 LangSmith 中
   - 智能体交互的详细日志
   - 性能指标和分析