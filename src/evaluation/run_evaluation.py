from langsmith import Client
from dotenv import load_dotenv
import os
import asyncio
import pandas as pd
from tabulate import tabulate
import json
from datetime import datetime

from ..main import create_workflow
from .evaluators import (
    evaluate_task_completion, 
    check_node_execution,
    check_image_generation_node
)
from .create_dataset import create_evaluation_dataset

async def run_evaluations():
    # Initialize environment and check API key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：环境变量中未找到 OPENAI_API_KEY")
        return
        
    print("\n🚀 开始评估流程")
    print("==============================")
    
    # Step 1: Dataset Creation/Retrieval
    print("\n1️⃣ 设置测试数据集...")
    dataset = create_evaluation_dataset()
    client = Client()
    print("✓ 数据集已准备好，测试用例：生成带文本叠加的图像")
    
    # Step 2: Workflow Setup
    print("\n2️⃣ 初始化工作流...")
    workflow = create_workflow()
    print("✓ 多智能体工作流已初始化")
    
    # Step 3: Input Preparation
    print("\n3️⃣ 准备输入处理器...")
    def process_request(inputs: dict) -> dict:
        return {
            "messages": [
                {"role": "user", "content": inputs["request"]}
            ],
            "next_agent": None,
            "current_task": None,
            "image_url": None,
            "processed_image_url": None
        }
    print("✓ 输入处理器已准备")
    
    # Step 4: Evaluation Setup
    print("\n4️⃣ 设置评估...")
    target = process_request | workflow
    print("✓ 评估目标已配置")
    
    # Step 5: Run Evaluation
    print("\n5️⃣ 运行多智能体系统评估")
    print("=====================================")
    print("评估三个关键标准：")
    print("1. 任务完成：整体系统性能")
    print("2. 节点执行：智能体交互模式")
    print("3. 单个节点：特定智能体性能")
    
    experiment_results = await client.aevaluate(
        target,
        data=dataset.name,
        evaluators=[
            evaluate_task_completion,
            check_node_execution,
            check_image_generation_node
        ],
        experiment_prefix="image_processing_eval",
        num_repetitions=1,
        max_concurrency=1
    )
    print("✓ 评估完成")
    
    # Step 6: Process Results
    print("\n6️⃣ 处理结果...")
    results_df = experiment_results.to_pandas()
    
    results_dict = {
        "Test Request": {
            "input": results_df['inputs.request'].iloc[0],
            "expected_sequence": results_df['reference.expected_sequence'].iloc[0]
        },
        "Execution Results": {
            "agent_messages": results_df['outputs.messages'].iloc[0],
            "final_state": {
                "next_agent": results_df['outputs.next_agent'].iloc[0],
                "current_task": results_df['outputs.current_task'].iloc[0],
                "image_url": results_df.get('outputs.image_url', ['N/A']).iloc[0],
                "processed_image_url": results_df.get('outputs.processed_image_url', ['N/A']).iloc[0]
            }
        },
        "Evaluation": {
            "task_completion": {
                "score": float(results_df['feedback.evaluate_task_completion'].iloc[0]),
                "reasoning": str(results_df['feedback.evaluate_task_completion'].iloc[0])
            },
            "node_execution": {
                "score": float(results_df['feedback.check_node_execution'].iloc[0]),
                "reasoning": str(results_df['feedback.check_node_execution'].iloc[0])
            },
            "image_generation": {
                "score": float(results_df['feedback.check_image_generation_node'].iloc[0]),
                "reasoning": str(results_df['feedback.check_image_generation_node'].iloc[0])
            },
            "execution_time_seconds": float(results_df['execution_time'].iloc[0])
        }
    }
    
    # Step 7: Display Results
    print("\n7️⃣ 按标准分类的评估结果")
    print("==============================")
    
    print("\n1️⃣ 任务完成评估：")
    print("   整体系统性能分数")
    print(f"分数： {results_dict['Evaluation']['task_completion']['score']}")
    print("分析：")
    print(results_dict['Evaluation']['task_completion']['reasoning'])
    
    print("\n2️⃣ 节点执行分析：")
    print("   智能体交互模式分数")
    print(f"分数： {results_dict['Evaluation']['node_execution']['score']}")
    print("分析：")
    print(results_dict['Evaluation']['node_execution']['reasoning'])
    
    print("\n3️⃣ 图像生成节点检查：")
    print("   单个节点性能分数")
    print(f"分数： {results_dict['Evaluation']['image_generation']['score']}")
    print("分析：")
    print(results_dict['Evaluation']['image_generation']['reasoning'])
    
    # Step 8: Summary
    print("\n8️⃣ 快速概述")
    print("===============")
    print(f"• 请求： {results_dict['Test Request']['input']}")
    print(f"• 任务完成分数： {results_dict['Evaluation']['task_completion']['score']}")
    print(f"• 节点执行分数： {results_dict['Evaluation']['node_execution']['score']}")
    print(f"• 图像生成分数： {results_dict['Evaluation']['image_generation']['score']}")
    print(f"• 执行时间： {results_dict['Evaluation']['execution_time_seconds']:.2f} 秒")
    
    return experiment_results

if __name__ == "__main__":
    asyncio.run(run_evaluations()) 