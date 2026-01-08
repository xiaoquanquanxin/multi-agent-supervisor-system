"""
多智能体系统评估框架

本模块实现了三个关键评估标准：
1. 任务完成：评估整个多智能体系统是否正确完成了请求的任务
2. 节点执行路径：分析智能体是否按正确顺序执行
3. 单个节点执行：检查特定节点/智能体性能

每个评估器返回一个分数（0.0-1.0）和详细推理。
"""

from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langsmith.schemas import Run, Example
import json

# Initialize judge LLM
judge_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

async def evaluate_task_completion(run: Run, example: Example) -> Dict:
    """
    评估标准1：任务完成
    
    评估多智能体系统作为整体是否正确完成了所有请求的任务。
    考虑因素：
    - 所有必需任务都已完成
    - 任务按逻辑顺序完成
    - 最终输出符合要求
    """
    try:
        # Extract actual sequence from run outputs
        actual_sequence = [
            msg["content"] for msg in run.outputs["messages"] 
            if msg.get("role") == "system" and "Agent:" in msg.get("content", "")
        ]
        
        # Get expected sequence from example
        expected_sequence = example.outputs["expected_sequence"]
        
        # Prepare instructions for the judge
        instructions = """
        您是一个评估判官。给定实际的智能体操作序列和预期序列，
        确定工作流是否正确完成了所有必需的任务。
        
        考虑因素：
        1. 是否执行了所有预期的操作？
        2. 是否按逻辑顺序执行？
        3. 是否发生了意外或不必要的操作？
        
        请回复 '正确' 或 '错误'，然后给出简要解释。
        """
        
        # Prepare the comparison message
        comparison_msg = f"""
        原始请求：{run.inputs.get('request', '未找到请求')}
        
        预期序列：
        {json.dumps(expected_sequence, indent=2)}
        
        实际序列：
        {json.dumps(actual_sequence, indent=2)}
        """
        
        # Get judge's evaluation
        response = await judge_llm.ainvoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": comparison_msg}
            ]
        )
        
        # Parse the response
        is_correct = response.content.upper().startswith("CORRECT")
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "reasoning": response.content
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}"
        }

async def check_node_execution(run: Run, example: Example) -> Dict:
    """
    评估标准2：节点执行路径
    
    分析智能体执行序列以验证正确的工作流。
    检查：
    - 所有必要的智能体都参与了
    - 智能体按正确顺序执行
    - 没有不必要的智能体调用
    """
    try:
        judge_llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Extract agent messages and their sequence
        agent_messages = [
            msg["content"] for msg in run.outputs["messages"] 
            if msg.get("role") == "system" and "Agent:" in msg.get("content", "")
        ]
        
        # Extract agent sequence from messages
        agent_sequence = [
            msg.split("Agent:")[0].strip() 
            for msg in agent_messages
        ]
        
        instructions = """
        You are an evaluation judge analyzing workflow execution. Given the sequence of agent actions,
        determine if:
        
        1. All necessary agents were involved based on the request
        2. The agents executed their tasks in the correct order
        3. The sequence makes logical sense for the task
        
        Respond with either 'CORRECT' or 'INCORRECT', followed by a brief analysis.
        """
        
        comparison_msg = f"""
        Original Request: {run.inputs.get('request', 'No request found')}
        
        EXPECTED WORKFLOW:
        {json.dumps(example.outputs["expected_sequence"], indent=2)}
        
        ACTUAL EXECUTIONS:
        {json.dumps(agent_messages, indent=2)}
        
        Agent Sequence: {json.dumps(agent_sequence, indent=2)}
        """
        
        response = await judge_llm.ainvoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": comparison_msg}
            ]
        )
        
        is_correct = response.content.upper().startswith("CORRECT")
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "reasoning": response.content
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}"
        }


async def check_image_generation_node(run: Run, example: Example) -> Dict:
    """
    评估标准3：单个节点执行
    
    单个节点评估的示例，专注于图像生成智能体。
    验证：
    - 特定智能体是否被调用
    - 智能体参与的简单二进制检查
    """
    try:
        judge_llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Extract messages specifically from Image Generation Agent
        image_gen_messages = [
            msg["content"] for msg in run.outputs.get("messages", [])
            if msg.get("role") == "system" and "Image Generation Agent:" in msg.get("content", "")
        ]
        
        instructions = """
        You are an evaluation judge. Your only task is to check if the Image Generation Agent was called.
        
        Respond with:
        - 'CORRECT' if you see any messages from the Image Generation Agent
        - 'INCORRECT' if there are no messages from the Image Generation Agent
        """
        
        comparison_msg = f"""
        Messages from Image Generation Agent:
        {json.dumps(image_gen_messages, indent=2)}
        """
        
        response = await judge_llm.ainvoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": comparison_msg}
            ]
        )
        
        is_correct = response.content.upper().startswith("CORRECT")
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "reasoning": response.content
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}"
        } 