from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from langchain_core.runnables.graph import MermaidDrawMethod

# Use relative imports (note the . before agents)
from .agents.supervisor import create_supervisor_agent
from .agents.image_generation import create_image_generation_agent
from .agents.text_overlay import create_text_overlay_agent
from .agents.background_removal import create_background_removal_agent
from .agent_types.state import AgentState

def create_workflow():
    # Create the graph
    builder = StateGraph(AgentState)

    # Add nodes for each agent
    builder.add_node("supervisor", create_supervisor_agent())
    builder.add_node("image_generation", create_image_generation_agent())
    builder.add_node("text_overlay", create_text_overlay_agent())
    builder.add_node("background_removal", create_background_removal_agent())

    # Add starting edge
    builder.add_edge(START, "supervisor")

    graph = builder.compile()
    
    # Generate and save the graph visualization
    graph_png = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API
    )
    
    with open("workflow_graph.png", "wb") as f:
        f.write(graph_png)
    
    print("\n📊 图形可视化已保存为 'workflow_graph.png'")
    
    return graph

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：环境变量中未找到 OPENAI_API_KEY")
        return

    # Create the workflow
    workflow = create_workflow()
    
    # Get user input
    print("\n🤖 图像处理多智能体系统")
    print("----------------------------------------")
    user_instruction = input("\n您希望对图像进行什么操作？\n(例如：'生成一张日落图片并在上面添加文字')\n\n您的请求：")
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=user_instruction)],
        "next_agent": None,
        "current_task": None,
        "image_url": None,
        "processed_image_url": None
    }
    
    print("\n🚀 启动工作流...")
    print("----------------------------------------")
    
    # Execute workflow
    final_state = workflow.invoke(initial_state)
    
    # Print results
    print("\n✨ 工作流完成！")
    print("----------------------------------------")
    print("\n执行路径：")
    for msg in final_state["messages"]:
        # Handle both dict messages and Message objects
        content = msg.content if hasattr(msg, 'content') else msg.get('content', str(msg))
        print(f"- {content}")
    
    print(f"\n最终图像URL：{final_state['processed_image_url']}")

if __name__ == "__main__":
    main() 