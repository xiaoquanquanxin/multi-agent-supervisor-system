from typing import Dict, Literal
from langgraph.types import Command
from ..agent_types.state import AgentState

def create_image_generation_agent():
    def image_generation_agent(state: AgentState) -> Command[Literal["supervisor"]]:
        print("\n🎨 图像生成智能体：正在处理请求...")
        
        return Command(
            goto="supervisor",
            update={
                "processed_image_url": "mock_generated_image.jpg",
                "messages": state["messages"] + [
                    {"role": "system", "content": "图像生成智能体：已生成新图像"}
                ]
            }
        )
    
    return image_generation_agent 