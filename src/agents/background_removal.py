from typing import Dict, Literal
from langgraph.types import Command
from ..agent_types.state import AgentState

def create_background_removal_agent():
    def background_removal_agent(state: AgentState) -> Command[Literal["supervisor"]]:
        print("\n✂️ 背景移除智能体：正在处理请求...")
        
        return Command(
            goto="supervisor",
            update={
                "processed_image_url": "mock_bg_removed_image.jpg",
                "messages": state["messages"] + [
                    {"role": "system", "content": "背景移除智能体：已移除图像背景"}
                ]
            }
        )
    
    return background_removal_agent 