from typing import Dict, Literal
from langgraph.types import Command
from ..agent_types.state import AgentState

def create_text_overlay_agent():
    def text_overlay_agent(state: AgentState) -> Command[Literal["supervisor"]]:
        print("\n✍️ 文本叠加智能体：正在处理请求...")
        
        return Command(
            goto="supervisor",
            update={
                "processed_image_url": "mock_text_overlay_image.jpg",
                "messages": state["messages"] + [
                    {"role": "system", "content": "文本叠加智能体：已在图像上添加文字"}
                ]
            }
        )
    
    return text_overlay_agent 