from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command

# Simplified imports without src
from ..agent_types.state import AgentState
from ..config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE

def create_supervisor_agent():
    llm = ChatOpenAI(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE
    )
    
    system_prompt = """您是一个协调图像处理任务的监督者智能体。
    根据用户的请求和当前状态，确定下一个应该执行的任务。
    
    可用任务：
    1. image_generation - 当用户需要创建新图像时
    2. text_overlay - 当需要在图像上添加文本时
    3. background_removal - 当需要从图像中移除背景时
    
    规则：
    - 按顺序处理任务，直到所有请求的操作都完成
    - 如果请求提到创建/生成图像，从 'image_generation' 开始
    - 在图像生成后，如果请求了文本/标题，使用 'text_overlay'
    - 如果请求提到移除/删除背景，使用 'background_removal'
    - 只有在所有请求的任务都完成时才回复 '__end__'
    - 在决定下一个任务时，要同时考虑原始请求和当前任务状态
    
    示例序列：
    - "生成一张图片并添加文字" → image_generation → text_overlay → __end__
    - "创建一张图片，移除背景，添加文字" → image_generation → background_removal → text_overlay → __end__
    """

    def supervisor_agent(state: AgentState) -> Command[Literal["image_generation", "text_overlay", "background_removal", "__end__"]]:
        print("\n🎯 监督者智能体：决定下一个任务...")
        
        # Get the initial request if this is the first run
        messages = state["messages"]
        user_request = messages[0]["content"] if isinstance(messages[0], dict) else messages[0].content
        
        # Use LLM to decide next task
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            原始请求： {user_request}
            当前任务： {state["current_task"]}
            
            下一个任务应该是什么？
            """)
        ]
        
        response = llm.invoke(messages).content
        
        # Parse the response to get the next task
        if "image_generation" in response.lower():
            next_agent = "image_generation"
        elif "text_overlay" in response.lower():
            next_agent = "text_overlay"
        elif "background_removal" in response.lower():
            next_agent = "background_removal"
        else:
            next_agent = "__end__"
        
        print(f"➡️ 下一个智能体： {next_agent}")
        
        return Command(
            goto=next_agent,
            update={
                "next_agent": next_agent,
                "current_task": next_agent,
                "messages": state["messages"] + [
                    {"role": "system", "content": f"监督者：路由到 {next_agent}"}
                ]
            }
        )
    
    return supervisor_agent 