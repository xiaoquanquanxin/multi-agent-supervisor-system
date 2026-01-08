from dotenv import load_dotenv
import os

load_dotenv()

# API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 智能体配置
SUPERVISOR_MODEL = "gpt-4"
SUPERVISOR_TEMPERATURE = 0

# 其他设置可以根据需要在此添加
