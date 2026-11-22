"""
核心参数示例
"""

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from my_tools import GoogleSearchTool, BingSearchTool
#  my_tools ? 项目自己定义的模块

# 创建一个deep_agent
llm = ChatOpenAI(model="gpt-3.5-turbo-0613")
tools = [GoogleSearchTool(), BingSearchTool()]
sys_prompt = """
You are a helpful assistant that can answer questions about the world and do research.
"""

deep_agent = create_deep_agent(
    llm=llm,
    tools=tools,
    system_prompt=sys_prompt,
    verbose=True
)

deep_agent.invoke({
    "message": {"role": "user", "content": "What is the weather in beijing?"}
})


