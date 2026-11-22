import os
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

model = ChatOpenAI(
    
    model=os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4-Flash"),
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)

def get_weather(city: str) -> str:
    """
    Get weather for a given city.
    
    """
    # func必须要有注释 否则 会报错 ValueError: Function must have a docstring if description not provided.
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
)

print(result["messages"][-1].content)
