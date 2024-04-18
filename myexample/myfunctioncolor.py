import asyncio
import chatlab
from chatlab import Chat, models, system
from chatlab.tools.colors import show_colors

# 另一种注册函数的方式, 好像调用函数出现问题
chat = chatlab.Chat(system("Format responses in markdown,. You are a skilled designer."),
    chat_functions=[show_colors],model="meetkai/functionary-small-v2.4", base_url="http://localhost:8000/v1", api_key="functionary")

async def main():
    await chat("Create a palette for a portfolio site with a dark theme.")
    await chat("Can you make a neon version now?")

if __name__ == '__main__':
    asyncio.run(main())
