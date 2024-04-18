import chatlab
import random
import asyncio

# 直接问答测试,测试通过
chat = chatlab.Chat(model="meetkai/functionary-small-v2.4", base_url="http://localhost:8000/v1", api_key="functionary")

async def main():
    await chat("你好啊!")
    print(chat.messages)

if __name__ == '__main__':
    asyncio.run(main())
