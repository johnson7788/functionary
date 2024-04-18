import chatlab
import random
import asyncio

def flip_a_coin():
    '''Returns heads or tails'''
    choice = random.choice(['heads', 'tails'])
    print(f"调用了函数：翻转硬币, 翻转的结果是: {choice}")
    return choice

chat = chatlab.Chat(model="meetkai/functionary-small-v2.4", base_url="http://localhost:8000/v1", api_key="functionary")
chat.register(flip_a_coin)

async def main():
    await chat("我想玩翻转硬币游戏，告诉我翻转硬币的结果:")
    print(chat.messages)

if __name__ == '__main__':
    asyncio.run(main())
