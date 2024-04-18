#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/4/17 15:44
# @File  : mycodefunc.py
# @Author: 
# @Desc  : 运行python的函数， 对于meetkai这个模型，不行，没有生成函数的能力
import asyncio
from chatlab import Chat
from chatlab.tools import run_python

chat = Chat(model="meetkai/functionary-small-v2.4", base_url="http://localhost:8000/v1", api_key="functionary",allow_hallucinated_python=True)
chat.register(run_python)

async def main():
    await chat("Please calculate sin(793.1)")
    print(chat.messages)

if __name__ == '__main__':
    asyncio.run(main())