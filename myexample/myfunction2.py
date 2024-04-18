#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2024/4/17 14:29
# @File  : myfunction2.py
# @Author: 
# @Desc  :
import asyncio
from chatlab import FunctionRegistry


def f(x: float):
    """Multiply x by 2."""
    return x * 2


registry = FunctionRegistry()
registry.register_function(f)


async def main():
    output = await registry.call("f", '{"x": 4}')
    print(output)

if __name__ == '__main__':
    asyncio.run(main())