"""
测试成员默认值是否只计算一次
"""
from dataclasses import dataclass, field

from altair.utils import data


@dataclass
class MutableType:
    # 针对于可变对象设置默认值会出现表达式只计算一次而出现共享问题
    # def __init__(self, my_set=[]):
    #     self.my_set = my_set



if __name__ == '__main__':
    t1 = MutableType()
    t2 = MutableType()

    t1.my_set.append(1)
    t2.my_set.append(2)

    print(t1.my_set)
    print(t2.my_set)