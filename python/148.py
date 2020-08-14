AcWing 148. 合并果子 python3优先队列法和单调栈法    原题链接    简单
作者：    夏天的梦是什么颜色的呢 ,  2020-04-09 16:52:25 ,  阅读 148

2


算法1
优先队列
from queue import PriorityQueue
from typing import List
class Solution:
    def apple(self, apples: List[int]):
        q = PriorityQueue()
        res = 0
        for i in apples:
            q.put(i)

        while q.qsize() > 1:
            a = q.get()
            b = q.get()
            sum = a+b
            res += sum
            q.put(sum)

        return res

if __name__ == '__main__':
    num = int(input())
    apples = list(map(int, input().split()))
    solution = Solution()
    res = solution.apple(apples)
    print(res)

算法2
单调栈
# 方法二：单调栈
class Solution:
    def apple(self, apples: List[int]):
        stack = []
        res = 0
        apples.sort(reverse=True)
        for i in apples:
            stack.append(i)

        while len(stack)>1:
            a = stack.pop()
            b = stack.pop()
            sum = a+b
            res += sum

            # 接着再把sum推进去
            tmp = []
            while stack and sum>stack[-1]:
                a = stack.pop()
                tmp.append(a)

            # 推进去
            stack.append(sum)
            # tmp中的元素也推进去
            for i in range(len(tmp)-1, -1, -1):
                stack.append(tmp[i])
        return res

if __name__ == '__main__':
    num = int(input())
    apples = list(map(int, input().split()))
    solution = Solution()
    res = solution.apple(apples)
    print(res)

作者：夏天的梦是什么颜色的呢
链接：https://www.acwing.com/solution/content/8515/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
