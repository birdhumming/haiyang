AcWing 154. 滑动窗口 - python 13行    原题链接    简单
作者：    roon2300 ,  2020-07-21 15:01:42 ,  阅读 28

2


2
n, k = map(int, input().split())
arr = list(map(int, input().split()))

from collections import deque

def max_min(arr, func):
    q = deque()
    for i, v in enumerate(arr, 1):
        while q and q[0][0] <= i - k: q.popleft()
        while q and func(v, q[-1][1]): q.pop()     # v 在前
        q.append((i, v))
        if i >= k: print(q[0][1], end=' ')
    print("")

max_min(arr, lambda x, y:x < y)     # 单调递增队列
max_min(arr, lambda x, y:x > y)     # 单调递减队列

作者：roon2300
链接：https://www.acwing.com/solution/content/16810/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
