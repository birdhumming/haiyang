AcWing 148. 合并果子 Python3 heap    原题链接    简单
作者：    Gyp ,  2020-04-23 21:45:40 ,  阅读 110

0


import heapq
n = int(input())
a = list(map(int, input().split()))
heapq.heapify(a)
ans = 0
while len(a) > 1:
    x = heapq.heappop(a)
    y = heapq.heappop(a)
    heapq.heappush(a,x+y)
    ans += x+y
print(ans)

作者：Gyp
链接：https://www.acwing.com/solution/content/12027/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
