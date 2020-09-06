AcWing 148. 合并果子-python    原题链接    简单
作者：    在找工作の肖山 ,  2020-01-06 04:45:40 ,  阅读 248

0


升序优先序列，优先pop出最小值
代码
from queue import PriorityQueue
n = int(input())
a = list(map(int,input().split()))

q = PriorityQueue()

for i in range(n):
    q.put(a[i])

sum = 0

while q.qsize()>1:
    a = q.get()
    b = q.get()
    q.put(a+b)
    sum += a+b

print(sum)

作者：在找工作の肖山
链接：https://www.acwing.com/solution/content/7397/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
