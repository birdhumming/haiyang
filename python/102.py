AcWing 102. 【Python】最佳牛围栏    原题链接    简单
作者：    tt2767 ,  2019-12-16 00:36:20 ,  阅读 179

1


import sys
import re

def read(raw):
    sbuffer = list()
    for s in raw:
        if re.match('\s', s) is None:
            sbuffer.append(s)
        elif len(sbuffer) > 0:
            yield ''.join(sbuffer)
            sbuffer = list()
    yield ''.join(sbuffer)

pin = read(sys.stdin.read())

N = int(pin.next())
F = int(pin.next())

l = float('inf')
r = float('-inf')

ox = [0] * (N+1)
for i in range(1, N+1):
    ox[i] = int(pin.next())
    l = min(l, ox[i]*1.0)
    r = max(r, ox[i]*1.0)

presum = [0.0]*(N+1)

def is_exist_avg(avg):

    for i in range(1, N+1):
        presum[i] = presum[i-1] + ox[i] - avg

    minimum = 0.0
    print ' '
    for i in range(0, N+1-F):  # 为什么从0 开始？   因为 presum[F] 是 1~F 的和
        minimum = min(minimum, presum[i])
        print '%s-%s-%s,' % (i+F, minimum, presum[i])
        if presum[i+F] >= minimum:
            return True
    return False

while (r -l > 1e-5):  # 为什么是实数域二分？  可能因为平均数为实数？？
                    # 根据 1e-(k+2), k=3 为啥k=3？因为最后要乘1000
    mid = (l + r ) / 2
    if is_exist_avg(mid):
        l = mid
    else:
        r = mid  # 为什么r 也 = mid？ 因为实数域 二分是根据精度来算，取任意分支均可

print int(r * 1000)  # 为什么结果是 r？尝试 l, mid 都比结果小1
                     # 因为不存在比r更大的平均数了，而且由于二分得到的结果，所以向下取整后为最大值

# 为什么答案具有单调性？ 因为结果区间必为 [min(ox), max[ox]] 之间


作者：tt2767
链接：https://www.acwing.com/solution/content/7063/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
