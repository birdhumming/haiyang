AcWing 130. 为什么没有python题解呢    原题链接    中等
作者：    CaprYang ,  2020-04-30 17:21:19 ,  阅读 195

4


1
在TLE的边缘徘徊的非正经题解
卡特兰数经典应用，C(2n,n)/(n+1)C(2n,n)/(n+1) 即为答案
本题n很大需要高精度做法，直接使用pythonpython自带的无限精度，注意求阶乘的时候用mathmath库会比手写的快很多

import math

n = int(input())
A = math.factorial(2 * n)
B = math.factorial(n)
print(A // B // B // (n + 1))

作者：CaprYang
链接：https://www.acwing.com/solution/content/9767/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
