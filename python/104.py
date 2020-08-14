'''
// is floor divide in python 3

AcWing 104. 货仓选址python3    原题链接    简单
作者：    xanxus1111 ,  2020-07-12 19:36:14 ,  阅读 26

'''


if __name__=='__main__':
    n = int(input())
    q = list(map(int,input().split()))
    q.sort()
    res = 0
    for i in range(n): 
        res += abs(q[i] - q[n//2])

    print(res)
