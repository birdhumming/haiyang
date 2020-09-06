'''
https://stackoverflow.com/questions/13081178/whats-the-difference-on-docstrings-with-triple-single-quotes-and-triple-double
AcWing 125. 耍杂技的牛python3    原题链接    中等
作者：    xanxus1111 ,  2020-07-12 19:57:52 ,  阅读 27

'''


if __name__ == '__main__':
    n = int(input())
    q = []
    for i in range(n):
        w,s = map(int,input().split())
        q.append([w+s,w,s])

    q.sort()
    res = -float('inf')
    sum = 0
    for i in range(n):
        res = max(res,sum - q[i][2])
        sum += q[i][1]
    print(res)

'''
作者：xanxus1111
链接：https://www.acwing.com/solution/content/16201/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''
