AcWing 143. 最大异或对python3    原题链接    简单
作者：    xanxus1111 ,  2020-05-05 23:35:18 ,  阅读 104

0




def insert(x):
    global idx
    p = 0
    i = 30
    while i >=0:
        u = x >> i & 1
        if not son[p][u]:
            idx += 1
            son[p][u] = idx
        p = son[p][u]
        i-=1


def query(x):
    p = 0
    i = 30
    res = 0
    while i >= 0:
        u = x >> i & 1
        if son[p][u^1]:
            p = son[p][u^1]
            res = res * 2 + (u^1)
        else:
            p = son[p][u]
            res = res * 2 + u
        i -= 1
    return res

if __name__ == "__main__":
    idx = 0
    res = 0
    n = int(input())
    son =     [[0] * (2) for i in range(n*32)]
    a = list(map(int,input().split()))
    for i in range(n):
        insert(a[i])
        t = query(a[i])
        res = max(res,a[i]^t)

    print(res)


作者：xanxus1111
链接：https://www.acwing.com/solution/content/12778/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
