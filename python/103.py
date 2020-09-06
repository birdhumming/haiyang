import collections
n = int(input())
c = collections.Counter(list(map(int, input().split())))
k = 3*10**5
ans = -1
curS = -1
m = int(input())
y = list(map(int, input().split()))
z = list(map(int, input().split()))
for i in range(m):
    if c[y[i]] * k + c[z[i]] > curS:
        ans = i
        curS = c[y[i]] * k + c[z[i]]
print(ans + 1)

作者：Gyp
链接：https://www.acwing.com/solution/content/12381/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
