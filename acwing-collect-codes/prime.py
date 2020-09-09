def gcd(a, b):
    if a > b:
        a, b, = b, a

    while a:
        a, b = b % a, a
    return b


def C3(n):
    return (n * (n-1) * (n-2)) // 6

n, m = map(int, input().split())
ans = C3((m+1)*(n+1)) - (n+1) * C3(m+1) - (m+1) * C3(n+1)


for a in range(1, n+1):
    for b in range(1, m+1):
        ans -= 2 * (gcd(a, b)-1) * (n+1-a) * (m+1-b)

print(ans)