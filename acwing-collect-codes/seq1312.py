def pow_mod(a, k, p):
    t = []
    pow_val = 1             # 2的次幂数, 初始是2^0 = 1
    a_pow = a % p           # a^(2 ^ i)的数值, 初始是a^(2^0) = a
    while pow_val <= k:
        t.append(a_pow)
        a_pow = (a_pow*a_pow) % p
        pow_val <<= 1

    ans = 1
    for i in range(len(t)):
        if k & 1:
            ans = (ans * t[i]) % p
        k >>= 1
    return ans


max_val = 1000003
MOD = 1000003
t = [(0, 0)] * (max_val+1)
val = 1
for i in range(1, max_val+1):
    val *= i
    val %= MOD
    _val = pow_mod(val, MOD-2, MOD)     # val的阶乘的逆元
    t[i] = (val, _val)
t[0] = t[1]     # 0! 和 1! 数值是一样的把数值补齐

# a对b的组合数对MOD取模的数值
def C(a, b, p):
    return (t[a][0] * t[b][1] * t[a-b][1]) % p


# 卢卡斯定理变形
def lucas(a, b, p):
    if a < b:
        return 0

    if a < p and b < p:
        return C(a, b, p)
    return (lucas(a%p, b%p, p) * lucas(a//p , b//p, p)) % p


n = int(input())
for _ in range(n):
    N, L, R = map(int, input().split())
    print((lucas(R-L+N+1, R-L+1, MOD) - 1) % MOD)