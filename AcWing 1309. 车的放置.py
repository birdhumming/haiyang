AcWing 1309. 车的放置
皓首不倦的头像皓首不倦
16分钟前
Screen Shot 2020-09-08 at 11.36.16 PM.png


MOD = 100003

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


max_val = 2000
t = [(0, 0)] * (max_val+1)
val = 1
for i in range(1, max_val+1):
    val *= i
    val %= MOD
    _val = pow_mod(val, MOD-2, MOD)     # val的阶乘的逆元
    t[i] = (val, _val)
t[0] = t[1]     # 0! 和 1! 数值是一样的把数值补齐


# a对b的组合数对MOD取模的数值
def C(a, b):
    return (t[a][0] * t[b][1] * t[a-b][1]) % MOD

# a对b排列数对MOD取模的数值
def A(a, b):
    return (t[a][0] * t[a-b][1]) % MOD


a, b, c, d, k = map(int, input().split())
ans = 0
for i in range(k+1):
    if i <= min(a, b) and (k-i) <= min(a+c, d):
        ans += C(b, i) * A(a, i) * C(d, k-i) * A(a+c-i, k-i)
        ans %= MOD
print(ans)



AcWing 1308. 方程的解
皓首不倦的头像皓首不倦
1小时前

'''
利用快速幂求x^x mod 1000 的数值n
利用隔板法，最后答案就是组合数C(n-1, k-1)
(n-1个空隙中选出位置不同的k-1个)
'''
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


import sys
sys.setrecursionlimit(999999)


from functools import lru_cache
@lru_cache(typed=False, maxsize=128000000)
def conbination_num(n, m):
    if m == n:
        return 1

    if m == 0:
        return 1

    if m == 1:
        return n

    return conbination_num(n-1, m-1) + conbination_num(n-1, m)


k, x = map(int, input().split())
n = pow_mod(x, x, 1000)

print(conbination_num(n-1, k-1))