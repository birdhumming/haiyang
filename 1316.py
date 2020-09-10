'''
想象构造合法序列的过程是从小到大分配每一个数字到两个队列中，最后要抱枕两个队列中都有n个数，
队列1和队列2内部肯定是有序的，所以只需要保证队列1中某位置i的数值一定小于队列2中i位置的数值
其实只需要保证任何时刻队列1的长度都大于等于队列2长度即可，本质就是卡特兰数的模型，就直接
算卡特兰数即可
'''



from collections import Counter

MAX_VAL = 2000000   # a, b可能出现的最大值

# 线性筛法筛质数
def get_prime(N):
    prime_vals = []
    flag = [True] * (N+1)
    for val in range(2, N+1):
        if flag[val]:
            prime_vals.append(val)

        for p_val in prime_vals:
            if val * p_val > N:
                break

            flag[val * p_val] = False

            if val % p_val == 0:
                break

    return prime_vals

primes = get_prime(MAX_VAL)


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

# 对n阶乘分解质因数
# 返回字典，键是质因子，值是次幂
def devide_prime_n(n):
    c = Counter()
    for val in primes:
        k = 1
        while val ** k <= n:
            c[val] += n // (val ** k)
            k += 1

    return {k: v for k, v in c.items()}

def C(a, b, p):
    c1, c2, c3 = devide_prime_n(a), devide_prime_n(b), devide_prime_n(a-b)
    for k, v in c2.items():
        c1[k] -= v
    for k, v in c3.items():
        c1[k] -= v

    ans = 1
    for k, v in c1.items():
        if v == 0:
            continue

        ans *= pow_mod(k, v, p)
        ans %= p
    return ans

# 返回卡特兰数对p取模结果，p不要求是质数
def get_catalan_num(n, p):
    return (C(2*n, n, p) - C(2*n, n-1, p)) % p

n, p = map(int, input().split())
print(get_catalan_num(n, p))