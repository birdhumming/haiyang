# combination number calculation using base p numbers; 
# mod p with lucas theorem

def pow_mod(a, k, p):
    t = []
    pow_val = 1             # 2 to powers, initialize as 2^0 = 1
    a_pow = a % p           # a^(2 ^ i) value, initialized as a^(2^0) = a
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


def C(a, b, p):
    b = min(b, a-b)

    i, j = 1, a
    val = 1
    while i <= b:
        val = (val * j) % p
        val = (val * pow_mod(i, p-2, p) % p)        # division mod must use reverse multiply mod;
        # direct division is very slow and will get TLE 
        i, j = i+1, j-1

    return val

# lucas theorem https://brilliant.org/wiki/lucas-theorem/
# prime p
def lucas(a, b, p):
    if a < p and b < p:
        return C(a, b, p)
    return (lucas(a%p, b%p, p) * lucas(a//p , b//p, p)) % p


n = int(input())
for _ in range(n):
    a, b, p = map(int, input().split())
    print(lucas(a, b, p)

