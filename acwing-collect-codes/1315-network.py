
def C(a, b):
    b = min(b, a-b)

    i, j = 1, a
    val1 = 1
    val2 = 1
    while i <= b:
        val1 *= j
        val2 *= i
        i, j = i+1, j-1

    return val1 // val2

a, b = map(int, input().split())
print(abs(C(a+b,a) - C(a+b, a+1)))