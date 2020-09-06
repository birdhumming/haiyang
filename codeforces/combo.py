# combination number calculation - direct method
# python has support for unlimited integer size
# so just use that to get combo number directly


def C(a, b):
    b = min(b, a-b)
    i, j = 1, a
    val1, val2 = 1, 1
    while i <= b:
        val1 *= i
        val2 *= j
        i, j = i+1, j-1

    return val2 // val1

a, b = map(int, input().split())
print(C(a, b))