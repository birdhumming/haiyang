题解AcWing 107. 超快速排序
皓首不倦的头像皓首不倦
1小时前

'''
归并排序求逆序对个数即可
'''


import bisect

def merge_sort(arr):
    if len(arr) == 1:
        return arr.copy(), 0

    mid = (len(arr)-1) // 2
    arr1, cnt1 = merge_sort(arr[:mid+1])
    arr2, cnt2 = merge_sort(arr[mid+1:])

    cnt = cnt1 + cnt2
    for val in arr1:
        idx = bisect.bisect_left(arr2, val)
        if idx - 1 >= 0 and idx - 1 < len(arr2):
            cnt += idx

    arr.sort()
    ret = arr.copy()
    ret.sort()
    return ret, cnt


while True:
    n = int(input())
    if n == 0:
        break

    arr = [0] * n
    for i in range(n):
        arr[i] = int(input())

    print(merge_sort(arr)[1])




    AcWing 105. 七夕祭
皓首不倦的头像皓首不倦
2小时前

'''
行列变换互不影响，转换成环形纸牌分配问题，排序找中位数
'''

m, n, p = map(int, input().split())

row_cnt = [0] * m   # 每一行的点数量
col_cnt = [0] * n   # 每一列的点数量
for _ in range(p):
    a, b = map(int, input().split())
    a, b = a-1, b-1
    row_cnt[a] += 1
    col_cnt[b] += 1

if p % m != 0 and p % n != 0:
    print('impossible')

else:
    cnt1, cnt2 = None, None
    if p % m == 0:
        mean = p // m
        arr = [0]
        S1, S2 = 0, 0
        for i in range(m-1):
            S1 += row_cnt[i]
            S2 += mean
            arr.append(S1-S2)

        arr.sort()
        mid_val = arr[m//2]         # 中位数
        cnt1 = 0
        for val in arr:
            cnt1 += abs(val - mid_val)

    if p % n == 0:
        mean = p // n
        arr = [0]
        S1, S2 = 0, 0
        for i in range(n - 1):
            S1 += col_cnt[i]
            S2 += mean
            arr.append(S1 - S2)

        arr.sort()
        mid_val = arr[n // 2]       # 中位数
        cnt2 = 0
        for val in arr:
            cnt2 += abs(val - mid_val)

    if cnt2 is None:
        print(f'row {cnt1}')
    elif cnt1 is None:
        print(f'column {cnt2}')
    else:
        print(f'both {cnt1 + cnt2}')

