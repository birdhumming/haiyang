AcWing 106. 动态中位数
皓首不倦的头像皓首不倦
52分钟前


'''
对顶堆简单应用
'''

from queue import PriorityQueue
n = int(input())
for i in range(n):
    case_idx, m = map(int, input().split())
    arr = []

    while len(arr) < m:
        d = list(map(int, input().split()))
        arr.extend(d)


    min_heap = PriorityQueue()  # 较小的一堆数值
    max_heap = PriorityQueue()  # 较大的一堆数值

    ans = []
    for idx, val in enumerate(arr):
        if min_heap.qsize() == 0:
            min_heap.put((-val, val))
            ans.append(val)

        else:
            if val <= min_heap.queue[0][1]:
                min_heap.put((-val, val))
            else:
                max_heap.put((val, val))

            if min_heap.qsize() >= max_heap.qsize() + 2:
                _, val = min_heap.get()
                max_heap.put((val, val))
            elif max_heap.qsize() >= min_heap.qsize() + 2:
                _, val = max_heap.get()
                min_heap.put((-val, val))

            if idx % 2 == 0:
                if min_heap.qsize() > max_heap.qsize():
                    ans.append(min_heap.queue[0][1])
                else:
                    ans.append(max_heap.queue[0][1])

    print(case_idx, len(ans))

    for idx, val in enumerate(ans):
        if (idx + 1) % 10 != 0 and idx != len(ans)-1:
            print(val, end=' ')
        else:
            print(val)