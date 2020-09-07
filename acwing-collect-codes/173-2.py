AcWing 173. 矩阵距离 Python 多源BFS    原题链接    简单
作者：    皓首不倦 ,  2020-08-01 23:40:09 ,  阅读 22

2




'''
多源BFS应用
'''


from collections import deque

m, n = map(int, input().split())
one_pos = []
for i in range(m):
    s = input()
    for j in range(n):
        if s[j] == '1':
            one_pos.append((i, j))


ans = [[0]*n for _ in range(m)]
visited = [[0]*n for _ in range(m)]

que = deque()
for pos in one_pos:
    que.append(pos)
    visited[pos[0]][pos[1]] = 1

step = 0
while len(que) > 0:
    node_num = len(que)
    for _ in range(node_num):
        i, j = que.popleft()
        ans[i][j] = step

        for ii, jj in [ [i-1,j], [i+1, j], [i, j-1], [i, j+1] ]:
            if ii >= 0 and ii < m and jj >= 0 and jj < n and visited[ii][jj] == 0:
                visited[ii][jj] = 1
                que.append((ii, jj))

    step += 1

for i in range(m):
    print(' '.join(map(str, ans[i])))

作者：皓首不倦
链接：https://www.acwing.com/solution/content/17538/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
