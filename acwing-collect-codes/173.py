AcWing 173. 矩阵距离 - python 14 行    原题链接    简单
作者：    roon2300 ,  2020-07-19 14:06:38 ,  阅读 34

2


1
n, m = map(int, input().split())

mat = [list(map(int, input())) for _ in range(n)]

dist = [[0 if i else -1  for i in row] for row in mat]

q = [(i, j) for i, row in enumerate(mat) for j, v in enumerate(row) if v == 1]

while q:
    nq = []
    for x, y in q:
        for dx, dy in zip((-1, 0, 1, 0), (0, -1, 0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and mat[nx][ny] == 0 and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                nq.append((nx, ny))
    q = nq


[print(' '.join([str(i) for i in row])) for row in dist]

作者：roon2300
链接：https://www.acwing.com/solution/content/16681/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
