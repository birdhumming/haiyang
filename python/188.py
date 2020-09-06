AcWing 188. 武士风度的牛 - python 18 行    原题链接    简单
作者：    roon2300 ,  2020-07-18 22:40:30 ,  阅读 33

1


1

c, r = map(int, input().split())

mat = [list(input()) for _ in range(r)]
dist = [[float('inf')] * c for _ in range(r)]

s, t = [(i, j) for i, row in enumerate(mat) for j, v in enumerate(row) if v == 'K' or v == 'H']

dist[s[0]][s[1]] = 0

q = [s]
while q:
    nq = []
    for x, y in q:
        for dx, dy in zip((-2, -1, 1, 2, 2, 1, -1, -2), (-1, -2, -2, -1, 1, 2, 2, 1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) == t:
                print(dist[x][y] + 1)
                exit(0)
            if 0 <= nx < r and 0 <= ny < c and mat[nx][ny] == '.' and dist[nx][ny] > dist[x][y] + 1:
                dist[nx][ny] = dist[x][y] + 1
                nq.append((nx, ny))
    q = nq

作者：roon2300
链接：https://www.acwing.com/solution/content/16665/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
