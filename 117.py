def main():
    arr = []
    for _ in range(13):
        arr.append(list(input().split()))
    res = [[] for _ in range(13)]
    d = {'0':10,'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'J':11,'Q':12,'K':13}
    m = 4
    def dfs(u,m):
        if m < 0:return
        if u == 'K':
            m = m - 1
            if m <= 0:return
            else:
                u = arr[-1].pop(0)
                dfs(u,m)
        else:
            res[d[u]-1].append(u)
            u = arr[d[u]-1].pop(-1)
            dfs(u,m)
    k = arr[-1].pop(0)
    dfs(k,m)
    count = 0
    for x in res:
        if len(x) == 4:count+=1
    print(count)

main()