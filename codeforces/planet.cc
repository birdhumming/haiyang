#include <bits/stdc++.h>
using namespace std;
 
const int N = 1010, M = N * 2;
 
int n;
int h[N], e[M], ne[M], idx;
bool st[N];
int d[N];
 
vector<int> cirs;
stack<int> path;
 
void add(int a, int b)
{
e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}
 
bool dfs_cir(int u, int p)
{
st[u] = true;
path.push(u);
 
for (int i = h[u]; ~i; i = ne[i])
{
    int j = e[i];
 
    if (j != p)
    {
        if (st[j])
        {
            // 找出环上的所有点
 
            while (path.top() != j)
            {
                cirs.push_back(path.top());
                path.pop();
            }
 
            cirs.push_back(j);
 
            return true;
        }
 
        if (dfs_cir(j, u)) return true;
    }
}
 
path.pop();
 
return false;
}
 
void dfs_dep(int u, int depth)
{
st[u] = true;
d[u] = depth;
 
for (int i = h[u]; ~i; i = ne[i])
{
    int j = e[i];
    if (!st[j])
        dfs_dep(j, depth + 1);
}
}
 
int main()
{
int T;
cin >> T;
 
for (int C = 1; C <= T; C ++ )
{
    cin >> n;
 
    memset(h, -1, sizeof h);
    idx = 0;
    memset(st, 0, sizeof st);
    cirs.clear();
    path = stack<int>();
 
    for (int i = 0; i < n; i ++ )
    {
        int a, b;
        cin >> a >> b;
        add(a, b);
        add(b, a);
    }
 
    dfs_cir(1, -1);
 
    memset(st, 0, sizeof st);
 
    for (auto c : cirs) st[c] = true;
    for (auto c : cirs) dfs_dep(c, 0);
 
    printf("Case #%d:", C);
    for (int i = 1; i <= n; i ++ ) printf(" %d", d[i]);
    puts("");
}
 
return 0;
}
