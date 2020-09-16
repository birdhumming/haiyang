https://www.acwing.com/blog/content/92/

https://www.bilibili.com/video/av47271195/

第十届蓝桥杯解题代码——Java B组
作者：    yxc ,  2019-03-24 23:56:38 ,  阅读 2458

14


5
讲解视频在b站。

试题E：迷宫
C++ 代码
```
#include <cstring>
#include <iostream>
#include <algorithm>
#include <set>
#include <queue>

using namespace std;

const int N = 55;

int n, m;
string g[N];
int dist[N][N];
int dx[4] = {1, 0, 0, -1}, dy[4] = {0, -1, 1, 0};
char dir[4] = {'D', 'L', 'R', 'U'};

void bfs()
{

    queue<pair<int,int>> q;
    memset(dist, -1, sizeof dist);
    dist[n - 1][m - 1] = 0;
    q.push({n - 1, m - 1});
    while (q.size())
    {
        auto t = q.front();
        q.pop();

        for (int i = 0; i < 4; i ++ )
        {
            int x = t.first + dx[i], y = t.second + dy[i];
            if (x >= 0 && x < n && y >= 0 && y < m && dist[x][y] == -1 && g[x][y] == '0')
            {
                dist[x][y] = dist[t.first][t.second] + 1;
                q.push({x, y});
            }
        }
    }
}

int main()
{
    cin >> n >> m;
    for (int i = 0; i < n; i ++ ) cin >> g[i];

    bfs();

    cout << dist[0][0] << endl;

    int x = 0, y = 0;
    string res;
    while (x != n - 1 || y != m - 1)
    {
        for (int i = 0; i < 4; i ++ )
        {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && g[nx][ny] == '0')
            {
                if (dist[x][y] == 1 + dist[nx][ny])
                {
                    x = nx, y = ny;
                    res += dir[i];
                    break;
                }
            }
        }
    }

    cout << res << endl;

    return 0;
}
试题F：特别数的和
C++代码
#include <cstring>
#include <iostream>
#include <algorithm>
#include <set>
#include <queue>

using namespace std;

const int N = 55;

bool check(int number)
{
    while (number)
    {
        int t = number % 10;
        if (t == 2 || t == 0 || t == 1 || t == 9) return true;
        number /= 10;
    }
    return false;
}

int main()
{
    int n;
    cin >> n;
    int res = 0;
    for (int i = 1; i <= n; i ++ )
    {
        if (check(i)) res += i;
    }

    cout << res << endl;
    return 0;
}
试题G：外卖店优先级
C++ 代码
#include <cstring>
#include <iostream>
#include <algorithm>
#include <set>
#include <queue>

using namespace std;

const int N = 100010;

int n, m, T;
int last[N], score[N];
bool st[N];
pair<int, int>orders[N];

int main()
{
    cin >> n >> m >> T;
    for (int i = 0; i < m; i ++ ) cin >> orders[i].first >> orders[i].second;
    sort(orders, orders + m);

    for (int i = 0; i < m && orders[i].first <= T; i ++ )
    {
        int j = i;
        while (j < m && orders[i] == orders[j]) j ++ ;
        int t = orders[i].first, id = orders[i].second;

        score[id] -= t - last[id] - 1;
        if (score[id] < 0) score[id] = 0;
        if (score[id] <= 3) st[id] = false;
        last[id] = t;

        score[id] += (j - i) * 2;
        if (score[id] > 5) st[id] = true;

        i = j - 1;
    }

    for (int i = 1; i <= n; i ++ )
        if (last[i] < T)
        {
            score[i] -= T - last[i];
            if (score[i] <= 3) st[i] = false;
        }

    int res = 0;
    for (int i = 1; i <= n; i ++ ) res += st[i];
    cout << res << endl;

    return 0;
}
试题H：人物相关性分析
C++ 代码：
#include <cstring>
#include <iostream>
#include <algorithm>
#include <set>
#include <queue>
#include <vector>

using namespace std;

const int N = 100010;

string str;
vector<int> a, b;

bool check(char c)
{
    return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z';
}

int main()
{
    int k;
    cin >> k;
    getchar();
    getline(cin, str);

    for (int i = 0; i + 5 <= str.size(); i ++ )
        if ((!i || !check(str[i - 1])) && (i + 5 == str.size() || !check(str[i + 5])))
        {
            if (str.substr(i, 5) == "Alice") a.push_back(i);
        }
    for (int i = 0; i + 3 <= str.size(); i ++ )
        if ((!i || !check(str[i - 1])) && (i + 3 == str.size() || !check(str[i + 3])))
        {
            if (str.substr(i, 3) == "Bob") b.push_back(i);
        }

    long long res = 0;
    for (int i = 0, l = 0, r = -1; i < a.size(); i ++ )
    {
        while (r + 1 < b.size() && a[i] >= b[r + 1]) r ++ ;
        while (l <= r && a[i] - 1 - (b[l + 1] + 3) + 1 > k) l ++ ;
        res += r - l + 1;
    }

    for (int i = 0, l = 0, r = -1; i < b.size(); i ++ )
    {
        while (r + 1 < a.size() && b[i] >= a[r + 1]) r ++ ;
        while (l <= r && b[i] - 1 - (a[l + 1] + 5) + 1 > k) l ++ ;
        res += r - l + 1;
    }

    cout << res << endl;

    return 0;
}
试题I：后缀表达式
C++ 代码
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 200010;

int n, m;
int a[N];

int main()
{
    scanf("%d%d", &m, &n);
    int k = n + m + 1;
    for (int i = 0; i < k; i ++ ) scanf("%d", &a[i]);

    LL sum = 0;
    if (!n)
    {
        for (int i = 0; i < k; i ++ ) sum += a[i];
    }
    else
    {
        int maxp = 0, minp = 0;
        for (int i = 0; i < k; i ++ )
        {
            if (a[i] > a[maxp]) maxp = i;
            if (a[i] < a[minp]) minp = i;
        }

        sum = a[maxp] - a[minp];
        for (int i = 0; i < k; i ++ )
            if (i != maxp && i != minp)
                sum += abs(a[i]);
    }

    printf("%lld\n", sum);

    return 0;
}
试题J：灵能传输
#include <cstring>
#include <iostream>
#include <algorithm>
#include <limits.h>

using namespace std;

typedef long long LL;
const int N = 300010;

int n;
LL sum[N], a[N], s0, sn;
bool st[N];

int main()
{
    int T;
    scanf("%d", &T);
    while (T -- )
    {
        scanf("%d", &n);
        for (int i = 1; i <= n; i ++ )
        {
            scanf("%lld", &sum[i]);
            sum[i] += sum[i - 1];
        }

        s0 = sum[0], sn = sum[n];
        if (s0 > sn) swap(s0, sn);

        sort(sum, sum + n + 1);

        for (int i = 0; i <= n; i ++ )
            if (s0 == sum[i])
            {
                s0 = i;
                break;
            }
        for (int i = n; i >= 0; i -- )
            if (sn == sum[i])
            {
                sn = i;
                break;
            }

        memset(st, 0, sizeof st);
        int l = 0, r = n;
        for (int i = s0; i >= 0; i -= 2)
        {
            a[l ++ ] = sum[i];
            st[i] = true;
        }
        for (int i = sn; i <= n; i += 2)
        {
            a[r -- ] = sum[i];
            st[i] = true;
        }
        for (int i = 0; i <= n; i ++ )
            if (!st[i])
            {
                a[l ++ ] = sum[i];
            }

        LL res = 0;
        for (int i = 1; i <= n; i ++ ) res = max(res, abs(a[i] - a[i - 1]));
        cout << res << endl;
    }
    return 0;
}
```

作者：yxc
链接：https://www.acwing.com/blog/content/92/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。