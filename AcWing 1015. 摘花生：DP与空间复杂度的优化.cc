AcWing 1015. 摘花生：DP与空间复杂度的优化    原题链接    简单
作者：    dasongshu ,  2020-02-11 11:31:03 ,  阅读 395

5


2
题目描述
Hello Kitty想摘点花生送给她喜欢的米老鼠。

她来到一片有网格状道路的矩形花生地(如下图)，从西北角进去，东南角出来。

地里每个道路的交叉点上都有种着一株花生苗，上面有若干颗花生，经过一株花生苗就能摘走该它上面所有的花生。

Hello Kitty只能向东或向南走，不能向西或向北走。

问Hello Kitty最多能够摘到多少颗花生。

输入格式
第一行是一个整数T，代表一共有多少组数据。

接下来是T组数据。

每组数据的第一行是两个整数，分别代表花生苗的行数R和列数 C。

每组数据的接下来R行数据，从北向南依次描述每行花生苗的情况。每行数据有C个整数，按从西向东的顺序描述了该行每株花生苗上的花生数目M。

输出格式
对每组输入数据，输出一行，内容为Hello Kitty能摘到得最多的花生颗数。

样例
输入样例：
2
2 2
1 1
3 4
2 3
2 3 4
1 6 5
输出样例：
8
16
思路
状态表示
集合：定义f[i][j]为从(1, 1)到达(i, j)的所有方案
属性：最大值
状态转移
(i, j)从(i-1, j)即上方过来
(i, j)从(i, j-1)即左方过来
空间压缩
f[i][j]只需要用到这一层和上一层的f元素，所以可以压缩成滚动数组。在此之上，还可以直接压缩成一维数组。
算法1
DP：空间复杂度O(n2)O(n2)
#include<iostream>
using namespace std;

const int N = 105;
int a[N][N], f[N][N];
int q, row, col;

int main()
{
    cin >> q;
    while(q--){
        cin >> row >> col;
        for(int i = 1; i <= row; i++){
            for(int j = 1; j <= col; j++){
                cin >> a[i][j];
            }
        }

        // f[i][j]指的是到(i, j)的最大花生数
        for(int i = 1; i <= row; i++){
            for(int j = 1; j <= col; j++){
                f[i][j] = max(f[i-1][j], f[i][j-1]) + a[i][j];
            }
        }

        cout << f[row][col] << endl;
    }

    return 0;
}
算法2
DP:滚动数组，空间复杂度O(n)O(n)
#include<cstring>
#include<iostream>
using namespace std;

const int N = 105;
int a[2][N], f[2][N], q, n, m;

int main()
{
    cin >> q;
    while(q--){
        cin >> n >> m;

        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                cin >> a[i&1][j];
                f[i&1][j] = max(f[i&1][j-1], f[(i-1)&1][j]) + a[i&1][j];
            }
        }
        cout << f[n&1][m] << endl;

        memset(f, 0, sizeof f);
    }
}
算法3
DP:空间复杂度O(n)O(n)
#include<cstring>
#include<iostream>
using namespace std;

const int N = 105;
int a[N][N], f[N], q, n, m;

int main()
{
    cin >> q;
    while(q--){
        cin >> n >> m;

        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                cin >> a[i][j];
            }
        }

        for(int i = 1; i <= n; i++){
            for(int j = 1; j <= m; j++){
                f[j] = max(f[j], f[j-1]) + a[i][j];
            }
        }
        cout << f[m] << endl;

        // 由于多组样例，而二维数组解法由于f[0][...]和f[...][0]都为0，所以没有问题。对于一维数组，上一样例的f数组需要清零，否则影响结果
        memset(f, 0, sizeof f);
    }
}

作者：dasongshu
链接：https://www.acwing.com/solution/content/8422/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

引言
作为本人DP的引路题，我想分享一下DP的使用条件，以及DP到底做了什么？DP为什么那么神奇？

DP源自递推。
摘自《算法竞赛进阶指南》:动态规划算法把原问题视作若干个重叠的子问题的逐层递进，每个子问题的求解过程都构成一个”阶段”,在完成前一个阶段的计算后，动态规划才会执行下一阶段。
DP使用的条件。
DP的使用条件为”无后效性”。即已经求解的子问题不受后续阶段影响。
DP为什么那么神奇（高效）？
总而言之一句话：已经做过考虑的最优决策无需再做。
本题算法:线性DP O(r * c)
因为只能想东或者向南，所以每一次走到的点，一定不会被重复走到。这很好证明所以证明省略。且只能被上面一格的点或者左边一格的点走到。所以状态计算迎刃而解。

代码
#include <bits/stdc++.h>

using namespace std;

const int N = 110;

int g[N][N];
int f[N][N];

int main(){

    ios :: sync_with_stdio(false);
    cin.tie(0);

    int T;

    cin >> T;

    while(T--)
    {
        int r,c;
        cin >> r >> c;

        f[0][0] = 0;

        for(int i=0;i<r;i++)
            for(int j=0;j<c;j++)
                cin >> g[i][j];

        for(int i=0;i<r;i++)
            for(int j=0;j<c;j++)
            {
                if(i == 0 && j == 0)
                    f[i][j] = g[i][j];
                else
                {
                    int t = 0;
                    if(i) t = f[i - 1][j];
                    if(j) t = max(t,f[i][j - 1]);
                    f[i][j] = t + g[i][j];

                }
            }

        cout << f[r - 1][c - 1] << endl;
    }

    return 0;
}

作者：胡图图
链接：https://www.acwing.com/solution/content/9145/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


#include<bits/stdc++.h>
using namespace std;
const int N = 110;
int T;
int n, m;
int a[N][N];
int f[N][N];//状态表示：f(i,j)表示从(1,1)到(i,j)最多摘到的花生数量
//集合属性：MAX
int main()
{
    scanf("%d", &T);
    while(T --)
    {
        scanf("%d%d", &n, &m);
        for(int i = 1; i <= n; i ++)
            for(int j = 1; j <= m; j ++)
                scanf("%d", &a[i][j]);
        //读入
        for(int i = 1; i <= n; i ++)
            for(int j = 1; j <= m; j ++)
                f[i][j] = max(f[i - 1][j], f[i][j - 1]) + a[i][j];//状态转移方程：从上面的和从左边的转移加上这个地方摘到的花生
        printf("%d\n", f[n][m]);//输出
    }
    return 0;
}

作者：cht
链接：https://www.acwing.com/solution/content/17469/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1

题目描述不再赘述，一道典型的二维dp，我的思考是要考虑每一步的“决策”是什么，也就是说我通过什么方法让这一步实现最优解进而获取整个题目的最优解，对于这道题而言，每一步的最大值取决于上一步的来源（来源于东还是南）。最后根据决策写出方程：dp[i][j]=max(dp[i-1][j],dp[i][j-1])+a[i][j];,这里要注意这一步是上一步来的，所以i-1/j-1

样例
输入样例：
2
2 2
1 1
3 4
2 3
2 3 4
1 6 5
输出样例：
8
16
算法1
#include<bits/stdc++.h>
using namespace std;
int T;
int n,m;
const int N=110;
int a[N][N];
int sum[N][N];
int main()
{
    cin>>T;
    while(T--)
    {
        cin>>n>>m;
        for(int i=1;i<=n;i++)
        {
            for(int j=1;j<=m;j++)
            {
                cin>>a[i][j];
            }
        }
        for(int i=1;i<=n;i++)
        {
            for(int j=1;j<=m;j++)
            {
                sum[i][j]=max(sum[i-1][j],sum[i][j-1])+a[i][j];
            }
        }
        cout<<sum[n][m]<<endl;
    }
    return 0;
}

作者：田所浩二
链接：https://www.acwing.com/solution/content/7121/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1

#include <iostream>
#include <algorithm>

using namespace std;
const int N = 110;

int T;
int w[N][N],f[N][N];

int main(){
    cin >> T;

    int r, c;
    while(T--){
        cin >> r >> c;
        for(int i = 1; i <= r;  i++){
            for(int j = 1; j <= c; j ++){
                cin >> w[i][j];
                f[i][j] = f[i - 1][j] + w[i][j];//由上至下
                f[i][j] = max(f[i][j], f[i][j - 1] + w[i][j]);//由左到右
            }
        }
        cout << f[r][c] << endl;
    }


    return 0;
}

作者：以梦为马
链接：https://www.acwing.com/solution/content/17461/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1.
dp问题

集合角度进行考虑

状态表示
集合: f[i,j] 所有从(1,1)走到(i,j)的路线
属性：Max/Min/数量： 集合中每个元素和的最大值
> 注意Min一般需要注意初始化
状态计算
集合划分：考虑最后一步
> 往下或者往右走
划分依据： 整个计算的连通性
不重复：最值时不需要考虑
不漏
代码
#include<iostream>
#include<algorithm>
using namespace std;
const int N=110;

int n, m;
int w[N][N];
int f[N][N];

int main() {
    int T;
    scanf("%d", &T);
    while(T--) {
        scanf("%d%d", &n, &m);
        for (int i=1; i<=n; i++)
            for (int j=1; j<=m; j++)
                scanf("%d", &w[i][j]);

        for (int i=1; i<=n; i++)
            for (int j=1; j<=m; j++)
                f[i][j] = max(f[i-1][j], f[i][j-1]) + w[i][j];

        printf("%d\n", f[n][m]);
    }
    return 0;
}

作者：就是个渣渣
链接：https://www.acwing.com/solution/content/3263/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

2

分析
我们把走到一个点看做一个状态，对HelloKittyHelloKitty来说，走到一个点只有两种方式，一种是从上面（北面）走到该点，另一种是从左边（西面）走到该点。要到达点(i,j)(i,j)，要么是从(i−1,j)(i−1,j)走到(i,j)(i,j)，要么是从点(i,j−1)(i,j−1)走到(i,j)(i,j)。所以从哪个点走到(i,j)就是一个决策。

接下来，我们用f[i][j]f[i][j]来代表走到点(i,j)(i,j)一共摘到的花生。我们需要走到(n,m)(n,m)，所以可以得到状态转移方程: f[i][j]=min(f[i−1][j]),f[i][j−1]))+a[i][j]f[i][j]=min(f[i−1][j]),f[i][j−1]))+a[i][j]。根据转移方程，我们可以推出走到(n,m)(n,m)处摘到的最多花生。

C++ 代码
#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;

int a[105][105], f[105][105];

int main()
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        memset(a, 0, sizeof(a));
        memset(f, 0, sizeof(f));
        int n, m;
        scanf("%d%d", &n, &m);
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                scanf("%d", &a[i][j]);
            }
        }
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                f[i][j] = max(f[i][j - 1], f[i - 1][j]) + a[i][j];
            }
        }
        printf("%d\n", f[n][m]);
    } 
    return 0;
}


作者：Tony_Thomas
链接：https://www.acwing.com/solution/content/7080/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
1

#include <iostream>

using namespace std;

const int N = 105;
int t;
int m[N][N];
int f[N][N];
int r,c;
int main()
{
    cin >> t;
    while(t -- )
    {
        cin >> r >> c;

        for(int i = 1; i <= r; i ++)
            for(int j = 1; j <= c; j ++)
                cin >> m[i][j];
        for(int i = 0; i <= r; i ++)
            f[i][0] = 0;
        for(int j = 0; j <= c; j ++)
            f[0][j] = 0;
        for(int i = 1; i <=r; i++)
            for(int j = 1; j <= c; j ++)
                f[i][j] = max(f[i-1][j], f[i][j-1]) + m[i][j];

        cout << f[r][c] << endl;
    }
    return 0;
}
//f[i,j]
//集合:从起点到(i,j)的所有可行路径
//属性：MAXN
//状态计算：f[i,j] = max(f[i-1,j], f[i,j-1]) + m[i][j]

作者：hegehog
链接：https://www.acwing.com/solution/content/15817/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1.
#include<bits/stdc++.h>
using namespace std;
int f[1000][1000];
int main()
{
    int n;
    cin>>n;
    for(int i=1;i<=n;i++)
    {
        int a,b;
        cin>>a>>b;
        for(int j=1;j<=a;j++)
        {
            for(int k=1;k<=b;k++)
            {
                cin>>f[j][k];
            }
        }
        for(int j=1;j<=a;j++)
        {
            for(int k=1;k<=b;k++)
            {
                f[j][k]=max(f[j-1][k],f[j][k-1])+f[j][k];
            }
        }
        cout<<f[a][b];
        cout<<endl;
    }
}

作者：ABlyh
链接：https://www.acwing.com/solution/content/17442/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1
#include<iostream>
#include<cstdlib>
#include<cstring>

using namespace std;

const int N = 1e2 + 10;

int n, m;
int t;
int g[N][N], f[N];

int main() {
    ios::sync_with_stdio(false);
    cin >> t;
    while (t--) {
        cin >> n >> m;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                cin >> g[i][j];
                f[j] = max(f[j], f[j - 1]) + g[i][j];
            }
        }
//        for (int i = 1; i <= n; i++) {
//            for (int j = 1; j <= m; j++) {
//                f[i][j] = max(f[i - 1][j], f[i][j - 1]) + g[i][j];
//            }
//        }
        cout << f[m] << endl;
        memset(f, 0, sizeof f);
    }
    return 0;
}

作者：BugFree
链接：https://www.acwing.com/solution/content/12120/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1
摘花生
动态规划入门基础题
只能向下或向右走

f[i][j]表示从1，1 到 i，j 所得花生的最大值。
w[i][j] 表示i，j 处有多少花生

容易得到 f[i][j] =max(f[i-1][j],f[i][j-1])+w[i][j];

状态枚举必须有f[i-1][j]和f[i][j-1]先有值，所以按正常输入矩阵,输出矩阵方式枚举就可以了。

答案就是f[n][m]
#include <bits/stdc++.h>

using namespace std;

const int N=110;

int w[N][N];
int f[N][N];

int main()
{
    int t;
    scanf("%d", &t);

    while(t--)
    {
        int n,m; //表示n行m列
        scanf("%d%d", &n, &m);

        for(int i=1;i<=n;i++)
            for(int j=1;j<=m;j++)
                scanf("%d", &w[i][j]),f[i][j]=max(f[i-1][j],f[i][j-1])+w[i][j];

        printf("%d\n", f[n][m]);
    }


    return 0;
}

作者：_empty
链接：https://www.acwing.com/solution/content/4300/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：ShidongDu time:2020/4/9
'''
Hello Kitty想摘点花生送给她喜欢的米老鼠。

她来到一片有网格状道路的矩形花生地(如下图)，从西北角进去，东南角出来。

地里每个道路的交叉点上都有种着一株花生苗，上面有若干颗花生，经过一株花生苗就能摘走该它上面所有的花生。

Hello Kitty只能向东或向南走，不能向西或向北走。

问Hello Kitty最多能够摘到多少颗花生。

1.gif

输入格式
第一行是一个整数T，代表一共有多少组数据。

接下来是T组数据。

每组数据的第一行是两个整数，分别代表花生苗的行数R和列数 C。

每组数据的接下来R行数据，从北向南依次描述每行花生苗的情况。每行数据有C个整数，按从西向东的顺序描述了该行每株花生苗上的花生数目M。

输出格式
对每组输入数据，输出一行，内容为Hello Kitty能摘到得最多的花生颗数。

数据范围
1≤T≤100,
1≤R,C≤100,
0≤M≤1000
输入样例：
2
2 2
1 1
3 4
2 3
2 3 4
1 6 5
输出样例：
8
16
'''
# 首先，动态规划问题要注意两个点：1、不重  2、不漏
# 不重的话，不是必要条件，对于属性为min、max的题目可以忽略，对于属性为count的题目不可以忽略
# 不漏的话，这是必要条件

# 状态表示：集合：dp[i][j]表示从dp[1][1]走到dp[i][j]的所有路线    属性：max
# 状态计算：有两种情况：①从上面下来     ②从左边过来
# dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + a[i][j]

from typing import List
class Solution:
    def penats(self, group: List[List[int]]):
        dp = [[0 for _ in range(len(group[0])+1)] for _ in range(len(group)+1)]
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + group[i-1][j-1]
        return dp[-1][-1]

if __name__ == '__main__':
    solution = Solution()
    res = []
    groups_num = int(input())
    for i in range(groups_num):
        row_num, col_num = list(map(int, input().split()))
        cur_group = []
        for j in range(row_num):
            cur_group.append(list(map(int, input().split())))
        tmp_res = solution.penats(cur_group)
        res.append(tmp_res)

    for k in res:
        print(k)


作者：夏天的梦是什么颜色的呢
链接：https://www.acwing.com/solution/content/11304/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。