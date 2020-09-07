#include<iostream>

using namespace std;
const int N = 503;
int n;
int a[N][N];
int f[N];

int main()
{
    cin>>n;
    for(int i = 1;i<=n;i++)
        for(int j = 1;j<=i;j++)
            cin>>a[i][j];
    for(int j = 1;j<=n;j++)
        f[j] = a[n][j];   // initialized to last line of triangle
    for(int i = n - 1;i>=1;i--)
        for(int j = 1;j<=i;j++)
            f[j] = a[i][j] + max(f[j],f[j+1]);

    cout << f[1] << endl;
    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/58479/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


0

题目描述
给定一个如下图所示的数字三角形，从顶部出发，在每一结点可以选择移动至其左下方的结点或移动至其右下方的结点，一直走到底层，要求找出一条路径，使路径上的数字的和最大。

    7
  3   8
8   1   0
2 7 4 4
4 5 2 6 5

输入格式
第一行包含整数n，表示数字三角形的层数。

接下来n行，每行包含若干整数，其中第 i 行表示数字三角形第 i 层包含的整数。

输出格式
输出一个整数，表示最大的路径数字和。

数据范围
1≤n≤500,
−10000≤三角形中的整数≤10000
输入样例：
5
7
3 8
8 1 0 
2 7 4 4
4 5 2 6 5
输出样例：
30
主要考点
动态规划

解题思路一 ---------------- 自上往下推
闫氏DP分析法

一、状态表示：f[i][j]
1. 集合：所有从顶点(1, 1) 到 (i, j)的路径之和的方案
2. 属性：最大值

二、状态计算：
1. 思想-----集合的划分
2. 集合划分依据：根据最后一步的来向, 即来自左上和来自右上两种.

f[i][j] = max(f[i - 1][j - 1] + a[i][j], f[i - 1][j] + a[i][j]);//依次为左上、右上

C++代码
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>

using namespace std;
const int N = 510, INF = -0x3f3f3f3f;

int a[N][N];
int f[N][N];
int n;

int main(){
    cin >> n;

    memset(f, INF, sizeof f);//初始化

    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= i; j ++){
            cin >> a[i][j];
        }
    }

    f[1][1] = a[1][1];//三角顶点为a[1][1]
    for(int i = 2; i <= n; i ++){
        for(int j = 1; j <= i; j ++){
            f[i][j] = max(f[i - 1][j - 1], f[i - 1][j]) + a[i][j];
        }
    }

    int res = INF;
    for(int j = 1; j <= n; j ++) res = max(res, f[n][j]);//遍历最后一层

    cout << res << endl;

    return 0;
}
解题思路二 ---------------- 自下往上推
闫氏DP分析法

一、状态表示：f[i][j]
1. 集合：所有从最后一层到顶点(1, 1)的路径之和的方案
2. 属性：最大值

二、状态计算：
1. 思想-----集合的划分
2. 集合划分依据：根据最后一步的来向, 即来自左下和来自右下两种.

f[i][j] = max(f[i + 1][j] + a[i][j], f[i + 1][j + 1] + a[i][j]);//依次为左下、右下

C++代码
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>

using namespace std;
const int N = 510;

int f[N][N];
int a[N][N];
int n;

int main(){
    cin >> n;

    for(int i = 1; i <= n; i ++){
        for(int j = 1; j <= i; j ++){
            cin >> a[i][j];
        }
    }

    for(int j = 1; j <= n; j ++) f[n][j] = a[n][j];//最后一层

    for(int i = n; i >= 1; i --){
        for(int j = 1; j <= i; j ++){
            f[i][j] = max(f[i + 1][j], f[i + 1][j + 1]) + a[i][j];
        }
    }

    cout << f[1][1] << endl;

    return 0;
}

作者：以梦为马
链接：https://www.acwing.com/solution/content/17544/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1

AcWing 898. 数字三角形    原题链接    简单
作者：    TaoZex ,  2019-08-06 10:54:47 ,  阅读 1042

6


2
#include<bits/stdc++.h>
using namespace std;

const int N=510,INF=0x3f3f3f3f;
int f[N][N];
int a[N][N];

int main(){
    int n;
    cin>>n;

    for(int i=1;i<=n;i++){
        for(int j=1;j<=i;j++){
            cin>>a[i][j];
        }
    }

    for(int i=1;i<=n;i++){             
        for(int j=0;j<=i+1;j++){          //因为有负数，所以应该将两边也设为-INF
            f[i][j]=-INF;
        }
    }

    f[1][1]=a[1][1];
    for(int i=2;i<=n;i++){
        for(int j=1;j<=i;j++){
            f[i][j]=a[i][j]+max(f[i-1][j-1],f[i-1][j]);
        }
    }

    int res=-INF;
    for(int i=1;i<=n;i++) res=max(res,f[n][i]);
    cout<<res<<endl;
}
也可以倒序dp，更简单些，因为倒序不需要考虑边界问题

#include<bits/stdc++.h>
using namespace std;

const int N=510;
int f[N][N];
int n;

int main(){
    cin>>n;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=i;j++){
            cin>>f[i][j];
        }
    }

    for(int i=n;i>=1;i--){
        for(int j=i;j>=1;j--){
            f[i][j]=max(f[i+1][j],f[i+1][j+1])+f[i][j];  //what about n+1??
        }
    }
    cout<<f[1][1]<<endl;
}

作者：TaoZex
链接：https://www.acwing.com/solution/content/3485/

2

AcWing 898. 数字三角形    原题链接    简单
作者：    Shadow ,  2019-10-27 22:46:28 ,  阅读 757

4


2
y总讲的炒鸡好！
但是
我快被那个鬼畜边界弄疯了

y总讲的是从三角形上方往下顺推
我来一个逆推
其实也不逆啦就是
往上爬
为什么我要往上爬呢
因为我不想处理那么多边界情况，一不小心就死翘翘了
所以我还是往上爬吧
（虽然要克服重力做功我这么胖也不容易）
我这个甚至不用考虑有没有负数！
原因自己想啦
代码来一波

#include<iostream>
#include<cstdio>
#include<algorithm>
using namespace std;
const int INF = 2e9;
int f[1001][1001];
int n,a[1001][1001];
int main() {
    scanf("%d",&n);
    for(register int i=1; i<=n; i++)
        for(register int j=1; j<=i; j++)
            scanf("%d",&a[i][j]);
    for(register int i=n; i>=1; i--)
        for(register int j=i; j>=1; j--)
            f[i][j] += max(f[i+1][j] , f[i+1][j+1]) + a[i][j];
    printf("%d\n",f[1][1]);
    return 0;
}
最后！
我们来对比一下顺着三角形往下滑（顺推）的代码
（不用克服重力可是你摩擦生热不烫吗）

#include <bits/stdc++.h>
using namespace std;
const int INF  = 2e9;
int a[1001][1001];
int f[1001][1001];
int n,ans=-INF;
int main() {//这个玩意是顺推的 
    scanf("%d",&n);
    for(register int i=1; i<=n; i++)
        for(register int j=1; j<=i; j++)
            scanf("%d",&a[i][j]);

    for(register int i=0; i<=n; i++) //注意要从0到n（一共n+1个数） 
        for(register int j=0; j<=i+1; j++)//而且这里要每行多初始化一个 
            f[i][j] = -INF;
    f[1][1] = a[1][1];

    for(register int x=2; x<=n; x++)//魔鬼边界，这儿从2开始 
        for(register int y=1; y<=x; y++)//但是这儿从1开始 
            f[x][y] = max(f[x-1][y-1] , f[x-1][y]) + a[x][y];

    for(register int i=1; i<=n; i++) ans = max(ans,f[n][i]);

    printf("%d\n",ans);
    return 0;
}
//别忘了-INF（见上）
//来自算法基础课

最后的最后
这个故事告诉我们
看上去艰难的路途不一定费力
看上去安逸的大道也不一定闲适
不管前途如何
要敢于尝试
（说不定顺便就减肥了呢对吧）

作者：Shadow
链接：https://www.acwing.com/solution/content/5610/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

3 python



感觉ACWing上还是很少有python 的解答。提供一些python的解法供大家讨论
算法
# 自下往上做更方便

N = int(input())
# 这里我多开了一些dp的空间，这样可以简化初始化的问题
dp = [[0]*(N+1) for _ in range(N+2)]
s = [[0]]
for i in range(N):
    a = list(map(int, input().split()))
    s.append(a)

for i in range(N, 0, -1):
    for j in range(i):
        dp[i][j] = max(dp[i+1][j], dp[i+1][j+1])+s[i][j]

print(dp[1][0])
欢迎各位大佬进行指点

作者：遍地跑小鸡
链接：https://www.acwing.com/solution/content/15199/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


4

动态规划⎧⎩⎨⎪⎪⎪⎪⎪⎪状态表示f(i,j){集合：所有从起点走到(i,j)的路径属性：路径数字和最大状态计算⎧⎩⎨⎪⎪分类{从左上方到达(i,j)从右上方到达(i,j)f(i−1,j−1)+a(i,j)f(i−1,j)+a(i,j)求解：两种可能求max
动态规划{状态表示f(i,j){集合：所有从起点走到(i,j)的路径属性：路径数字和最大状态计算{分类{从左上方到达(i,j)f(i−1,j−1)+a(i,j)从右上方到达(i,j)f(i−1,j)+a(i,j)求解：两种可能求max
#include <iostream>
#include <cstring>
using namespace std;

const int N = 510, INF = 1e9;

int a[N][N], f[N][N];
int n;

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++)
        for (int j = 1; j <= i; j ++)
            scanf("%d", &a[i][j]);

    memset(f, 0x8f, sizeof f);  // 初始化为负无穷

    f[1][1] = a[1][1];
    for (int i = 2; i <= n; i ++)
        for (int j = 1; j <= i; j ++)
            f[i][j] = max(f[i-1][j-1], f[i-1][j]) + a[i][j];

    int res = -INF;
    for (int j = 1; j <= n; j ++)
        res = max(res, f[n][j]);

    printf("%d\n", res);
    return 0;
}

作者：西河二当家
链接：https://www.acwing.com/solution/content/12295/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


5. memset to initialize

#include <iostream>
#include <cstring>
using namespace std;
const int N = 510;
int a[N][N], f[N][N];
int main(){
    int n;
    cin >> n;
    for(int i = 1; i <= n; i ++ ){
        for(int j = 1; j <= i; j ++ ){
            cin >> a[i][j];
        }
    }
    memset(f, -0x3f3f3f, sizeof f);
    f[0][0] = 0;   // 定义一个入口
    for(int i = 1; i <= n; i ++ ){
        for(int j = 1; j <= i; j ++ ){
            f[i][j] = max(f[i - 1][j], f[i - 1][j - 1]) + a[i][j];
        }
    }
    int res = -0x3f3f3f3f;
    for(int i = 1; i <= n; i ++ )   res = max(res, f[n][i]);
    cout << res << endl;
    return 0;
}

作者：魔鬼
链接：https://www.acwing.com/solution/content/13902/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

6. 1D array instead of 2D array

2


在基础课代码基础上从二维优化为一维，滚动数组

#include <iostream>

using namespace std;

const int N = 510, INF = 1e9;

int f[N], a[N][N];

int main()
{
    int n;
    cin >> n;

    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= i; j++)
            cin >> a[i][j];

    for(int j = 0; j <= 2; j++)
        f[j] = -INF;

    f[1] = a[1][1];

    for(int i = 2; i <= n; i++)
    {
        f[0] = -INF;
        f[i+1] = -INF;
        for(int j = i; j >= 1; j--)
            f[j] = max(f[j-1], f[j]) + a[i][j];
    }

    int res = -INF;
    for(int i = 1; i <= n; i++) res = max(res, f[i]);
    cout << res << endl;
    return 0;
}

作者：HIHIA
链接：https://www.acwing.com/solution/content/15323/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1.

数字三角形
从问题的状态表示分析，f[i][j]f[i][j]表示从起点开始到达(i,j)(i,j)的路径。
自底向上考虑，对于第ii层而言，状态转移方程可以表示为：
f[i][j]=max(f[i+1][j]+a[i][j],f[i+1][j+1]+a[i][j])f[i][j]=max(f[i+1][j]+a[i][j],f[i+1][j+1]+a[i][j])
此时，可以直接在原数组上进行DPDP，而无需开辟额外数组来保存转移信息。
当到达最顶层时，就是问题答案。

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 510;

int n;
int a[N][N];


int main() {
    cin >> n;

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= i; j++)
            cin >> a[i][j];

    for (int i = n - 1; i >= 1; i--)
        for (int j = 1; j <= i; j++)
            a[i][j] = max(a[i + 1][j] + a[i][j], a[i + 1][j + 1] + a[i][j]);

    cout << a[1][1];
    return 0;
}

作者：Logic
链接：https://www.acwing.com/solution/content/10397/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1



1


1
算法1
DP
我们可以倒着想这道题qwq.不难得到状态转移方程:

f[i][j]=max(f[i+1][j],f[i+1][j+1])+w[i][j];
C++ 代码
#include <iostream>
#include <cstdio>

using namespace std;

int w[1001][1001];
int f[1001][1001];

int main()
{
    int n;
    cin>>n;
    for(int i=1;i<=n;i++)
    {
        for(int j=1;j<=i;j++)
        {
            scanf("%d",&w[i][j]);
        }
    }
    for(int i=n;i>=1;i--)
    {
        for(int j=i;j>=1;j--)
        {
            f[i][j]=max(f[i+1][j],f[i+1][j+1])+w[i][j];
        }
    }
    printf("%d",f[1][1]);
    return 0;
}

作者：smallfang
链接：https://www.acwing.com/solution/content/8132/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1. python
# 输入样例
n = int(input().strip())
arr = [[0 for i in range(n+1)] for j in range(n+1)]    # arr.shape [0~n][0~n],默认值为0

for i in range(1,n+1):
    in_li = list(map(int, input().split()))
    for j in range(1,i+1):
        arr[i][j] = in_li[j-1]

# 初始化dp数组
dp = [[float("-inf") for i in range(n+1)] for j in range(i+2)]   
# !!!技巧：对输入三角形扩大一行两列，扩充的元素用-inf填充（因为是求max），目的是高效处理边界条件
dp[1][1] = arr[1][1]

# 状态计算
for i in range(2, n+1):    # 因为i=1已经初始化，所以i从2开始遍历
    for j in range(1,i+1):
        dp[i][j] = max(dp[i-1][j-1]+arr[i][j], dp[i-1][j]+arr[i][j])    # 状态转移方程

res = float("-inf")
for i in range(1,n+1):    # 遍历最后一行所有元素，取最大值作为输出结果
    res = max(res, dp[n][i])
print(res)

作者：Actor丶
链接：https://www.acwing.com/solution/content/7944/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1.


0


自底向上考虑可以省去边界问题和判断f[n][1]–f[n][n]的最大值
直接在存数的数组里进行更新就行
#include<iostream>
using namespace std;
int n, a[505][505];
int main()
{
    cin >> n;
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= i; j++)
            scanf("%d",&a[i][j]);
    for(int i = n-1; i >= 1; i--)
        for(int j = 1; j <= i; j++)
            a[i][j] = max(a[i+1][j], a[i+1][j+1]) + a[i][j];
    cout << a[1][1];
    return 0;
}


作者：东D
链接：https://www.acwing.com/solution/content/8052/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1
#include<bits/stdc++.h>
using namespace std;

int n, a[1005][1005], f[1005][1005];

int main()
{
    cin >> n;
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= i; j++)
            cin >> a[i][j];
    for(int i = 1; i <= n; i++)
        f[n][i] = a[n][i];
    for(int i = n - 1; i >= 1; i--)
        for(int j = 1; j <= i; j++)
            f[i][j] = max(f[i + 1][j + 1], f[i + 1][j]) + a[i][j];
    cout << f[1][1] << endl;       
    return 0;
}

作者：Bobby
链接：https://www.acwing.com/solution/content/7942/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1.

#include<iostream>
#include<algorithm>
#include<cstring>
using namespace std;
const int N=510;
int dp[N][N];
int main()
{
    int n;cin>>n;
    //这里由于没有初始化dp为负无穷，导致如果有负数就会选择不存在的0
    memset(dp,-0x3f,sizeof dp);
    cin>>dp[1][1];
    for(int i=2;i<=n;i++)
    {
        for(int j=1;j<=i;j++)
        {
            cin>>dp[i][j];
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-1])+dp[i][j];
        }
    }
    int ans=0;
    for(int i=1;i<=n;i++){ans=max(ans,dp[n][i]);}
     cout<<ans;
    return 0;
}

作者：yukino
链接：https://www.acwing.com/solution/content/10293/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


1.

#include <cstdio>
#include <iostream>

using namespace std;

const int N = 510;

int f[N][N];
int a[N][N];
int main()
{
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= i; j++)
            cin >> a[i][j];

    // 初始化
    for (int i = 1; i <= n; i++)
        f[n][i] = a[n][i];

    // 状态转移
    for (int i = n-1; i >=1; i--)
        for (int j = i; j >= 1; j--)
            f[i][j] = max(f[i+1][j], f[i+1][j+1]) + a[i][j];

    cout << f[1][1] << endl;

    return 0;
}

作者：小辉_9
链接：https://www.acwing.com/solution/content/18577/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1

(滚动数组优化动态规划状态转移数组)
复杂度分析
时间复杂度: O (n ^ 2)
空间复杂度: O ( n )

有关动态规划中滚动数组优化的Tips
如果dp[j] 的更新需要用到dp[j],dp[j+1] 则需要顺序遍历
如果dp[j] 的更新需要用到dp[j],dp[j-1] 则需要逆序遍历 (因为我们需要保证j是在j-1之前遍历的，相同的道理同样适用于其他滚动数组优化一维DP)

C++ 代码
#include <iostream>
#include <algorithm>
using namespace std;
const int N = 510;
int n;
int g[N][N];
int dp[N];
int main()
{
    cin>>n;
    for(int i=0;i<n;i++)
    {  
        for(int j=0;j<=i;j++) 
        {   
            cin>>g[i][j];
        }
    }
    dp[0] = g[0][0];
    int res = 0; 
    for(int i=1;i<n;i++)
    {
        for(int j=i;~j;j--)
        {
            if(!j) dp[j] = dp[j] + g[i][j];
            if(j==i) dp[j] = dp[j-1] + g[i][j];
            if(j != i && j) dp[j] = max(dp[j],dp[j-1]) + g[i][j];
            if(i == n-1) res = max(res,dp[j]);
        }
    }
    cout<<res<<endl;
    return 0;
}

作者：NEU-DHM
链接：https://www.acwing.com/solution/content/19119/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。