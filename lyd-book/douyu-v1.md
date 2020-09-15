https://v.douyu.com/show/wLjGvLp2nYNvmO90

https://www.acwing.com/problem/search/2/?csrfmiddlewaretoken=aYKyrGXyfWy5Xc83a58KiUtOanID7BhNL8SP00u0IkQb96yekngVifjPwotIWHOY&search_content=%E7%AE%97%E6%B3%95%E7%AB%9E%E8%B5%9B%E8%BF%9B%E9%98%B6%E6%8C%87%E5%8D%97

https://www.acwing.com/problem/content/160/

pdf book page 101 - necklace ch 1807
4 problems - in chap2 summary last:


http://noi-test.zzstep.com/contest/0x18%E3%80%8C%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E3%80%8D%E7%BB%83%E4%B9%A0

https://github.com/lydrainbowcat/tedukuri/blob/master/%E9%85%8D%E5%A5%97%E5%85%89%E7%9B%98/%E4%B9%A0%E9%A2%98/0x18%20%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%20%E6%80%BB%E7%BB%93%E4%B8%8E%E7%BB%83%E4%B9%A0/Necklace/BZOJ1398.cpp


https://www.cnblogs.com/wyboooo/p/9825196.html
http://contest-hunter.org:83/contest?type=1

AcWing 158. 项链    原题链接    简单
作者：    秦淮岸灯火阑珊 ,  2019-01-31 21:15:02 ,  阅读 818

3


3
题目描述
有一天，达达捡了一条价值连城的宝石项链，但是，一个严重的问题是，他并不知道项链的主人是谁！

在得知此事后，很多人向达达发来了很多邮件，都说项链是自己的，要求他归还（显然其中最多只有一个人说了真话）。

达达要求每个人都写了一段关于自己项链的描述： 项链上的宝石用数字0至9来标示。

一个对于项链的表示就是从项链的某个宝石开始，顺指针绕一圈，沿途记下经过的宝石，比如项链： 0-1-2-3 ，它的可能的四种表示是0123、1230、2301、3012。

达达现在心急如焚，于是他找到了你，希望你能够编写一个程序，判断两个给定的描述是否代表同一个项链（注意，项链是不会翻转的）。

也就是说给定两个项链的表示，判断他们是否可能是一条项链。

输入格式
输入文件只有两行，每行一个由字符0至9构成的字符串，描述一个项链的表示（保证项链的长度是相等的）。

输出格式
如果两个对项链的描述不可能代表同一个项链，那么输出’No’，否则的话，第一行输出一个’Yes’，第二行输出该项链的字典序最小的表示。

数据范围
设项链的长度为L，1≤L≤10000001≤L≤1000000
样例
输入样例：
2234342423
2423223434
输出样例：
Yes
2234342423
两遍最小表示法 O(n2)O(n2)
这道题目很简单,因为题目中有多次提示我们这道题目的解法,我们发现这道题目要求我们首先check两个项链是不是一样,然后数据范围极为庞大,那么这里我们可以用kmp匹配或者是最小表示法,然后呢第二问更加明显,直接让我们求出最小表示,所以我们可以肯定这道题目要求我们用最小表示法.
如果说两个字符串是同一条项链的话,那么毫无疑问,他们的最小表示肯定一样的.因此我们可以使用O(n)的最小表示法.
最小表示法极为容易理解,但是建议看李煜东大佬的书,书上写得很详细作者我写的不详细
C++ 代码
#include <bits/stdc++.h>
using namespace std;
#define fir(i,a,b) for(int i=a;i<=b;i++)
char a[2000010],b[2000010];
int calc_min(char*s)
{
    int n=strlen(s+1);
    fir(i,1,n)
        s[n+i]=s[i];//将一条链变成一个区间,或者说将这个区间翻一倍.
    int i=1,j=2,k;
    while(i<=n && j<=n)//在合法区间里面
    {
        for (k=0;k<=n && s[i+k]==s[j+k];k++)//找出最大的不满足两个数相等的位置
            ;
        if (k==n)//如果完全一模一样
            break;
        if (s[i+k]>s[j+k])//如果大的话
        {
            i+=k+1;//那么前面的数,肯定都不是最小表示.
            if (i==j)
                i++;
        }
        else
        {
            j+=k+1;//同理
            if (i==j)
                j++;
        }
    }
    return min(i,j);
}
int main()
{
    scanf("%s",a+1);
    int n=strlen(a+1),x=calc_min(a);
    scanf("%s",b+1);
    int m=strlen(b+1),y=calc_min(b);
    a[x+n]=b[y+m]=0;
    if (n==m && !strcmp(a+x,b+y))//check判断
    {
        puts("Yes");
        puts(a+x);
    }
    else 
        puts("No");
}

作者：秦淮岸灯火阑珊
链接：https://www.acwing.com/solution/content/911/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


2.

AcWing 159. 奶牛矩阵    原题链接    中等
作者：    whsstory ,  2019-08-25 21:51:05 ,  阅读 304

4


一道不错的题。
首先仔细审题，是覆盖而不是蓝书上的完全匹配，因而没有长度必须为原串长度因数的限制，于是n−nxt[n]n−nxt[n]一定能覆盖长为nn的串。

考虑将宽度和高度分开：
宽度上，找出一个最小的pos，满足对于所有1≤i≤n,[1,pos]1≤i≤n,[1,pos]能覆盖串ii
这个暴力即可，复杂度O(nm2)O(nm2)
高度上，由于我们已经保证对于所有1≤i≤n,[1,pos]1≤i≤n,[1,pos]能覆盖串ii，也就是每一行都已经合法了，找出最小的hh，使串[1,h][1,h]能覆盖整个矩阵即可。
这个如何快速求？
将每一行看做一个字符，用kmp处理next[]next[],则h=n−next[n]h=n−next[n]（将kmp对于字符的比较转换为对字符串的比较即可）

输入输出量较大，我用字符数组代替了string.

//Wan Hong 2.2
//home
#include<iostream>
#include<cstdio>
#include<algorithm>
#include<cstring>
#include<queue>
#include<vector>
typedef long long ll;
typedef std::pair<ll,ll> pll;
typedef std::string str;
#define INF (1ll<<58)
ll read()
{
    ll f=1,x=0;
    char c=getchar();
    while(c<'0'||c>'9')
    {
        if(c=='-')f=-1;
        c=getchar();
    }
    while(c>='0'&&c<='9')x=x*10+c-'0',c=getchar();
    return f*x;
}
ll max(ll a,ll b)
{
    return a>b?a:b;
}
ll min(ll a,ll b)
{
    return a<b?a:b;
}
bool umax(ll& a,ll b)
{
    if(b>a)return a=b,1;
    return 0;
}
bool umin(ll& a,ll b)
{
    if(b<a)return a=b,1;
    return 0;
}

/**********/
#define MAXN 10011
#define MAXM 111
char a[MAXN][MAXM];
bool p[MAXM];//p[i]=1:[1,i]能覆盖所有串，或者说pos=min{i|p[i]=1}
ll nxt[MAXN];
ll n,m;
bool check(ll x,ll k)//[1,k]能否覆盖串x
{
    ll it=1;
    for(ll i=k+1;i<=m;++i)
    {
        if(a[x][i]==a[x][it])
        {
            ++it;
            if(it>k)it=1;
        }
        else return 0;
    }
    return 1;
}
int main()
{
    n=read(),m=read();
    memset(p,1,sizeof p);
    for(ll i=1;i<=n;++i)
    {
        scanf("%s",a[i]+1);
        for(ll j=1;j<=m;++j)
            if(p[j])p[j]&=check(i,j);//注意要对每个串都满足，所以我使用&=
    }
    ll min_w=0;//即pos
    for(ll i=1;i<=m;++i)
        if(p[i])
        {
            min_w=i;break;
        }
    ll j=0;
    for(ll i=2;i<=n;++i)//kmp求解h
    {
        while(j&&strcmp(a[j+1]+1,a[i]+1))j=nxt[j];//从1开始用，那么比较时要+1
        if(!strcmp(a[j+1]+1,a[i]+1))++j;
        nxt[i]=j;
    }
    printf("%lld",min_w*(n-nxt[n]));
    return 0;
}

作者：whsstory
链接：https://www.acwing.com/solution/content/4144/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


AcWing 159. 奶牛矩阵（kmp）    原题链接    中等
作者：    羽笙 ,  2019-08-07 11:32:39 ,  阅读 399

3


1
本题kmp的应用，现将宽度确定，题中范围是75，直接枚举宽度判断，也只能枚举判断
因为这题在行列上都可以不完全覆盖，当不完全覆盖时不满足所有循环节都是最小循环节的倍数，所以只能枚举
在找高度时使用kmp，在kmp比较是否相等时用strcmp，将每行宽度下一个赋值成0就是’/0’代表字符的结尾

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <cstring>

using namespace std;

const int N=10010,M=100;

int n,m,nxt[N];
char s[N][M];
bool st[M];

int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        scanf("%s",s[i]);
        for(int j=1;j<m;j++)
        {
            for(int k=j;k<m;k+=j)
            {
                for(int u=0;u<j&&u+k<m;u++)
                    if(s[i][u]!=s[i][u+k])st[j]=1;
            }
        }
    }

    int wet;
    for(int i=1;i<=m;i++)if(st[i]==0){wet=i;break;}
    for (int i = 1; i <= n; i ++ ) s[i][wet] = 0;

    for(int i=2,j=0;i<=n;i++)
    {
        while(j && strcmp(s[i],s[j+1]) ) j=nxt[j];
        if(!strcmp(s[i],s[j+1]))j++;
        nxt[i]=j;
    }
    int het=n-nxt[n];

    cout<<wet*het;

}

作者：羽笙
链接：https://www.acwing.com/solution/content/3528/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 159. 奶牛矩阵（字符串哈希）    原题链接    中等
作者：    Overnoise ,  2020-01-14 16:04:12 ,  阅读 710

2


自己yy了好久，终于打出了2维哈希求循环节
这个题解在主要思路就是降维
就是把2维降成1维处理
我们就分析一下下面这个情况
ABCDEA
AAAABA
不难看出，这个的答案是10
我们怎么降维呢？？？
首先，这个数据有2行
我们想办法把它变成1行
(直接对每一列哈希一下即可)
得到每一列的哈希值，组成一个一维
再利用哈希求最小循环节求出ans1
这样我们就解决了在列上的最小循环节
只需要再求出行上面的最小循环节ans2即可
（ans2求法和ans1求法同理）
最终答案=ans1*ans2

#include<bits/stdc++.h>
using namespace std;
long long r,c,ans=1,k=1e9+7,p[100005],s[100005],f[100005];
string t,a[10004];
int main() {
    p[0]=1;
    for(long long i=1; i<=100000; i++)
        p[i]=p[i-1]*k;
    cin>>r>>c;
    for(int i=1; i<=r; i++) {
        cin>>a[i];
        a[i]=' '+a[i];
    }
    for(int i=1; i<=c; i++)
        for(int j=1; j<=r; j++)
            f[i]=f[i]*k+a[j][i];
    for(long long i=1; i<=c; i++)
        s[i]=s[i-1]*k+f[i];
    for(long long i=1; i<=c; i++) {
        long long kk=i+1;
        while(kk+i-1<=c&&s[kk+i-1]-s[kk-1]*p[i]==s[i])
            kk+=i;
        if(c-kk+1<i&&s[c]-s[kk-1]*p[c-kk+1]==s[c-kk+1]) {
            ans=i;
            break;
        }
    }
    memset(f,0,sizeof(f));
    for(int i=1; i<=c; i++)
        for(int j=1; j<=r; j++)
            f[j]=f[j]*k+a[j][i];
    for(long long i=1; i<=r; i++)
        s[i]=s[i-1]*k+f[i];
    for(long long i=1; i<=r; i++) {
        long long kk=i+1;
        while(kk+i-1<=r&&s[kk+i-1]-s[kk-1]*p[i]==s[i])
            kk+=i;
        if(r-kk+1<i&&s[r]-s[kk-1]*p[r-kk+1]==s[r-kk+1]) {
            cout<<ans*i<<endl;
            return 0;
        }
    }
    return 0;
}

作者：Overnoise
链接：https://www.acwing.com/solution/content/7565/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



lyd code

//Author:XuHt
#include <cstdio>
#include <cstring>
#include <iostream>
#define ull unsigned long long
using namespace std;
const int R = 10006, C = 81, P = 13331;
int r, c, nxt[R];
char s[R][C];
ull H[R];

int work(int len) {
	int i = 0, j = -1;
	nxt[0] = -1;
	while (i < len)
		if (j == -1 || H[i] == H[j]) nxt[++i] = ++j;
		else j = nxt[j];
	return len - nxt[len];
}

int main() {
	cin >> r >> c;
	for (int i = 0; i < r; i++) scanf("%s", s[i]);
	memset(H, 0, sizeof(H));
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			H[i] = H[i] * P + s[i][j];
	int ans = work(r);
	memset(H, 0, sizeof(H));
	for (int i = 0; i < c; i++)
		for (int j = 0; j < r; j++)
			H[i] = H[i] * P + s[j][i];
	ans *= work(c);
	cout << ans << endl;
	return 0;
}


https://github.com/lydrainbowcat/tedukuri/tree/master/%E9%85%8D%E5%A5%97%E5%85%89%E7%9B%98/%E4%B9%A0%E9%A2%98/0x18%20%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%20%E6%80%BB%E7%BB%93%E4%B8%8E%E7%BB%83%E4%B9%A0/%E5%8C%B9%E9%85%8D%E7%BB%9F%E8%AE%A1


lyd code of matching statistics:

#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
const int SIZE = 200010;
int f[SIZE], next[SIZE], cnt[SIZE];
char a[SIZE], b[SIZE];
int n, m, q;

int main()
{
	cin >> n >> m >> q;
	scanf("%s", a + 1); // A[1..n]保存A串
	scanf("%s", b + 1); // B[1..n]保存B串

	for (int i = 2, j = 0; i <= m; i++) // 对B串自匹配，求next数组
	{
		while (j>0 && b[j + 1] != b[i]) j = next[j];
		if (b[j + 1] == b[i]) j++;
		next[i] = j;
	}

	for (int i = 1, j = 0; i <= n; i++) // A串与B串进行模式匹配
	{
		while (j>0 && (j == m || a[i] != b[j + 1])) j = next[j];
		if (a[i] == b[j + 1]) j++;
		f[i] = j;
	}

	for (int i = 1; i <= n; i++) cnt[f[i]]++;
	for (int i = n; i; i--) cnt[next[i]] += cnt[i];

	// 此时cnt[x]保存的是匹配长度>=x的位置个数

	for (int i = 1; i <= q; i++)
	{
		int x;
		scanf("%d", &x);
		printf("%d\n", cnt[x] - cnt[x + 1]);
	}
}

AcWing 160. 匹配统计    原题链接    中等
作者：    垫底抽风 ,  2020-07-07 11:12:57 ,  阅读 147

13


题目描述
阿轩在纸上写了两个字符串，分别记为 AA 和 BB。

利用在数据结构与算法课上学到的知识，他很容易地求出了“字符串 AA 从任意位置开始的后缀子串”与“字符串 BB ”匹配的长度。

不过阿轩是一个勤学好问的同学，他向你提出了 QQ 个问题：

在每个问题中，他给定你一个整数 xx，请你告诉他有多少个位置，满足“字符串 AA 从该位置开始的后缀子串”与 BB 匹配的长度恰好为 xx。

例如：A=A= aabcde，B=B= ab，则 AA 有 aabcde、abcde、bcde、cde、de、e 这 66 个后缀子串，它们与 B=B= ab 的匹配长度分别是 11、22、00、00、00、00。

因此 AA 有 44 个位置与 BB 的匹配长度恰好为 00，有 11 个位置的匹配长度恰好为 11，有 11 个位置的匹配长度恰好为 22。

输入格式
第一行输入三个整数 N,M,QN,M,Q，分别表示 AA 串长度、BB 串长度、问题个数。

第二行输入字符串 AA，第三行输入字符串 BB。

接下来 QQ 行每行输入 11 个整数 xx，表示一个问题。

输出格式
输出共 QQ 行，依次表示每个问题的答案。

数据范围
1≤N,M,Q,x≤2000001≤N,M,Q,x≤200000
输入样例：
6 2 5
aabcde
ab
0
1
2
3
4
输出样例：
4
1
1
0
0
算法1
(哈希) (NlogN+M+Q)O(Nlog⁡N+M+Q)
先分别求出 AA 与 BB 的哈希数组，对于 aa 中的每一个后缀，二分求一下能匹配的 BB 的最大前缀即可。
详见代码注释

时间复杂度
求出 AA 的哈希数组，时间复杂度是 (N)O(N)
求出 BB 的哈希数组，时间复杂度是 (M)O(M)
一共要二分 NN 次，每次二分的时间复杂度是 (logN)O(log⁡N)，所以二分的总时间复杂度是 (NlogN)O(Nlog⁡N)
要处理 QQ 次询问，每次询问的时间复杂度是 (1)O(1)，处理所有询问的时间复杂度就是 (Q)O(Q)
所以总的时间复杂度为 (NlogN+M+Q)O(Nlog⁡N+M+Q)
C++ 代码
#include <stdio.h>
#include <string.h>

typedef unsigned long long ULL;
const int N = 200005;
const ULL P = 131;

int n, m, q;                    // 题目中 N, M, Q
char A[N], B[N];                // 题目中 A, B
ULL hash_A[N], hash_B[N], p[N]; // hash_A, hash_B 分别存 A, B 的哈希值。p 存 P 的 i 次幂，用于求出每个子串的哈希值。
int cnt[N];                     // 二分预处理的 A 中每个后缀与 B 匹配的最长长度，存入 cnt

ULL get(ULL h[], int l, int r)  // 返回 h 中 [l, r] 的哈希值
{
    return h[r] - h[l - 1] * p[r - l + 1];
}

int main()
{
    scanf("%d%d%d\n", &n, &m, &q);
    scanf("%s\n%s", A + 1, B + 1); // 由于要处理哈希，从 1 开始输入会方便一些
    p[0] = 1;                      // 根据定义，P 的 0 次幂为 1
    for (int i = 1; i <= n; i ++ ) p[i] = p[i - 1] * P;                  // 预处理 p
    for (int i = 1; i <= n; i ++ ) hash_A[i] = hash_A[i - 1] * P + A[i]; // 预处理 hash_A
    for (int i = 1; i <= m; i ++ ) hash_B[i] = hash_B[i - 1] * P + B[i]; // 预处理 hash_B
    for (int i = 1; i <= n; i ++ ) // 二分预处理 cnt
    {
        int l = i, r = i + m, mid; // 二分左边界为 i，右边界为 i + m
        if (r > n + 1) r = n + 1;  // 如果右边界不在 A 中，让其指向 A 的右边界
        while (l < r)              // 二分板子
        {
            mid = l + r >> 1;
            if (get(hash_A, i, mid) != get(hash_B, 1, mid - i + 1)) r = mid;
            else    l = mid + 1;
        }
        cnt[r - i] ++ ; // 二分之后，r 表示的是 B 与 A 匹配的最靠后的位置（从 i 开始），r - i 是 A 从 i 开始的后缀与 B 匹配的最长长度
    }
    while (q -- )       // 处理询问
    {
        int x;
        scanf("%d", &x);
        printf("%d\n", cnt[x]);
    }
    return 0;
}
算法2
(KMPKMP) O(N+M+Q)O(N+M+Q)
这个解法的确比较难想。。需要对 KMPKMP 足够的熟悉。。
先对 BB 求 KMPKMP，得到 BB 的 nextnext 数组。
然后对 AA 做一遍匹配，回忆一下匹配的代码：

for (int i = 1, j = 0; i <= n; i ++ )
{
    while (j && a[i] != b[j + 1]) j = ne[j];
    if (a[i] == b[j + 1]) j ++ ;
    // blablabla
}
我们发现，在 blablabla 那个位置，jj 正好是 BB 能匹配 AA 的以 ii 为终点的最长字符串长度。
也就是说，字符串 AA 中，以 i−j+1i−j+1 为起点的与 BB 匹配的长度最小为 jj
但是，以 ii 为终点的，与 BB 匹配的字符串只有 A[i−j+1∼i]A[i−j+1∼i] 嘛？
不一定，我们发现 A[i−next[j]+1∼i]A[i−next[j]+1∼i] 也是与 BB 的前缀匹配的字符串
同理，A[i−next[next[j]]+1∼i]A[i−next[next[j]]+1∼i] 也是与 BB 的前缀匹配的字符串
⋯⋯
那么，我们在让 cnt[j] ++ 时，就还需要让 cnt[next[j]] ++，还需要让 cnt[next[next[j]]] ++⋯⋯
那我们匹配的时间复杂度就会退化为 (NM)O(NM) 了，显然是过不了这道题的。
观察下我们操作 cnt[x] 的过程，每次都会让 cnt[next[x]] ++，也就是说，cnt[x] ++了多少次，cnt[next[x]] ++也就要相应的执行多少次。
那么我们就可以先只操作 cnt[j] ++，最后从 mm 到 11 循环枚举一遍 cnt[i]，让 cnt[next[i]] += cnt[i] 即可。
注意最后 cnt[i] 存的是满足匹配的前缀至少为 xx 的后缀数量，而题目中所要求的满足匹配的前缀恰好为 xx 的答案的应为匹配的前缀至少为 x 的后缀数量 减去 匹配的前缀至少为 x + 1 的后缀数量，即 cnt[x] - cnt[x + 1]（后缀和思想），

时间复杂度
求 BB 的 nextnext 数组，时间复杂度为 (M)O(M)
将 AA 与 BB 做匹配，时间复杂度为 (N)O(N)
处理询问，时间复杂度为 (Q)O(Q)
故总的时间复杂度为 (N+M+Q)O(N+M+Q)
C++ 代码
#include <stdio.h>
#include <string.h>

const int N = 200005;

int n, m, q;
char A[N], B[N];
int ne[N], cnt[N];  // ne 存 B 的 next 数组，cnt 即上述 cnt 数组

int main()
{
    scanf("%d%d%d\n", &n, &m, &q);
    scanf("%s\n%s", A + 1, B + 1);

    for (int i = 2, j = 0; i <= m; i ++ ) // KMP 模板
    {
        while (j && B[i] != B[j + 1]) j = ne[j];
        if (B[i] == B[j + 1]) j ++ ;
        ne[i] = j;
    }

    for (int i = 1, j = 0; i <= n; i ++ ) // 将 A 与 B 做匹配
    {
        while (j && A[i] != B[j + 1]) j = ne[j];
        if (A[i] == B[j + 1]) j ++ ;
        cnt[j] ++ ;                       // 先只将 cnt[j] ++ 
    }
    for (int i = m; i; i -- ) cnt[ne[i]] += cnt[i]; // 从 m 到 1 枚举 cnt[i]，处理出所有的 cnt[next[i]]

    while (q -- )
    {
        int x;
        scanf("%d", &x);
        printf("%d\n", cnt[x] - cnt[x + 1]); // 输出的结果应为 cnt[x] - cnt[x + 1]
    }
    return 0;
}

作者：垫底抽风
链接：https://www.acwing.com/solution/content/15841/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

这道题是2015年北京大学数据结构与算法期末上机考试题(见http://dsa.openjudge.cn/final2015/6/)，然而即使是在有代码框架和注释的情况下，223人中也仅有6人AC，足见此题难度之大……


AcWing 160. 匹配统计——哈希暴力解&KMP正解    原题链接    中等
作者：    wuxigk ,  2020-01-22 02:41:54 ,  阅读 322

7


2
题目描述(简略版)
输入字符串A、BA、B,对于每一个询问xx,求恰与BB中前xx个字符匹配的AA的后缀个数.

样例
输入：

6 2 5
aabcde
ab
0
1
2
3
4
输出

4
1
1
0
0
算法1
(字符串哈希+二分) O(NlogM+M+Q)O(NlogM+M+Q)
预处理AA、BB的前缀哈希,枚举AA的所有后缀,二分确定匹配长度.

时间复杂度
前缀哈希预处理时间O(N+M)O(N+M),AA后缀个数O(N)O(N),每个后缀二分复杂度O(logM)O(logM),询问次数QQ,总时间复杂度O(NlogM+M+Q)O(NlogM+M+Q)
C++ 代码
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;
typedef unsigned long long ull;
vector<ull> prime;
vector<ull> get_hash(string& s)
{
    //求字符串的前缀哈希
    vector<ull> res(s.length() + 1);
    res[0] = 0;
    for (int i = 1; i <= s.length(); i++)
        res[i] = res[i - 1] * 131 + (s[i - 1] - 'a');
    return res;
}
ull substr_hash(vector<ull>& hsh, int l, int r)
{
    //求子串的哈希值
    if (l > r)//空串直接返回零
        return 0;
    return hsh[r] - hsh[l - 1] * prime[r - l + 1];
}
int main()
{
    int len1, len2, q;
    cin >> len1 >> len2 >> q;
    string a, b;
    cin >> a >> b;
    vector<int> res(len2 + 1);
    res.resize(len2 + 1);//res[i]表示匹配长度为i的后缀个数
    prime.resize(max(len1, len2));//用于储存素数的幂
    prime[0] = 1;
    for (int i = 1; i < prime.size(); i++)//计算素数幂
        prime[i] = prime[i - 1] * 131;
    //计算A、B的前缀哈希
    vector<ull> hash1, hash2;
    hash1 = get_hash(a);
    hash2 = get_hash(b);
    for (int i = 1; i <= len1; i++)//枚举A的后缀
    {
        //二分法确定匹配长度
        int l = 0, m, r = len2;
        while (l < r)
        {
            m = (l + r + 1) / 2;
            //比较两个串长度为m的前缀,如果两者相等说明匹配长度不小于m
            if (substr_hash(hash1, i, i + m - 1) == substr_hash(hash2, 1, m))
                l = m;
            else
                r = m - 1;
        }
        res[l]++;//l即为匹配长度,统计答案
    }
    for (int i = 0; i < q; i++)
    {
        int x;
        cin >> x;
        //如果x比B的长度还大,一定没有后缀满足
        cout << (x <= len2 ? res[x] : 0) << endl;
    }
    return 0;
}
算法2
(KMPKMP匹配) O(N+M+Q)O(N+M+Q)
用num[i,j]num[i,j]表示可作为BB的前缀的以A[j]A[j]结尾的长度为ii的AA的子串个数,注意到每个这样的子串都可以扩展为匹配长度不小于ii的后缀,而且对于固定的ii和不同的jj,这些后缀一定各不相同.因此
∑j=1Nnum[i,j]=defsum[i]
∑j=1Nnum[i,j]=defsum[i]
就是匹配长度不小于ii的后缀个数,匹配长度恰为ii的后缀个数即为
sum[i]−sum[i+1]
sum[i]−sum[i+1]

以AA为主串,BB为模式串进行KMPKMP匹配,在匹配过程中,假设AA与BB已经匹配的部分是A[pa−pb+1:pa]A[pa−pb+1:pa]和B[1:pb]B[1:pb],长度为pbpb,那么A[pa−pb+1:pa]A[pa−pb+1:pa]就是最长的可作为BB的前缀的以A[pa]A[pa]结尾的AA的子串,其长度为pbpb,也就是num[pb,pa]++num[pb,pa]++.考虑nextnext数组的性质,
B[1:next[pb]]=B[pb−next[pb]+1:pb]
B[1:next[pb]]=B[pb−next[pb]+1:pb]
因此
A[pa−next[pb]+1:pa]=B[pb−next[pb]+1:pb]=B[1:next[pb]]
A[pa−next[pb]+1:pa]=B[pb−next[pb]+1:pb]=B[1:next[pb]]
即A[pa−next[pb]+1:pa]A[pa−next[pb]+1:pa]是最长的可作为BB的前缀的以A[pa]A[pa]结尾的长度小于pbpb的AA的子串,其长度为next[pb]next[pb],也就是num[next[pb],pa]++num[next[pb],pa]++.以此类推可以不断做下去,最终可以不重不漏地找到所有可作为BB的前缀的以A[pa]A[pa]结尾的子串.在整个匹配过程中,papa遍历了AA,因此一趟匹配就可以求出numnum
但是这样做的时间复杂度和空间复杂度太大,需要加以优化.
(空间优化)首先注意到最终要获得的是sumsum,而sumsum是对numnum求和得到的,因此完全不需要开一个二维数组numnum,而可以直接在sumsum中进行统计,将num[pb,pa]++num[pb,pa]++变为sum[pb]++sum[pb]++.
(时间优化)在上面的算法中,对于特定的papa通过不断取nextnext数组,每一个子串都被单独算了一遍.注意到对于每一个可作为BB的前缀的长度为pbpb的AA的子串,对应地有一个可作为BB的前缀的长度为next[pb]next[pb]的AA的子串,而sumsum的计算并不区分papa,因此对于每一个papa,可以只统计长度为pbpb的那个子串,当一趟匹配完成后,自顶向下地将所有比它短的衍生的子串求出.具体处理详见代码.

时间复杂度
KMPKMP匹配过程时间复杂度O(N+M)O(N+M),询问次数QQ,总时间复杂度O(N+M+Q)O(N+M+Q)
C++ 代码
#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main()
{
    int n, m, q;
    cin >> n >> m >> q;
    string a, b;
    cin >> a >> b;
    b += '$';//加一个结束标志,简化KMP匹配过程
    vector<int> nxt(m + 1);//B的next数组
    vector<int> sm(m + 2);//sum数组,多开一个避免越界
    nxt[0] = -1;
    for (int i = 1; i <= m; i++)
    {
        //next数组计算
        int k = nxt[i - 1];
        while (k != -1 && b[k] != b[i - 1])
            k = nxt[k];
        nxt[i] = k + 1;
    }
    int pa = 0, pb = 0;//两个指针,分别指向A和B上待匹配的字符
    while (pa < n)
        if (pb == -1 || a[pa] == b[pb])
        {
            pa++;
            pb++;
            //每次更新pa后都需要统计sum数组
            sm[pb]++;
            //注意这里只统计最长的子串
            //其他由next数组获得的更短的子串在匹配结束后统一处理
        }
        else
            pb = nxt[pb];//匹配失败回溯
    for (int i = m; i > 0; i--)
        sm[nxt[i]] += sm[i];//自顶向下将之前未统计的子串也统计进去
    for (int i = 0; i < q; i++)
    {
        int x;
        cin >> x;
        //如果询问的长度比B的长度还大就直接输出0
        cout << (x <= m ? sm[x] - sm[x + 1] : 0) << endl;
    }
    return 0;
}

作者：wuxigk
链接：https://www.acwing.com/solution/content/7792/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


AcWing 160. 匹配统计    原题链接    中等
作者：    夏日 ,  2019-09-25 20:50:31 ,  阅读 346

4


神奇的题目，真出在比赛里估计只能写hash

首先，看题面统计后缀前缀之类的东西，先想暴力hash（划掉
然后我们来说正解，看到这样字符串前缀后缀相等的题目自然会想到KMP
考虑到其是统计后缀，我先尝试了下把字符串reverse一下再看看能否统计，尝试后发现不行
然后考虑正常KMP的过程能否转化
发现对于一个i，正常KMP求出了最大匹配长度f[i]，这是子串前缀匹配
尝试转化成后缀，自然想到匹配的是以i-f[i]+1开头的后缀，然而这个后缀可能并不是“最大”的，后面它可能还有后续，我们只能知道这次后缀匹配长度一定大于等于f[i]

然后是最难想的一步，由于我们只能统计答案中“大于等于某个值”的数量，所以考虑对答案求后缀和，即设cnt[i]为至少匹配了i这么长的后缀数量，询问时输出cnt[i] - cnt[i+1]
然后我们具体写出代码，发现对于一个i，需要累加所有能与它匹配的后缀，即f[i] nxt[f[i]] nxt[nxt[f[i]]] 一直到0, 这么跑下去复杂度又退化了，还不如直接写暴力，比上一步好想，考虑每次累加到一个匹配长度，那么它后面一直nxt到0的匹配长度都会被累加一次，所以我们再次利用差分的思想，直接在它身上加个1，最后累加起来即可，类似“区间修改”变成“两个单点修改” 。

本题正解有三步不是非常显然的转化，都需要有一定的“奇思妙想”，尤其是第二步，事实上难以想到（所以还是暴力hash部分分好

C++ 代码
#include <iostream>
#include <cstdio>

#define rint register int
#define lint long long
#define ull unsigned long long
#define isnum(x) ('0' <= (x) && (x) <= '9')
template<typename tint>
inline void readint(tint& x) {
    int f = 1; char ch = getchar(); x = 0;
    for(; !isnum(ch); ch = getchar()) if(ch == '-') f = -1;
    for(; isnum(ch); ch = getchar()) x = x * 10 + ch - '0';
    x *= f;
}
using namespace std;
const int maxn = 200000 + 10;
int n, m, t;
int snxt[maxn], f[maxn];
int cnt[maxn];
char s1[maxn], s2[maxn];

int main() {
    readint(n), readint(m), readint(t);
    scanf("%s %s", s1+1, s2+1);

    for(rint i=2, j=0; i<=m; i++) {
        while(j > 0 && s2[i] != s2[j+1]) j = snxt[j];
        if(s2[i] == s2[j+1]) j++;
        snxt[i] = j;
    }
    for(rint i=1, j=0; i<=n; i++) {
        while(j > 0 && (j == m || s1[i] != s2[j+1])) j = snxt[j];
        if(s1[i] == s2[j+1]) j++;
        f[i] = j, cnt[j]++;
    }
    for(rint i=m; i>=1; i--) cnt[snxt[i]] += cnt[i];
    int x = 0;
    while(t--) readint(x), printf("%d\n", cnt[x] - cnt[x+1]);
    return 0;
}

作者：夏日
链接：https://www.acwing.com/solution/content/952/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


AcWing 160. 匹配统计    原题链接    中等
作者：    yingzhaoyang ,  2019-12-01 17:13:55 ,  阅读 285

4


1
字符串哈希+二分
抽象一下题目,其实题目就是要让你求在a串中有多少个位置i,
使得a[i..n]和b[1..m]的最长公共前缀的长度恰好为x(注意是恰好,不能多,也不能少).

听说正解是扩展KMP....
但我不会呀!!!

于是上字符串哈希吧…
先预处理出k的幂次,a串的哈希值,b串的哈希值.
然后我们就要考虑如何求出a[i..n]和b[1..m]的最长公共前缀的长度.

枚举长度,计算哈希值???
明显会超时.

观察一下数据范围,n,m,q<=200000,感觉是个nlognnlogn的算法.
这不就是二分吗!!!!
直接二分最长匹配长度,计算哈希值即可.

时间复杂度 O(nlogn)O(nlogn)
C++ 代码
//相信奇迹的人,本身就和奇迹一样了不起.
#include<bits/stdc++.h>
#define N 200010
#define ll long long
using namespace std;
const ll k=131;
int n,m,q;
ll Hash_a[N],Hash_b[N],power_k[N];
ll cnt[N];
char a[N],b[N];
inline void init()//预处理k的幂次,a串的哈希值,b串的哈希值
{
    power_k[0]=1;
    for(int i=1;i<=max(n,m);i++) 
        power_k[i]=(power_k[i-1]*k);
    for(int i=1;i<=n;i++) 
        Hash_a[i]=(Hash_a[i-1]*k+a[i]);
    for(int i=1;i<=m;i++) 
        Hash_b[i]=(Hash_b[i-1]*k+b[i]);
}
inline ll get_Hash_a(int x,int y){//求a的子串a[x..y]的哈希值
    return Hash_a[y]-Hash_a[x-1]*power_k[y-x+1];
}
inline ll get_Hash_b(int x,int y){//求b的子串b[x..y]的哈希值
    return Hash_b[y]-Hash_b[x-1]*power_k[y-x+1];
}
int main()
{

    scanf("%d %d %d",&n,&m,&q);
    scanf("%s",a+1);
    scanf("%s",b+1);
    init();
    for(int i=1;i<=n;i++){
        if(a[i]!=b[1]){//如果第一个字符都不匹配,则最大匹配长度为0
            cnt[0]++;
            continue;
        }
        int l=1,r=min(m,n+1-i);
        while(l<r){//寻找最大匹配长度
            int mid=(l+r+1)>>1;
            if(get_Hash_b(1,mid)==get_Hash_a(i,i+mid-1))//如果当前可以匹配,向上缩短下界
                l=mid;
            else //否则向下缩短上界
                r=mid-1;
        }
        cnt[r]++;//当前长度的匹配位置+1
    }
     while(q--){
        int x;
        scanf("%d",&x);
        printf("%lld\n",cnt[x]);
    }
    return 0;
}

作者：yingzhaoyang
链接：https://www.acwing.com/solution/content/6668/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 160. 匹配统计    原题链接    中等
作者：    whsstory ,  2019-08-26 21:30:43 ,  阅读 280

3


可能是有O(n+m+q)O(n+m+q)的做法，但我只会O(nlogn)O(nlogn)的【我太菜了】

化简一下题意，就是对于串AA的每个后缀，求与串BB的最大相同前缀（也就是最长公共前缀）
最长公共前缀？那不就是《后缀数组》那题的求法吗？
先预处理A,BA,B的前缀Hash值，然后二分找最长公共前缀即可。
时间复杂度O(nlogn)O(nlogn)
（按理说字符串Hash要三Hash才稳，但为了代码简洁就base131%ull了（其实是懒））

//Wan Hong 2.2
//home
#include<iostream>
#include<cstdio>
#include<algorithm>
#include<cstring>
#include<queue>
#include<vector>
typedef long long ll;
typedef std::pair<ll,ll> pll;
typedef std::string str;
#define INF (1ll<<58)
ll read()
{
    ll f=1,x=0;
    char c=getchar();
    while(c<'0'||c>'9')
    {
        if(c=='-')f=-1;
        c=getchar();
    }
    while(c>='0'&&c<='9')x=x*10+c-'0',c=getchar();
    return f*x;
}
ll max(ll a,ll b)
{
    return a>b?a:b;
}
ll min(ll a,ll b)
{
    return a<b?a:b;
}
bool umax(ll& a,ll b)
{
    if(b>a)return a=b,1;
    return 0;
}
bool umin(ll& a,ll b)
{
    if(b<a)return a=b,1;
    return 0;
}

/**********/
#define MAXN 200011
typedef unsigned long long ull;
char a[MAXN],b[MAXN];
ull fa[MAXN],fb[MAXN],pw[MAXN];
ll n,m,q,c[MAXN];
ll get(ull* f,ull l,ull r)//在前缀Hash数组中求[l,r]的Hash值，O(1)
{
    return f[r]-f[l-1]*pw[r-l+1];
}
int main()
{
    n=read(),m=read(),q=read();
    scanf("%s%s",a+1,b+1);
    pw[0]=1;
    for(ll i=1;i<=n;++i)//预处理前缀Hash值
    {
        pw[i]=pw[i-1]*131;
        fa[i]=fa[i-1]*131+a[i]-'0';
    }
    for(ll i=1;i<=m;++i)fb[i]=fb[i-1]*131+b[i]-'0';
    for(ll i=1;i<=n;++i)
    {
        ull l=0,r=min(n-i+1,m),mid;
        while(l<r)//二分最长公共前缀
        {
            mid=(l+r+1)>>1;
            if(get(fa,i,i+mid-1)==get(fb,1,mid))l=mid;
            else r=mid-1;
        }
        ++c[l];
    }
    for(ll i=1;i<=q;++i)
    {
        ll x=read();
        printf("%lld\n",c[x]);
    }
    return 0;
}

作者：whsstory
链接：https://www.acwing.com/solution/content/4178/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
啥是三Hash啊？


whsstory   2个月前     回复
就选三个模数做hash，都相同才判定为相等


AcWing 160. 匹配统计    原题链接    中等
作者：    优雅的瑞尔 ,  2019-07-24 09:27:21 ,  阅读 434

2


1
正解未理解，先用哈希写。

题目描述
阿轩在纸上写了两个字符串，分别记为A和B。

利用在数据结构与算法课上学到的知识，他很容易地求出了“字符串A从任意位置开始的后缀子串”与“字符串B”匹配的长度。

不过阿轩是一个勤学好问的同学，他向你提出了Q个问题：

在每个问题中，他给定你一个整数x，请你告诉他有多少个位置，满足“字符串A从该位置开始的后缀子串”与B匹配的长度恰好为x。

例如：A=aabcde，B=ab，则A有aabcde、abcde、bcde、cde、de、e这6个后缀子串，它们与B=ab的匹配长度分别是1、2、0、0、0、0。

因此A有4个位置与B的匹配长度恰好为0，有1个位置的匹配长度恰好为1，有1个位置的匹配长度恰好为2。

输入格式
第一行输入三个整数N,M,Q，分别表示A串长度、B串长度、问题个数。

第二行输入字符串A，第三行输入字符串B。

接下来Q行每行输入1个整数x，表示一个问题。

输出格式
输出共Q行，依次表示每个问题的答案。

数据范围
1≤N,M,Q,x≤200000

样例
#include<bits/stdc++.h>

using namespace std;
typedef unsigned long long  ULL;

const int N=200000+10,P=131;
ULL h1[N],h2[N],p[N];
// h1[] 储存s1的哈希数组;
// h2[] 储存s2的哈希数组;
// p[] 储存p的i倍 (用来平衡倍数差，使之可以相减);
int n,m,q,K[N];
// K[] 储存该长度的点数;
char s1[N],s2[N];

ULL f1(int q,int h){  // 求s1[q]到s1[h]的哈希表示;
    return h1[h]-h1[q-1]*p[h-q+1];
}

ULL f2(int q,int h){  // 求s2[q]到s2[h]的哈希表示;
    return h2[h]-h2[q-1]*p[h-q+1];
}

int main(){
    scanf("%d%d%d%s%s",&n,&m,&q,s1+1,s2+1);//  输入;
    p[0]=1;
    for(int i=1;i<=n;i++) p[i]=p[i-1]*P;
    for(int i=1;i<=n;i++) h1[i]=h1[i-1]*P+s1[i];
    for(int i=1;i<=m;i++) h2[i]=h2[i-1]*P+s2[i];//  哈希处理;
    for(int i=1;i<=n;i++){
        if(s1[i]!=s2[1]){
            K[0]++;
            continue;
        }
        int l=1,r=min(m,n+1-i);
        while(l!=r){
            int zj=(l+r+1)/2;
            if(f2(1,zj)==f1(i,i+zj-1)) l=zj;
            else r=zj-1;
        }
        ++K[l];
    }
    while(q--){
        int k;
        scanf("%d",&k);
        printf("%d\n",K[k]);
    }
    return 0;
}
先将俩字符串用哈希储存。
再将以s1各个点为起点的匹配长度打表。
最后逐个输入长度并输出该长度的点数。
ps : 范围过大，需用二分。
如有不足，请多指教。

作者：优雅的瑞尔
链接：https://www.acwing.com/solution/content/3039/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 160. 匹配统计    原题链接    简单
作者：    shyyhs ,  2020-05-19 00:56:13 ,  阅读 100

0


这种字符相同的一般哈希还是好些,毕竟好懂嘛~代码也不长…下面也有注释.这题和那个排序的那个题目基本类似.还是很简单的,第一次在acwing写题解–有点不适应hh

/*
算法:哈希+二分
实现:
for(int i=1;i<=n;i++)//枚举a串的起点.然后二分a串寻找和b串长度最优匹配的,然后cnt一下最长长度
{

}
二分后的长度怎么检测呢?
假定我们已经处理完了h[n],我现在要求l~l+len
*/
#include <bits/stdc++.h>
using namespace std;
const int N=2e5+5;
typedef unsigned long long ull;
int cnt[N],base=131;
ull p[N],h1[N],h2[N];
char s1[N],s2[N];
ull get(int l,int r,ull h[]) { return h[r]-h[l-1]*p[r-l+1]; }
int main()
{
    int n,m,q;
    cin>>n>>m>>q;
    scanf("%s",s1+1);
    scanf("%s",s2+1);
    p[0]=1;
    for(int i=1;i<=n;i++) p[i]=p[i-1]*base;
    for(int i=1;i<=n;i++) { h1[i]=h1[i-1]*base+s1[i]-'a';}
    for(int i=1;i<=m;i++) { h2[i]=h2[i-1]*base+s2[i]-'a';}
    for(int i=1;i<=n;i++)
    {
        int l=0,r=min(n,m),ans=0;
        while(l<=r)
        {
            int mid=(l+r)>>1;
            if(get(i,i+mid,h1)==get(1,mid+1,h2)) { ans=max(ans,mid)+1; l=mid+1; }
            else               r=mid-1;
        }
        cnt[ans]++;
    }
    int x;
    while(q--)
    {
        cin>>x;
        cout<<cnt[x]<<endl;
    }
    return 0;
}

作者：shyyhs
链接：https://www.acwing.com/solution/content/12793/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。