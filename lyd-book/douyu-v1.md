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
