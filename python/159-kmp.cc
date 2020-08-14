AcWing 159. 奶牛矩阵（kmp）    原题链接    中等
作者：    羽笙 ,  2019-08-07 11:32:39 ,  阅读 380

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


作者：羽笙
链接：https://www.acwing.com/solution/content/3528/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
