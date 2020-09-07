cWing 152. 城市游戏(悬线法）    原题链接    中等
作者：    Tyouchie ,  2019-10-31 11:45:39 ,  阅读 225

0


说白了 这个题目就是dp中悬线法的一个求解最大合法矩阵的一个问题
可以使用单调栈优化 不过 直接n^2 没有问题
https://www.cnblogs.com/Tyouchie/p/11382288.html
曾经总结过的小计 如果不熟悉 可以学习一下 不过如果会的
直接看code8

C++ 代码
#include<bits/stdc++.h>
using namespace std;
int n,m,a[1010][1010],l[1010][1010],r[1010][1010],up[1010][1010];
char s[1010][1010]; 
template<typename T>inline void read(T &x) {
    x=0;T f=1,ch=getchar();
    while(!isdigit(ch)) {if(ch=='-') f=-1;ch=getchar();}
    while(isdigit(ch)) {x=(x<<1)+(x<<3)+(ch^48);ch=getchar();}
    x*=f;
}
int main() {
    read(n); read(m);
    for(int i=1;i<=n;i++) {
        scanf("%s",s+1);
        for(int j=1;j<=m;j++) {
            if(s[i][j]=='F') a[i][j]=0;
            else a[i][j]=1;
            l[i][j]=r[i][j]=j;
            up[i][j]=1;
        }
    }
    for(int i=1;i<=n;i++) {
        for(int j=2;j<=m;j++) {
            if(!a[i][j]&&!a[i][j-1]) {
                l[i][j]=l[i][j-1];
            }
        }
    }
    for(int i=1;i<=n;i++) {
        for(int j=m-1;j>=1;j--) {
            if(!a[i][j]&&!a[i][j+1]) {
                r[i][j]=r[i][j+1];
            }
        }
    }
    int ans=0;
    for(int i=1;i<=n;i++) {
        for(int j=1;j<=m;j++) {
            if(i>1&&!a[i][j]&&!a[i-1][j]) {
                l[i][j]=max(l[i][j],l[i-1][j]);
                r[i][j]=max(r[i][j],r[i-1][j]);
                up[i][j]=up[i-1][j]+1;
            }
            ans=max(ans,(r[i][j]-l[i][j]+1)*up[i][j]);
        }
    }
    cout<<ans*3<<endl;
    return 0;
}

作者：Tyouchie
链接：https://www.acwing.com/solution/content/4873/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
