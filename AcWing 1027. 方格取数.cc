AcWing 1027. 方格取数    原题链接    简单
作者：    清南 ,  2019-09-15 13:44:27 ,  阅读 1162

19


7
算法1
(三维DP) O(n3)O(n3)
把 f[i1][j1][i2][j2]f[i1][j1][i2][j2] 转化为 f[k][i1][i2]f[k][i1][i2]
其中 k==i1+j1==i2+j2k==i1+j1==i2+j2
等价于 f[i1][k−i1][i2][k−i2]f[i1][k−i1][i2][k−i2]
这样在处理 两次走到相同点的时候，可以转化为

i1==i2i1==i2 或者 j1==j2j1==j2 判断其中之一即可

由f[i1][j1−1][i2][j2−1]f[i1][j1−1][i2][j2−1] 转化为f[k−1][i1][i2]f[k−1][i1][i2]
因为 k−1==i1+j1−1==i2+j2−1k−1==i1+j1−1==i2+j2−1
同理可得

f[i1−1][j1][i2−1][j2]==f[k−1][i1−1][i2−1]f[i1−1][j1][i2−1][j2]==f[k−1][i1−1][i2−1]
f[i1][j1−1][i2−1][j2]==f[k−1][i1][i2−1]f[i1][j1−1][i2−1][j2]==f[k−1][i1][i2−1]
f[i1−1][j1][i2][j2−1]==f[k−1][i1−1][i2]f[i1−1][j1][i2][j2−1]==f[k−1][i1−1][i2]
注意 kk的范围 2−n+n2−n+n，因为 刚开始的时候k==i1+j1==2k==i1+j1==2
j1 和 j2 要判断范围 ，因为 他们是从 k转化过来的，不能超过地图的边界范围

判断 两次取同一个格子的时候 满足以下条件
k==kk==k
i1==i2i1==i2
因此 j1==k−i1==j2==k−i2,w[i1][j2]==w[i2][j2]j1==k−i1==j2==k−i2,w[i1][j2]==w[i2][j2]
时间复杂度
O(n3)O(n3)
参考文献
算法提高课 DP 第一讲

C++ 代码
#include <bits/stdc++.h>
using namespace std;
const int N = 15;
int w[N][N],f[N*2][N][N];
int main(){
    ios::sync_with_stdio(0);
    cin.tie(0),cout.tie(0);
    int n ,x,y,z;
    cin >> n;
    while(cin >> x >> y >> z,x && y && z) w[x][y] = z;
    for(int k = 2;k <= n + n; ++k){
        for(int i1 = 1;i1 <= n; ++i1){
            for(int i2 = 1;i2 <= n; ++i2){
                int j1 = k - i1 , j2 = k - i2;
                if(j1 >= 1 && j1 <= n && j2 >= 1 && j2 <= n){
                    int t = w[i1][j1];
                    if(i1 != i2) t += w[i2][j2];
                    int &x = f[k][i1][i2];
                    x = max(x , f[k-1][i1-1][i2-1] + t);
                    x = max(x , f[k-1][i1-1][i2] + t);
                    x = max(x , f[k-1][i1][i2-1] + t);
                    x = max(x , f[k-1][i1][i2] + t);
                }
            }
        }
    }
    cout << f[n+n][n][n] << endl;


    return 0;
}

作者：清南
链接：https://www.acwing.com/solution/content/4578/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



1

#include <iostream>
#include <algorithm>

using namespace std;

const int N = 15;

int n;
int w[N][N];
int f[N * 2][N][N];

int main()
{
    scanf("%d", &n);

    int a, b, c;
    while (cin >> a >> b >> c, a || b || c) w[a][b] = c;

    for (int k = 2; k <= n + n; k ++ )
        for (int i1 = 1; i1 <= n; i1 ++ )
            for (int i2 = 1; i2 <= n; i2 ++ )
            {
                int j1 = k - i1, j2 = k - i2;
                if (j1 >= 1 && j1 <= n && j2 >= 1 && j2 <= n)
                {
                    int t = w[i1][j1];
                    if (i1 != i2) t += w[i2][j2];
                    int &x = f[k][i1][i2];
                    x = max(x, f[k - 1][i1 - 1][i2 - 1] + t);
                    x = max(x, f[k - 1][i1 - 1][i2] + t);
                    x = max(x, f[k - 1][i1][i2 - 1] + t);
                    x = max(x, f[k - 1][i1][i2] + t);
                }
            }

    printf("%d\n", f[n + n][n][n]);
    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/112798/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。