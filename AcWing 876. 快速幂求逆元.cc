AcWing 876. 快速幂求逆元    原题链接    简单
作者：    AnrolsP ,  2020-09-10 21:35:29 ,  阅读 6

0


题目描述
费马定理


样例
3
4 3
8 5
6 3
算法1
(快速幂) O(logn)O(logn)
时间复杂度
参考文献
C++ 代码
#include<iostream>
using namespace std;
int res = 1;
int gcd(int a, int b)
{
    return b ? gcd(b, a % b) : a;
}
void q(int a, int b, int k)
{
    res = 1;
    while(b)
    {
        if(b & 1)res = (long long) res * a % k;
        b >>= 1;
        a = (long long)a * a % k;
    }
}

int main()
{
    int n;cin >> n;
    while(n--)
    {
        int b, m; cin >> b >> m;
        /*由费马定理可知，x = b ^ (m - 2) % m*/
        q(b, m - 2, m);
        /*题目限定了是质数，因此不互质只能是倍数条件*/
        if(gcd(b, m) == 1)cout << res << endl;
        else puts("impossible");
    }
    return 0;
}

作者：AnrolsP
链接：https://www.acwing.com/file_system/file/content/whole/index/content/1280384/


//https://www.geeksforgeeks.org/multiplicative-inverse-under-modulo-m/
