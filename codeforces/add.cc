#include <iostream>
#include <cstdio>
using namespace std;

int add(int n);

int main()
{
    int n;

    freopen("io.in", "r", stdin);
    freopen("io.out", "w", stdout);

    cin >> n;

    cout << add(n);

    return 0;
}

int add(int n)
{
    if(n != 0)
        return n + add(n - 1);
    return 0;
}
