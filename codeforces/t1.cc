#include <iostream>
#include <fstream>
#include <cstring>

using namespace std;

int memo[21][21][21];
int find_point_value(int x, int y, int z);

int main()
{
    freopen("io.in", "r", stdin);
//    freopen("io.out", "w", stdout);

    memset(memo, -1, sizeof memo);

    int x, y, z;
    cin >> x >> y >> z;
    cout << x << " " << y << " " <<z <<'\n';	
//    cout << find_point_value(x, y, z);
}

int find_point_value(int x, int y, int z)
{
    int ans = 0;
    if (memo[x][y][z] != -1)
    {
        return memo[x][y][z];
    }
    if (x < 1 || y < 1 || z < 1)
    {
        ans = 1;
    }
    else if (x > 20 || y > 20 || z > 20)
    {
        ans = find_point_value(20, 20, 20);
    }
    else if (x < y && y < z)
    {
        ans = find_point_value(x, y, z-1) + find_point_value(x, y-1, z-1) - find_point_value(x, y-1, z);
    }
    else
    {
        ans = find_point_value(x-1, y, z) + find_point_value(x-1, y-1, z) + find_point_value(x-1, y, z-1) - find_point_value(x-1, y-1, z-1);
    }
    //cout << x << " " << y << " " << z << " " << ans <<'\n';
    memo[x][y][z] = ans;
    return ans;
}
