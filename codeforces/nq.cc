#include <fstream>
using namespace std;

int n, queen[100];

int check(int y, int x)
{
    for (int i=0; i<y; i++) {
        if (i==y || queen[i]==x || i-y==queen[i]-x || i-y==x-queen[i]) {
            return 1;
        }
    }
    return 0;
}

int rec(int y)
{
    if (y>=n) return 1;
    int ans=0;
    for (int i=0; i<n; i++) {
        if (!check(y, i)) {
            queen[y]=i;
            ans+=rec(y+1);
        }
    }
    return ans;
}

int main()
{
    cin>>n;
    cout<<rec(0)<<'\n';
}
