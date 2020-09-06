#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

int n;
vector<int> nums;

int find(int target)
{
    cout<<"finding\n";
    int ans = -1;
    int diff = 0x3f3f3f3f;
    for (int i = 0; i <= 1000; i++) {
        int curdiff = abs(0 - (target - pow(i, n - 1)));
        if (curdiff < diff) {
            diff = curdiff;
            ans = i;
        }
    }
    return ans;
}

void solve(int x)
{
    double ans = 0;
    for (int i = 0; i < n; i++) {
        int diff = abs(nums[i] - pow(x, i));
        ans += diff;
    }

    cout<<ans<<endl;
}

int main()
{
    cin>>n;
    nums.resize(n);
    for (int i = 0; i < n; i++) cin>>nums[i];
    sort(nums.begin(), nums.end());

    int x = find(nums[nums.size() - 1]);
    cout<<"x = "<<x<<endl;
    solve(x);
}
