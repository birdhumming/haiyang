class Solution {
public:
    int M=1e9+7;
    int countRoutes(vector<int>& l, int start, int finish, int f) {
        int n=l.size();
        vector<vector<int>> dp(f+1, vector<int>(n));
        dp[0][start]=1;
        for(int i=0; i<f; ++i) {
            for(int j=0; j<n; ++j) {
                for(int k=0; k<n; ++k) {
                    if(k==j||i+abs(l[k]-l[j])>f)
                        continue;
                    dp[i+abs(l[k]-l[j])][k]=(dp[i+abs(l[k]-l[j])][k]+dp[i][j])%M;
                }
            }
        }
        int ans=0;
        for(int i=0; i<=f; ++i)
            ans=(ans+dp[i][finish])%M;
        return ans;
    }
};