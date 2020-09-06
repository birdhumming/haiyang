class Solution {
public:
    int findLengthOfShortestSubarray(vector<int>& a) {
        int n=a.size();
        vector<int> b;
        for(int i=n-1; ~i; --i)
            if(!b.size()||a[i]<=b.back())
                b.push_back(a[i]);
            else
                break;
        reverse(b.begin(), b.end());
        int ans=n-b.size(), l=-1;
        for(int i=0; i<n; ++i) {
            if(a[i]<l)
                break;
            l=a[i];
            ans=min(n-(i+1+(int)b.size()-(int)(lower_bound(b.begin(), b.end(), a[i])-b.begin())), ans);
        }
        return max(ans, 0);
    }
};