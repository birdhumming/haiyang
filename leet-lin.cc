/* weekly contest 201

 https://leetcode.com/contest/weekly-contest-201/problems/make-the-string-great/

https://leetcode.com/contest/weekly-contest-201/problems/find-kth-bit-in-nth-binary-string/

https://leetcode.com/contest/weekly-contest-201/problems/maximum-number-of-non-overlapping-subarrays-with-sum-equals-target/

https://leetcode.com/contest/weekly-contest-201/problems/minimum-cost-to-cut-a-stick/

*/

class Solution {
public:
    string makeGood(string s) {
        bool ch=1;
        while(ch) {
            ch=0;
            string t=s;
            for(int i=0; i<(int)s.size()-1; ++i) {
                if(s[i]+32==s[i+1]||s[i+1]+32==s[i]) {
                    t=s.substr(0, i)+s.substr(i+2);
                    ch=1;
                    break;
                }
            }
            s=t;
        }
        return s;
    }
};

class Solution {
public:
    string gen(int n) {
        if(n<2)
            return "0";
        string s=gen(n-1);
        string t=s;
        reverse(t.begin(), t.end());
        for(char& c : t)
            c^=1;
        return s+"1"+t;
    }
    char findKthBit(int n, int k) {
        return gen(n)[k-1];
    }
};


class Solution {
public:
    int maxNonOverlapping(vector<int>& nums, int target) {
        set<int> s;
        s.insert(0);
        int ps=0, ans=0;
        for(int a : nums) {
            ps+=a;
            if(s.find(ps-target)!=s.end()) {
                ++ans;
                s.clear();
            }
            s.insert(ps);
        }
        return ans;
    }
};

class Solution {
public:
    #define ll long long
    ll dp[102][102];
    int minCost(int n, vector<int>& cuts) {
        sort(cuts.begin(), cuts.end());
        cuts.insert(cuts.begin(), 0);
        cuts.push_back(n);
        for(int i=cuts.size()-1; ~i; --i) {
            for(int j=i+1; j<cuts.size(); ++j) {
                if(i+1<j) {
                    dp[i][j]=LLONG_MAX;
                    for(int k=i+1; k<j; ++k)
                        dp[i][j]=min(dp[i][j], dp[i][k]+dp[k][j]);
                }
                dp[i][j]+=cuts[j]-cuts[i];
            }
        }
        return dp[0][cuts.size()-1]-n;
    }
};

// weekly 200



class Solution {
public:
    int countGoodTriplets(vector<int>& arr, int a, int b, int c) {
        int n=arr.size(), ans=0;
        for(int i=0; i<n; ++i)
            for(int j=i+1; j<n; ++j)
                for(int k=j+1; k<n; ++k)
                    ans+=abs(arr[i]-arr[j])<=a&&abs(arr[j]-arr[k])<=b&&abs(arr[i]-arr[k])<=c;
        return ans;
    }
};

class Solution {
public:
    int getWinner(vector<int>& arr, int k) {
        int mx=0;
        for(int i=1, j=0; i<arr.size()&&j<k; ++i) {
            if(arr[0]>arr[i]) {
                ++j;
            } else {
                swap(arr[0], arr[i]);
                j=1;
            }
        }
        return arr[0];
    }
};

class Solution {
public:
    int minSwaps(vector<vector<int>>& grid) {
        int n=grid.size();
        vector<int> v(n);
        for(int i=0; i<n; ++i) {
            while(v[i]<n&&grid[i][n-v[i]-1]==0)
                ++v[i];
        }
        int ans=0;
        for(int i=0; i<n; ++i) {
            //n-1-i
            int j=i;
            while(j<n&&v[j]<n-1-i)
                ++j;
            if(j>=n)
                return -1;
            ans+=j-i;
            // cout << j << endl;
            while(j>i) {
                swap(v[j], v[j-1]);
                --j;
            }
        }
        return ans;
    }
};


class Solution {
public:
    #define ll long long
    int M=1e9+7;
    int maxSum(vector<int>& nums1, vector<int>& nums2) {
        int n=nums1.size(), m=nums2.size();
        vector<ll> dp1(n+1), dp2(m+1);
        for(int i=n-1, j=m-1; ~i||~j; ) {
            if(~i&&(j<0||nums2[j]<nums1[i])) {
                dp1[i]=dp1[i+1]+nums1[i];
                --i;
            } else if(~j&&(i<0||nums1[i]<nums2[j])) {
                dp2[j]=dp2[j+1]+nums2[j];
                --j;
            } else {
                dp1[i]=dp1[i+1]+nums1[i];
                dp2[j]=dp2[j+1]+nums2[j];
                dp1[i]=dp2[j]=max(dp1[i], dp2[j]);
                --i, --j;
            }
        }
        return max(dp1[0], dp2[0])%M;
    }
};

// biweekly 31

class Solution {
public:
    int f(int x) {
        //<x
        return x/2;
    }
    int countOdds(int low, int high) {
        return f(high+1)-f(low);
    }
};
class Solution {
public:
    int M=1e9+7;
    int numOfSubarrays(vector<int>& arr) {
        int s=0, ans=0;
        map<int, int> mp;
        mp[s]++;
        for(int a : arr) {
            s=(s+a)%2;
            ans=(ans+mp[s^1])%M;
            ++mp[s];
        }
        return ans;
    }
};
class Solution {
public:
    int numSplits(string s) {
        int n=s.size();
        vector<int> l(n+1), r(n+1);
        set<char> t;
        for(int i=0; i<n; ++i) {
            t.insert(s[i]);
            l[i+1]=t.size();
        }
        t.clear();
        for(int i=n-1; ~i; --i) {
            t.insert(s[i]);
            r[i]=t.size();
        }
        int ans=0;
        for(int i=1; i<n; ++i)
            if(l[i]==r[i])
                ++ans;
        return ans;
    }
};
class Solution {
public:
    int minNumberOperations(vector<int>& t) {
        int n=t.size();
        t.push_back(0);
        t.insert(t.begin(), 0);
        vector<int> l(n+2), r(n+2);
        for(int i=1; i<=n; ++i) {
            l[i]=i-1;
            while(l[i]&&t[l[i]]>t[i])
                l[i]=l[l[i]];
        }
        int ans=0;
        for(int i=n; i; --i) {
            r[i]=i+1;
            while(r[i]<=n&&t[r[i]]>=t[i])
                r[i]=r[r[i]];
            ans+=t[i]-max(t[l[i]], t[r[i]]);
        }
        return ans;
    }
};

//wekkly 198

class Solution {
public:
    int numWaterBottles(int numBottles, int numExchange) {
        int ans=0, ne=0;
        while(numBottles||numBottles+ne>=numExchange) {
            ans+=numBottles;
            int ne2=(numBottles+ne)%numExchange;
            numBottles=(numBottles+ne)/numExchange;
            ne=ne2;
        }
        return ans;
    }
};
class Solution {
public:
    vector<vector<int>> adj;
    string l;
    vector<int> ans;
    int c[26]={};
    void dfs(int u=0, int p=-1) {
        int lc=c[l[u]-'a']++;
        for(int v : adj[u])
            if(v^p)
                dfs(v, u);
        ans[u]=c[l[u]-'a']-lc;
    }
    vector<int> countSubTrees(int n, vector<vector<int>>& edges, string labels) {
        adj=vector<vector<int>>(n);
        for(auto e : edges)
            adj[e[0]].push_back(e[1]), adj[e[1]].push_back(e[0]);
        l=labels;
        ans=vector<int>(n);
        dfs();
        return ans;
    }
};

class Solution {
public:
    #define ar array
    bool adj[26][26], vis[26];
    vector<string> maxNumOfSubstrings(string s) {
        vector<int> p[26];
        for(int i=0; i<s.size(); ++i)
            p[s[i]-'a'].push_back(i);
        for(int i=0; i<26; ++i)
            for(int j=0; j<26; ++j)
                if(p[i].size()&&p[j].size()&&lower_bound(p[i].begin(), p[i].end(), p[j][0])!=lower_bound(p[i].begin(), p[i].end(), p[j].back()))
                    adj[j][i]=1;
        for(int k=0; k<26; ++k)
            for(int i=0; i<26; ++i)
                for(int j=0; j<26; ++j)
                    adj[i][j]|=adj[i][k]&&adj[k][j];
        vector<ar<int, 2>> v;
        for(int i=0; i<26; ++i) {
            if(p[i].size()) {
                bool ok=1;
                for(int j=0; j<26; ++j)
                    if(adj[i][j]&&!adj[j][i])
                        ok=0;
                if(ok)
                    v.push_back({p[i][0], p[i].back()});
            }
        }
        /*
        for(int i=0; i<26; ++i) {
            if(p[i].size()) {
                bool ok=1;
                for(int j=0; j<26; ++j)
                    if(j^i&&p[j].size()&&p[i][0]<=p[j][0]&&p[j].back()<=p[i].back()&&lower_bound(p[i].begin(), p[i].end(), p[j][0])==lower_bound(p[i].begin(), p[i].end(), p[j].back()))
                        ok=0;
                if(ok)
                v.push_back({p[i][0], p[i].back()});
            }
        }
        */
        bool ch=1;
        while(ch) {
            ch=0;
            for(int i=0; i<v.size(); ++i) {
                for(int j=0; j<v.size()&&!ch; ++j) {
                    if(i==j)
                        continue;
                    if(v[i][0]<=v[j][0]&&v[j][0]<=v[i][1]) {
                        auto a=v[i], b=v[j];
                        v.erase(find(v.begin(), v.end(), a));
                        v.erase(find(v.begin(), v.end(), b));
                        v.push_back({a[0], max(a[1], b[1])});
                        ch=1;
                    }
                }
            }
        }
        vector<string> ans;
        for(auto a : v)
            ans.push_back(s.substr(a[0], a[1]-a[0]+1));
        return ans;
    }
};


class Solution {
public:
    int closestToTarget(vector<int>& arr, int target) {
        set<int> s;
        int ans=1e9;
        for(int i=0; i<arr.size(); ++i) {
            set<int> ns;
            ns.insert(arr[i]);
            for(auto a : s)
                ns.insert(a&arr[i]);
            for(auto a : ns)
                ans=min(ans, abs(a-target));
            s=ns;
        }
        return ans;
    }
};

//biweek 30

class Solution {
public:
    string reformatDate(string date) {
        int n=date.size();
        string r=date.substr(n-4)+"-";
        string m=date.substr(n-8, 3);
        vector<string> v{"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
        int i=0;
        while(v[i]!=m)
            ++i;
        ++i;
        if(i<10)
            r+='0';
        r+=to_string(i)+"-";
        i=0;
        string r2;
        while(date[i]>='0'&&date[i]<='9') {
            r2+=date[i];
            ++i;
        }
        if(r2.size()<2)
            r2="0"+r2;
        return r+r2;
    }
};

class Solution {
public:
    int rangeSum(vector<int>& nums, int n, int left, int right) {
        vector<int> a;
        for(int i=0; i<n; ++i) {
            int s=0;
            for(int j=i; j<n; ++j) {
                s+=nums[j];
                a.push_back(s);
            }
        }
        sort(a.begin(), a.end());
        --left, --right;
        int s=0, M=1e9+7;
        for(int i=left; i<=right; ++i)
            s=(s+a[i])%M;
        return s;
    }
};

class Solution {
public:
    int minDifference(vector<int>& nums) {
        if(nums.size()<=4)
            return 0;
        sort(nums.begin(), nums.end());
        int ans=2e9;
        for(int i=3, j=nums.size()-1; ~i; --i, --j)
            ans=min(nums[j]-nums[i], ans);
        return ans;
    }
};

class Solution {
public:
    bool winnerSquareGame(int n) {
        vector<int> dp(n+1);
        for(int i=1; i<=n; ++i) {
            for(int j=1; j*j<=i; ++j)
                dp[i]|=!dp[i-j*j];
        }
        return dp[n];
    }
};

//weeky 186 no william lin taiwan