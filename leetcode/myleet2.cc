
class Solution {
public:
    
    const int mod = 1e9 + 7;
    
    
    long nCr(long n) {
        //r = 2
        long x = (n * (n - 1) / 2) % mod;
        return x;
    }
    
    int numWays(string s) {
        int n = s.length();
        
        vector<int> ones;
        int one_count = 0;
        for (int i = 0; i < n; i++) {
            if (s[i] == '1') {
                one_count++;
                ones.push_back(i);
            }
        }
        
        if (one_count % 3 != 0) return 0;
        if (!one_count) {
            return nCr(n - 1);
        }
        
        int first = one_count / 3 - 1;
        int second = one_count / 3 * 2 - 1;
        
        long zeros1 = ones[first + 1] - ones[first];
        long zeros2 = ones[second + 1] - ones[second];
        
        return (zeros1 * zeros2) % mod;
        
    }
};