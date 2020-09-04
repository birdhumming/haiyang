def findIntegers(self, num):
        """
        :type num: int
        :rtype: int
        """
        dp = [1, 2]
        for x in range(2, 32):
            dp.append(dp[x - 1]+ dp[x - 2])
        bnum = bin(num)[2:]
        size = len(bnum)
        ans = dp[size]
        for idx in range(1, size):
            if bnum[idx] == bnum[idx - 1] == '1':
                break
            if bnum[idx] == bnum[idx - 1] == '0':
                ans -= dp[size - idx] - dp[size - idx - 1]
        return ans
