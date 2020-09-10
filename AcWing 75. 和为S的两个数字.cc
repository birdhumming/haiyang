AcWing 75. 和为S的两个数字 ---简洁    原题链接    简单
作者：    巨鹿噜噜噜路 ,  2020-06-01 18:03:18 ,  阅读 128

1


C++ 代码
class Solution {
public:
    vector<int> findNumbersWithSum(vector<int>& nums, int target) {
        unordered_map<int, bool> hash;
        for(auto &x:nums) {
            if(hash[target - x]) return vector<int>{x, target - x};
            else hash[x] = true;
        }
    }
};

作者：巨鹿噜噜噜路
链接：https://www.acwing.com/solution/content/14055/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



1


双指针
C++ 代码
class Solution {
public:

    vector<int> findNumbersWithSum(vector<int>& nums, int target) {
        sort(nums.begin(),nums.end());
        for(int i = 0 ,j = nums.size() - 1; i <j;){
            if(nums[i] +nums[j] == target)
               return  vector<int>{nums[i],nums[j]};
            else if(nums[i] + nums[j] < target)
                i++;
            else 
                j--;
        }
    }
};

作者：yzm0211
链接：https://www.acwing.com/solution/content/1365/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1
算法1
(暴力枚举) O(n)O(n)
使用hash表来记录是否已经出现过。

时间复杂度分析：blablabla

C++ 代码
class Solution {
public:
    vector<int> findNumbersWithSum(vector<int>& nums, int target) {
        unordered_set<int> hash;
        for(auto x:nums)
        {
            if(hash.count(target-x)) return vector<int>{x,target-x};
            hash.insert(x);
        }

    }
};
算法2
(暴力枚举) O(n2)

作者：季科
链接：https://www.acwing.com/solution/content/2136/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

1. python3

python3 代码
class Solution(object):
    def findNumbersWithSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = dict()
        for num in nums:
            if num in d.keys():
                return [target-num,num]
            else:
                d[target-num] = num

作者：polaris
链接：https://www.acwing.com/solution/content/3557/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。