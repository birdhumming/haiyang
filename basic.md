作者：CYa搜索DP中
链接：https://www.acwing.com/file_system/file/content/whole/index/content/1337042/

https://www.acwing.com/file_system/file/content/whole/index/content/1337130/

DP https://www.acwing.com/file_system/file/content/whole/index/content/1336954/

图论和搜索【Py3模板】
https://www.acwing.com/file_system/file/content/whole/index/content/1336883/

字节跳动广告技术部-广告算法凉经
https://www.acwing.com/file_system/file/content/whole/index/content/1339859/
算法题
//扭转后的有序数组，找到指定的数，返回下标，找不到时返回-1
//样例输入：3，4，5，6，7，1，2 查找数字7
//样例输出：4
算法题
有个数组，寻找一个切分点，使左右两个数组的方差差距最小
min abs（a-b）
算法题
二叉树之字形遍历
进程和线程的区别
你了解什么深度学习的优化器
如何抑制过拟合
LR的损失函数，推导一下
为什么会产生梯度消失和梯度爆炸，如何解决？


BFS 模板存档（NOI官网例题 电梯）

```
#include<bits/stdc++.h>
using namespace std;
int main()
{   
    //bfs
    int n,st,en,a[205],d[205][3],bz[205],c[2]={-1,1};
    scanf("%d%d%d",&n,&st,&en);
    if(st == en)
    {
        printf("%d",0);
        return 0;
    }

    memset(d,0,sizeof(d));
    memset(bz,0,sizeof(bz));

    for(int i=1;i<=n;i++) scanf("%d",&a[i]);

    d[1][1] = st; //楼层
    d[1][2] = 0; //到达这一层要的步数
    bz[st] = 1;

    int i=0,j=1;
    while(i<j)
    {
        i++;
        for(int t = 0;t<=1;t++)
        {
            int k = d[i][1]+a[d[i][1]]*c[t];
            if(k>=1 && k<=n)
            {
                j++;
                d[j][1] = k;
                d[j][2] = d[i][2]+1;
                if(bz[k] == 0)
                {
                    bz[k] =1;
                    if(k == en)
                    {
                        printf("%d",d[j][2]);
                        return 0;
                    }
                }
                else j--; 
            }
        }
    }
    printf("%d",-1);
    return 0;
}
```

算法
(二分) O(logn)O(logn)
这道题目给定的是递增数组，假设数组中第一个缺失的数是 xx，那么数组中的数如下所示；


从中可以看出，数组左边蓝色部分都满足nums[i] == i，数组右边橙色部分都不满足nums[i] == i，因此我们可以二分出分界点 xx 的值。

另外要注意特殊情况：当所有数都满足nums[i] == i时，表示缺失的是 nn。

时间复杂度分析
二分中的迭代只会执行 O(logn)O(logn) 次，因此时间复杂度是 O(logn)O(logn)。

C++ 代码
```
class Solution {
public:
    int getMissingNumber(vector<int>& nums) {
        if (nums.empty()) return 0;

        int l = 0, r = nums.size() - 1;
        while (l < r)
        {
            int mid = l + r >> 1;
            if (nums[mid] != mid) r = mid;
            else l = mid + 1;
        }

        if (nums[r] == r) r ++ ;
        return r;
    }
};
```

AcWing 67. 三种stl方法：multiset、遍历vector、lower/upper_bound函数    原题链接    简单
作者：    醉生梦死 ,  2020-07-11 11:15:54 ,  阅读 129

7


题目描述
求一个排好序的数组中k的个数

解题思路
目的是练习stl和常用的库函数

题解一：使用有序多重集合multiset
```
class Solution {
public:
    int getNumberOfK(vector<int>& nums , int k) {
        multiset<int> s;

        for(int x : nums) s.insert(x);

        return s.count(k);
    }
};
```
题解二：遍历vector，计数
```
class Solution {
public:
    int getNumberOfK(vector<int>& nums , int k) {
        int cnt = 0;
        for(int x : nums)
            if(x == k)
                cnt++;
        return cnt;
    }
};
```
题解三：使用lower_bound和upper_bound,指针运算得出次数
```
class Solution {
public:
    int getNumberOfK(vector<int>& nums , int k) {

        auto l = lower_bound(nums.begin(), nums.end(), k);
        auto r = upper_bound(nums.begin(), nums.end(), k);

        return r - l;
    }
};
```

(双指针扫描) O(n)O(n)
用两个指针分别从首尾开始，往中间扫描。扫描时保证第一个指针前面的数都是奇数，第二个指针后面的数都是偶数。

每次迭代时需要进行的操作：

第一个指针一直往后走，直到遇到第一个偶数为止；
第二个指针一直往前走，直到遇到第一个奇数为止；
交换两个指针指向的位置上的数，再进入下一层迭代，直到两个指针相遇为止；
时间复杂度
当两个指针相遇时，走过的总路程长度是 nn，所以时间复杂度是 O(n)O(n)。

C++ 代码
```
class Solution {
public:
    void reOrderArray(vector<int> &array) {
         int l = 0, r = array.size() - 1;
         while (l < r) {
             while (l < r && array[l] % 2 == 1) l ++ ;
             while (l < r && array[r] % 2 == 0) r -- ;
             if (l < r) swap(array[l], array[r]);
         }
    }
};
```

(遍历链表) O(n)O(n)
单链表只能从前往后遍历，不能从后往前遍历。 因此我们先从前往后遍历一遍输入的链表，将结果记录在答案数组中。最后再将得到的数组逆序即可。
时间复杂度分析
链表和答案数组仅被遍历了常数次，所以总时间复杂度是 O(n)O(n)。

C++ 代码
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> printListReversingly(ListNode* head) {
        vector<int> res;
        while (head) {
            res.push_back(head->val);
            head = head->next;
        }
        return vector<int>(res.rbegin(), res.rend());
    }
};
```

AcWing 20. 用两个栈实现队列    原题链接    简单
算法
(栈，队列) O(n)O(n)
这是一道基础题，只要把功能实现对就可以，不需要考虑运行效率。

我们用两个栈来做，一个主栈，用来存储数据；一个辅助栈，用来当缓存。

push(x)，我们直接将x插入主栈中即可。
pop()，此时我们需要弹出最先进入栈的元素，也就是栈底元素。我们可以先将所有元素从主栈中弹出，压入辅助栈中。则辅助栈的栈顶元素就是我们要弹出的元素，将其弹出即可。然后再将辅助栈中的元素全部弹出，压入主栈中。
peek()，可以用和pop()操作类似的方式，得到最先压入栈的元素。
empty()，直接判断主栈是否为空即可。
时间复杂度分析
push()：O(1)O(1)；
pop(): 每次需要将主栈元素全部弹出，再压入，所以需要 O(n)O(n) 的时间；
peek()：类似于pop()，需要 O(n)O(n) 的时间；
empty()：O(1)O(1)；
C++ 代码
```
class MyQueue {
public:
    /** Initialize your data structure here. */
    stack<int> stk, cache;
    MyQueue() {

    }

    /** Push element x to the back of queue. */
    void push(int x) {
        stk.push(x);
    }

    void copy(stack<int> &a, stack<int> &b) {
        while (a.size()) {
            b.push(a.top());
            a.pop();
        }
    }

    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        copy(stk, cache);
        int res = cache.top();
        cache.pop();
        copy(cache, stk);
        return res;
    }

    /** Get the front element. */
    int peek() {
        copy(stk, cache);
        int res = cache.top();
        copy(cache, stk);
        return res;
    }

    /** Returns whether the queue is empty. */
    bool empty() {
        return stk.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty();
 */

```

AcWing 53. 最小的k个数    原题链接    简单
作者：    nihaotian ,  2019-09-23 21:37:04 ,  阅读 1384

10


2
题目描述
输入n个整数，找出其中最小的k个数。

注意：

数据保证k一定小于等于输入数组的长度;
输出数组内元素请按从小到大顺序排序;

样例
输入：[1,2,3,4,5,6,7,8] , k=4

输出：[1,2,3,4]
算法1
(快速选择) O(klogn)O(klogn)
运用快速排序的思想，每次快速选择会将一个数放置到正确的位置（即满足左边的数都比它小，右边的数都比它大），因此我们可以对原数组做k次快速选择。

时间复杂度分析：一次快速选择的时间复杂度是O(logn)O(logn)，进行k次，时间复杂度为O(klogn)O(klogn)
C++ 代码
```
class Solution {
public:
    vector<int> getLeastNumbers_Solution(vector<int> input, int k) {
        vector<int> res;
        for(int i = 1;i <= k;i++)
            res.push_back(quick_select(input,0,input.size()-1,i));
        return res;
    }

    int quick_select(vector<int>& q,int l,int r,int k)
    {
        if(l >= r) return q[l];
        int i = l-1,j = r+1,x = q[l+r >> 1];
        while(i < j)
        {
            do i++;while(q[i] < x);
            do j--;while(q[j] > x);
            if(i < j) swap(q[i],q[j]);
        }
        if(k <= j-l+1) return quick_select(q,l,j,k);
        else return quick_select(q,j+1,r,k-(j-l+1));
    }

};
```
算法2
(堆排序) O(nlogk)O(nlogk)
维护一个大小为k的大根堆，将数组元素都push进堆，当堆中的数大于k时弹出堆顶元素。注意弹出堆顶的顺序是从大到小的k个数，要进行逆序操作

时间复杂度分析：建堆的时间复杂度是O(logk)O(logk)，要进行n次建堆的操作。

C++ 代码 yxc idea
```
class Solution {
public:
    vector<int> getLeastNumbers_Solution(vector<int> input, int k) {
        vector<int> res;
        priority_queue<int> heap;
        for(auto x : input)
        {
            heap.push(x);
            if(heap.size() > k) heap.pop(); 
        }
        while(heap.size())
        {
            res.push_back(heap.top());
            heap.pop();
        }
        reverse(res.begin(),res.end());
        return res;
    }
};
```

AcWing 75. 和为S的两个数字    原题链接    简单
作者：    yzm0211 ,  2019-04-04 22:16:12 ,  阅读 820

1


双指针
C++ 代码
```
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
```
blablabla
哈希表
C++ 代码
blablabla

作者：yzm0211
链接：https://www.acwing.com/solution/content/1365/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


permuttions with and without repeating numbers
https://www.acwing.com/solution/leetcode/content/124/
https://www.acwing.com/solution/LeetCode/content/126/
https://www.acwing.com/solution/content/776/
https://www.acwing.com/solution/content/1155/

AcWing 26. 二进制中1的个数    原题链接    简单
作者：    yxc ,  2019-01-06 02:01:39 ,  阅读 3230

19


11
算法
(位运算) O(logn)O(logn)
迭代进行如下两步，直到 nn 变成0为止：

如果 nn 在二进制表示下末尾是1，则在答案中加1；
将 nn 右移一位，也就是将 nn 在二进制表示下的最后一位删掉；
这里有个难点是如何处理负数。
在C++中如果我们右移一个负整数，系统会自动在最高位补1，这样会导致 nn 永远不为0，就死循环了。
解决办法是把 nn 强制转化成无符号整型，这样 nn 的二进制表示不会发生改变，但在右移时系统会自动在最高位补0。

时间复杂度
每次会将 nn 除以2，最多会除 lognlogn 次，所以时间复杂度是 O(logn)O(logn)。

C++ 代码
```
class Solution {
public:
    int NumberOf1(int n) {
        int res = 0;
        unsigned int un = n; 
        while (un) res += un & 1, un >>= 1;
        return res;
    }
};
```

作者：yxc
链接：https://www.acwing.com/solution/content/732/


2.stl:map 不用auto的写法 (蓝桥杯不让写auto)

Note:
printf函数输出字符串是针对char *的，即printf只能输出c语言的内置数据类型，而string不是c语言的内置数据类型。如需输出string对象中的字符串，可以使用string的成员函数c_str()，该函数返回字符串的首字符的地址。

map 正向迭代器

map<int, PII>::iterator iter; //迭代器
for (iter = ans.begin(); iter != ans.end(); iter ++ ){}
4. STL:vector 这里用了pair<int, pair<double, string >> 嵌套pair构成三元组

https://www.acwing.com/solution/content/9475/

leetcode 191, acwing 26 is better - casting of int to unsigned int;
https://stackoverflow.com/questions/17358445/why-does-right-shifting-negative-numbers-in-c-bring-1-on-the-left-most-bits

https://leetcode.com/problems/number-of-1-bits/
leetcode 338 - not done
https://leetcode.com/problems/counting-bits/



AcWing 21. 斐波那契数列    原题链接    简单
作者：    yxc ,  2019-01-06 00:00:29 ,  阅读 3320

20


5
算法
(递推) O(n)O(n)
这题的数据范围很小，我们直接模拟即可。
当数据范围很大时，就需要采用其他方式了，可以参考 求解斐波那契数列的若干方法 。

用两个变量滚动式得往后计算，aa 表示第 n−1n−1 项，bb 表示第 nn 项。
则令 c=a+bc=a+b 表示第 n+1n+1 项，然后让 a,ba,b 顺次往后移一位。

时间复杂度分析
总共需要计算 nn 次，所以时间复杂度是 O(n)O(n) 。

C++ 代码
```
class Solution {
public:
    int Fibonacci(int n) {
        int a = 0, b = 1;
        while (n -- ) {
            int c = a + b;
            a = b, b = c;
        }
        return a;
    }
};
```



O(N) dp大内存法

```
class Solution {
public:
    int Fibonacci(int n) {
        int dp[n+1];
        dp[0] =0; dp[1] = 1;
        for(int i=2;i<=n;i++) {
        dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
};
```

算法1
(线性扫描) O(n)O(n)
这个题在C++里比较好做，我们可以从前往后枚举原字符串：

如果遇到空格，则在string类型的答案中添加 "%20"；
如果遇到其他字符，则直接将它添加在答案中；
但在C语言中，我们没有string这种好用的模板，需要自己malloc出char数组来存储答案。
此时我们就需要分成三步来做：

遍历一遍原字符串，计算出答案的最终长度；
malloc出该长度的char数组；
再遍历一遍原字符串，计算出最终的答案数组；
时间复杂度分析
原字符串只会被遍历常数次，所以总时间复杂度是 O(n)O(n)。

C++ 代码
```
class Solution {
public:
    string replaceSpaces(string &str) {
        string res;
        for (auto x : str)
            if (x == ' ')
                res += "%20";
            else
                res += x;
        return res;
    }
};
```
算法2
(双指针扫描) O(n)O(n)
在部分编程语言中，我们可以动态地将原数组长度扩大，此时我们就可以使用双指针算法，来降低空间的使用：

首先遍历一遍原数组，求出最终答案的长度length；
将原数组resize成length大小；
使用两个指针，指针i指向原字符串的末尾，指针j指向length的位置；
两个指针分别从后往前遍历，如果str[i] == ' '，则指针j的位置上依次填充'0', '2', '%'，这样倒着看就是"%20"；如果str[i] != ' '，则指针j的位置上填充该字符即可。
由于i之前的字符串，在变换之后，长度一定不小于原字符串，所以遍历过程中一定有i <= j，这样可以保证str[j]不会覆盖还未遍历过的str[i]，从而答案是正确的。

时间复杂度分析
原字符串只会被遍历常数次，所以总时间复杂度是 O(n)O(n)。

C++ 代码
```
class Solution {
public:
    string replaceSpaces(string &str) {

        int len = 0;
        for (auto c : str)
            if (c == ' ')
                len += 3;
            else
                len ++ ;

        int i = str.size() - 1, j = len - 1;

        str.resize(len);

        while (i >= 0)
        {
            if (str[i] == ' ')
            {
                str[j -- ] = '0';
                str[j -- ] = '2';
                str[j -- ] = '%';
            }
            else str[j -- ] = str[i];
            i -- ;
        }
        return str;
    }
};
```

练习STL
```
string replaceSpaces(string &str) {
        size_t t;
        while((t=str.find(' '))!=string::npos){
            str.replace(t,1,"%20");
        }
        return str;
    }
```

不用改动字符串，使用string中的replace函数即可。
```
class Solution {
public:
    string replaceSpaces(string &str) 
    {
        for(int i=0;i<str.size();i++)
            if(str[i]==' ')
                str.replace(i,1,"%20") ;
        return str ;
    }
};
```
李乾   8个月前     回复
Q:时间复杂度=O(?)


李乾   8个月前    回复了 李乾 的评论     回复
@yxc


yxc   8个月前    回复了 李乾 的评论     回复
这里取决于replace函数的时间复杂度，由于每次执行该函数，会将一个字符变成三个字符，所以会涉及到数组的整体拷贝，那么每执行一次replace最坏情况下需要 O(n)O(n) 的时间，所以总时间复杂度最坏是 O(n2)O(n2)。

yls，模仿基础课提供的解法，这样的双指针更加简洁：
```
class Solution {
public:
    string replaceSpaces(string &str) {
        string res;
        for (int i = 0; str[i]; i++) {
            int j = i;
            while (j < str.size() && str[j] != ' ') j++;
            res += str.substr(i, j - i);
            if (j < str.size()) res += "%20";
            i = j;
        }
        return res;
    }
};
```
yxc   11个月前     回复
不错！


即将升入大三的菜鸡   8个月前     回复
这样申请了额外空间，就跟解法一没差了啊


威   2019-01-31 05:04     回复
灿哥这种解法开辟额外空间了啊，感觉面试官会不会不满意，用指针怎么解呀


yxc   2019-01-31 08:33     回复
刚刚在上面添加了算法2——双指针算法，可以参考一下hh
。

算法
(递归) O(n)O(n)
最直接的想法就是用递归，sum(n) = n+sum(n-1)，但是要注意终止条件，由于求的是1+2+…+n的和，所以需要在n=0的时候跳出递归，但是题目要求不能使用if,while等分支判断，可以考虑利用&&短路运算来终止判断。

时间复杂度分析：递归，复杂度为O(n)O(n)。

C++ 代码
```
class Solution {
public:
    int getSum(int n) {
        int res = n;
        (n>0) && (res += getSum(n-1));//利用短路运算终止递归
        return res;
    }
};
```

。

题目描述
求1+2+…+n,要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

样例
输入

10
输出

55
算法1
二逼做法，这已经是个梗了

时间复杂度
常数

C++ 代码
```
class Solution {
public:
    int getSum(int n) {
        char a[n][n+1];
        return sizeof(a)>>1;
    }
};
```

Code 1
class Solution {
public:
    long long getSum(long long n) {
        return (n * n + n) / 2;
    }
};
Solution
重新审视题面

要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

但是我的代码用了乘除法，会被 D 的，所以要换一种方法。

假设 sisi 为 11 加到 ii，那么我们可以得到一个递推公式

si=si−1+i
si=si−1+i
然后初始状态的 s1=1s1=1。

最后没有用到题面不让用的东西，直接递推即可。
```
class Solution {
public:
    long long getSum(long long n) {
        if (n == 1) return 1;
        return getSum(n - 1) + n;
    }
};
```


(链表) O(1)O(1)
由于是单链表，我们不能找到前驱节点，所以我们不能按常规方法将该节点删除。
我们可以换一种思路，将下一个节点的值复制到当前节点，然后将下一个节点删除即可。

时间复杂度
只有常数次操作，所以时间复杂度是 O(1)O(1)。

C++ 代码
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void deleteNode(ListNode* node) {

        auto p = node->next;

        node->val = p->val;
        node->next = p->next;
        // 这两步的作用就是将 *(node->next) 赋值给 *node，所以可以合并成一条语句：
        // *node = *(node->next);

        delete p;
    }
};

```


(二路归并) O(n)O(n)
新建头部的保护结点dummy，设置cur指针指向dummy。
若当前l1指针指向的结点的值val比l2指针指向的结点的值val小，则令cur的next指针指向l1，且l1后移；否则指向l2，且l2后移。
然后cur指针按照上一部设置好的位置后移。
循环以上步骤直到l1或l2为空。
将剩余的l1或l2接到cur指针后边。
时间复杂度
两个链表各遍历一次，所以时间复杂度为O(n)

C++ 代码
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode *dummy = new ListNode(0);
        ListNode *cur = dummy;
        while (l1 != NULL && l2 != NULL) {
            if (l1 -> val < l2 -> val) {
                cur -> next = l1;
                l1 = l1 -> next;
            }
            else {
                cur -> next = l2;
                l2 = l2 -> next;
            }
            cur = cur -> next;
        }
        cur -> next = (l1 != NULL ? l1 : l2);
        return dummy -> next;
    }
};
```

兄弟们，走过路过不要错过，精简版本瞅一瞅
精简递归

```
class Solution {
public:
    ListNode* merge(ListNode* l1, ListNode* l2) {
        if(!l1 || !l2) return l1 ? l1 : l2;
        if(l1->val > l2->val) swap(l1, l2);
        l1->next = merge(l1->next, l2);
        return l1;
    }
};
```

精简迭代

```
class Solution {
public:
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode *dummy = new ListNode(0);
        auto head = dummy;
        while(l1 && l2){
            if(l1->val > l2->val) swap(l1, l2);
            head->next = l1;
            l1 = l1->next;
            head = head->next;
        }
        head->next = l1 ? l1 : l2;
        return dummy->next;
    }
};
```
yxc   1个月前     回复


作者：yxc
链接：https://www.acwing.com/solution/content/744/

end of section 7

stlnb!!
```
#include<stdio.h>
#include<algorithm>
using namespace std;
int n,a[1000];
int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;++i){a[i]=i;}
    do{
        for(int i=1;i<=n;++i) printf("%d ",a[i]);
        puts("");
    }while(next_permutation(a+1,a+n+1));
    return 0;
}
```

算法1
(dfs) O(n!)O(n!)
大佬们都不会来写这些简单题目 我就分享下自己的学习记录
使用DFS 进入递归循环的时候 将使用过的数字记录置为零 当从该函数递归出来的时候进行还原。
DFS递归的终止条件是已经选中n个数字 然后打印该数字组合，退出。
DFS的基本用法 初次接触可能会有概念上的理解难度 可以尝试单步调试或者打印每次进入函数的状况来帮助理解

C++ 代码
```
#include <iostream>
#include <vector>

using namespace std;

vector<int> v;

int n ;

void dfs(int i, vector<int>& result){
    if(i== n){
        for(auto& e:result){
           cout << e << ' ';
       }
       cout <<endl;
       return;
    }

   for(int j = 0; j < v.size();j++){
       if(v[j] != 0){
           result[i] = v[j];
           v[j] = 0;
           dfs(i+1,result);
           v[j] =  result[i] ;
       }
   }
}

int main()
{
    cin >> n;
    for(int i= 1; i<=n;i++){
        v.push_back(i);
    }
    vector<int> result(n,0);

    dfs(0,result);

    return 0;
}
```

题目描述
给定一个 n×mn×m 的方格阵，沿着方格的边线走，从左上角 (0,0)(0,0) 开始，每次只能往右或者往下走一个单位距离，问走到右下角 (n,m)(n,m) 一共有多少种不同的走法。

输入格式
共一行，包含两个整数 nn 和 mm。

输出格式
共一行，包含一个整数，表示走法数量。

数据范围
1≤n,m≤101≤n,m≤10
输入样例：
2 3
输出样例：
10
样例
blablabla
先吐槽一下这题题目描述中没用 MarkDown

这里会给出本体的四种解法。

算法1
(暴搜) (2n+m)O(2n+m)
首先题目数据范围不大，可以使用爆搜。

每次搜索中

若搜到了点 (n,m)(n,m)，那么 res++res++ 并返回
否则如果不是最下面的点，那么搜该点下面的点
如果不是最右边的点，那么搜该点右边的点
CC 代码
```
#include <stdio.h>

int n, m;
int res;                  // res 存答案

void dfs(int x, int y)    // 爆搜函数
{
    if (x == n && y == m) // 如果搜到了点 (n, m), 那么 res ++ 并返回
    {
        res ++ ;
        return ;
    }
    if (x < n) dfs(x + 1, y); // 如果不是最下面的点，那么搜该点下面的点
    if (y < m) dfs(x, y + 1); // 如果不是最右边的点，那么搜该点右边的点
}

int main()
{
    scanf("%d %d", &n, &m);
    dfs(0, 0);            // 从点 (0, 0) 开始爆搜
    printf("%d\n", res);
    return 0;
}
```
算法2
(动态规划) (nm)O(nm)
f[i][j]f[i][j] 表示走到点 (i,j)(i,j) 的方案数，由于每次只能往下走或往右走，所以点 (i,j)(i,j) 只能从点 (i−1,j)(i−1,j) 或点 (i,j−1)(i,j−1) 上走过来
所以走到点 (i,j)(i,j) 的方案数是走到点 (i−1,j)(i−1,j) 的方案数与走到点 (i,j−1)(i,j−1) 的方案数之和，推出 f[i][j]=f[i−1][j]+f[i][j−1]f[i][j]=f[i−1][j]+f[i][j−1]
边界：f[i][0]=f[0][j]=1f[i][0]=f[0][j]=1
CC 代码
```
#include <stdio.h>

int n, m;
int f[11][11];

int main()
{
    scanf("%d %d", &n, &m);
    for (int i = 0; i <= n; i ++ )
        for (int j = 0; j <= m; j ++ )
            if (!i || !j) f[i][j] = 1; // 如果 i == 0 或 j == 0，那么 f[i][j] = 1
            else    f[i][j] = f[i - 1][j] + f[i][j - 1]; // 否则 f[i][j] = f[i - 1][j] + f[i][j - 1]
    printf("%d\n", f[n][m]);
    return 0;
}
```
算法3
（动态规划优化) (nm)O(nm)
用滚动数组优化一下上述dp，将空间复杂度优化至 O(m)O(m)
CC 代码
```
#include <stdio.h>

int n, m;
int f[11];

int main()
{
    scanf("%d %d", &n, &m);
    for (int i = 0; i <= m; i ++ )
        f[i] = 1;
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            f[j] += f[j - 1];
    printf("%d\n", f[m]);
    return 0;
}
```
算法4
(组合数) (n+m)O(n+m)
首先将dp的数组打印出来，找下规律。

    1     1     1     1     1     1     1     1     1
    1     2     3     4     5     6     7     8     9
    1     3     6    10    15    21    28    36    45
    1     4    10    20    35    56    84   120   165
    1     5    15    35    70   126   210   330   495
    1     6    21    56   126   252   462   792  1287
    1     7    28    84   210   462   924  1716  3003
    1     8    36   120   330   792  1716  3432  6435
    1     9    45   165   495  1287  3003  6435 12870
如果你从左上往右下斜着看，不难发现这就是一个旋转后的杨辉三角
其中，数组中的第 ii 行，第 jj 个数字是杨辉三角中的第 i+ji+j 行，第 jj 个数字。（坐标为从 第 00 行，第 00 列开始）
杨辉三角中的第 nn 行，第 mm 个数正好是 Cnm=n!m!(n−m)!Cmn=n!m!(n−m)!
所以我们只需要求下 Cnn+mCn+mn 就好啦~
当然，感性的理解下，你要走到点 (n,m)(n,m)，一共必然要走 n+mn+m 步，且必然有 nn 步是往下走的，就相当于是从 n+mn+m 步中，选出 nn 步往下走，答案为 Cnn+mCn+mn
所以我们可以通过求组合数的方式来快速求出答案。

CC 代码
```
#include <stdio.h>

int n, m;
long long res = 1;

int main()
{
    scanf("%d %d", &n, &m);
    int i = m + 1, j = 2;
    for (; i <= n + m; i ++ )
    {
        res *= i;
        while (j <= n && res % j == 0)
            res /= j, j ++ ; // 这里边乘边除是为了防止溢出，当然对于这题来说所有的数都乘完之后再除也是可以的
    }
    printf("%d\n", res);
    return 0;
}
```

作者：垫底抽风
链接：https://www.acwing.com/solution/content/15154/


AcWing 821. 跳台阶    原题链接    困难
作者：    itdef ,  2019-05-21 23:40:39 ,  阅读 832

4


2
题目描述
一个楼梯共有n级台阶，每次可以走一级或者两级，问从第0级台阶走到第n级台阶一共有多少种方案。

输入格式
共一行，包含一个整数n。

输出格式
共一行，包含一个整数，表示方案数。

数据范围
1≤n≤15

样例
输入样例：
5
输出样例：
8
算法1
(动态规划)
动态规划入门题

第0级台阶到第1级台 只有一种方法 上1级台阶
第0级台阶到第2级台 有两种方法 1种是0-2 上2级台阶 1种是上到1级台阶 再上2级台阶
第0级台阶到第3级台 有两种方法 1种是0-2 再2-3 1种是0-1 1-3 (其中0-1 1-2 2-3已经包含在前面的方法中了)

逆向来看就是 n台阶的方案数量 = n-1台阶方案数量 + n-2的方案数量

C++ 代码
```
#include <iostream>

using namespace std;

int arr[20];

int main()
{
    int n;
    cin >> n;
    arr[1] = 1; arr[2] = 2;
    for(int i = 3;i <=15;i++){
        arr[i] = arr[i-1]+arr[i-2];
    }
    cout << arr[n];

    return 0;

}
```

作者：itdef
链接：https://www.acwing.com/solution/content/2163/

```
题目描述
给定一个长度为 nn 的数组 aa 以及两个整数 ll 和 rr，请你编写一个函数，void sort(int a[], int l, int r)void sort(int a[], int l, int r)，将 a[l]a[l] ~ a[r]a[r] 从小到大排序。

输出排好序的数组 aa。

注意，这里的数组下标从 00 开始
样例输入
5 2 4
4 5 1 3 2
样例输出
4 5 1 2 3
哎，难度是困难？那当然是用高端解法来操作啦

算法 11
堆排序 O(nlogn)O(nlogn)
构建大根堆，每次将最大的元素放到最后

C++C++ 代码
#include <stdio.h>

int a[1005];

void swap(int i, int j)  // 技巧：手写交换，传入数组下标
{
    if (i ^ j)           // 特判 i = j 的情况，i ^ j 等价于 i != j
    {
        a[i] ^= a[j];    // 交换 a[i], a[j]
        a[j] ^= a[i];
        a[i] ^= a[j];
    }
}

void down(int l, int r, int p) // 将更小的元素
{
    int t = p;
    if ((p << 1) <= r - l && a[t] < a[(p << 1) - l])
        t = (p << 1) - l;
    if ((p << 1) + 1 - l <= r && a[t] < a[(p << 1) + 1 - l])
        t = (p << 1) + 1 - l;
    if (t != p)
    {
        swap(t, p);
        down(l, r, t);
    }
}

void heap_sort(int l, int r)
{
    for (int i = r - l >> 1; i; i -- ) // O(n)建堆
        down(l, r, i);
    while (r ^ l)        // 排序，同样用 r ^ l 代替 r != l
    {
        swap(1, r -- );  // 每次将最大的元素交换至最后，并在堆中删除
        down(l, r, 1);   // 将交换过来的元素向下交换，使剩余元素重构堆
    }
}

int main()
{
    int n, l, r;
    scanf("%d%d%d", &n, &l, &r);
    for (int i = 1; i <= n; i ++ )
        scanf("%d", &a[i]);

    heap_sort(l, r + 1);

    for (int i = 1; i <= n; i ++ )
        printf("%d ", a[i]);

    return 0;
}
算法 22
归并排序 O(nlogn)O(nlogn)
每次将数组划分成两个部分，分别处理

C++C++ 代码
#include <stdio.h>

const int N = 1005;

int a[N];
int t[N];

void merge_sort(int l,int r)
{
    if (l >= r) return;
    int mid = l + r >> 1;
    merge_sort(l, mid);
    merge_sort(mid + 1, r);
    int i = l, j = mid + 1, k = 0;
    while (i <= mid && j <= r)
        if (a[i] < a[j]) t[k ++ ] = a[i ++ ];
        else    t[k ++ ] = a[j ++ ];
    while (i <= mid) t[k ++ ] = a[i ++ ];
    while (j <= r) t[k ++ ] = a[j ++ ];
    for (int i = l, j = 0; i <= r; i ++, j ++ )
        a[i] = t[j];
}

int main()
{
    int n, l, r;
    scanf("%d%d%d", &n, &l, &r);
    for (int i = 0; i < n; i ++ )
        scanf("%d", &a[i]);

    merge_sort(l, r);

    for (int i = 0; i < n; i ++ )
        printf("%d ", a[i]);

    return 0;
}
// 懒得注释了
算法 33
快速排序 O(nlogn)O(nlogn)
每次将数组划分成两个部分，分别处理

C++C++ 代码
#include <stdio.h>

const int N = 1005;

int a[N];

void swap(int i, int j) // 由于当 i < j 的时候才会 swap，所以不用特判
{
    a[i] ^= a[j];
    a[j] ^= a[i];
    a[i] ^= a[j];
}

void quick_sort(int l,int r)
{
    if (l >= r) return;
    int x = a[l + r >> 1];
    int i = l - 1, j = r + 1;
    while (i < j)
    {
        while (a[ ++ i] < x);
        while (a[ -- j] > x);
        if (i < j) swap(i, j);
    }
    quick_sort(l, j);
    quick_sort(j + 1, r);
}

int main()
{
    int n, l, r;
    scanf("%d%d%d", &n, &l, &r);
    for (int i = 0; i < n; i ++ )
        scanf("%d", &a[i]);

    quick_sort(l, r);

    for (int i = 0; i < n; i ++ )
        printf("%d ", a[i]);

    return 0;
}
这就完了？

算法 44
三向切分快排 O(nlogn)O(nlogn)
用 ii，jj，kk 三个下标将数组切分成四部分。
a[l,i−1]a[l,i−1] 表示小于 xx 的部分，a[i,k−1]a[i,k−1]表示等于 xx 的部分，a[j+1]a[j+1] 表示大于 xx 的部分，而 a[k,j]a[k,j] 表示未判定的元素（即不知道比 xx 大，还是比中轴元素小）。
同时要注意 a[i]a[i] 始终位于等于 xx 部分的第一个元素，a[i]a[i] 的左边是小于 xx 的部分。

C++C++ 代码
#include <stdio.h>

const int N = 1005;

int a[N];

void swap(int i, int j)
{
    if (i ^ j)
    {
        a[i] ^= a[j];
        a[j] ^= a[i];
        a[i] ^= a[j];
    }
}

void quick_sort_3way(int l, int r)
{
    if(l >= r) return;
    int x = a[l];
    int i = l, j = r, k = l + 1;
    while(k <= j)
        if(a[k] < x)swap(i ++ , k ++ );
        else if(a[k] == x) k ++ ;
        else
        {
            while(a[j] > x)
                if( -- j < k)break;
            if (j < k) break;
            if(a[j] == x)
                swap(k ++ , j -- );
            else
            {
                swap(i ++ , j);
                swap(j -- , k ++ );
            }
        }
    quick_sort_3way(l, i - 1);
    quick_sort_3way(j + 1, r);
}

int main()
{
    int n, l, r;
    scanf("%d%d%d", &n, &l, &r);
    for (int i = 0; i < n; i ++ )
        scanf("%d", &a[i]);

    quick_sort_3way(l, r);

    for (int i = 0; i < n; i ++ )
        printf("%d ", a[i]);

    return 0;
}
算法 55
双轴快排 O(nlogn)O(nlogn)
同样，使用 ii，jj，kk 三个变量将数组分成四部分
同时，使用两个轴，通常选取最左边的元素作为 x1x1 和最右边的元素作 x2x2。
首先要比较这两个轴的大小，如果 x1>x2x1>x2，则交换最左边的元素和最右边的元素，以保证 x1<=x2x1<=x2。

神奇的是y总快排那题的数据把这两种优化过但不取中的快排都卡掉了。。。

C++C++ 代码
#include <stdio.h>

const int N = 1005;

int a[N];

void swap(int i, int j)
{
    if (i ^ j)
    {
        a[i] ^= a[j];
        a[j] ^= a[i];
        a[i] ^= a[j];
    }
}

void quick_sort_2(int l, int r)
{
    if(l >= r) return;
    if(a[l] > a[r]) swap(l, r);
    int x1 = a[l], x2 = a[r];
    int i = l, k = l + 1, j = r;
    while(k < j)
        if(a[k] < x1) swap( ++ i, k ++ );
        else if(a[k] >= x1 && a[k] <= x2) k ++ ;
        else
        {
            while(a[ -- j] > x2)
                if(j <= k) break;
            if (j <= k) break;
            if(a[j] >= x1 && a[j] <= x2)
                swap(k ++ , j);
            else
            {
                swap(j, k);
                swap( ++ i, k ++ );
            }
        }
    swap(l, i),swap(r, j);
    quick_sort_2(l, i - 1);
    quick_sort_2(i + 1, j - 1);
    quick_sort_2(j + 1, r);
}

int main()
{
    int n, l, r;
    scanf("%d%d%d", &n, &l, &r);
    for (int i = 0; i < n; i ++ )
        scanf("%d", &a[i]);

    quick_sort_2(l, r);

    for (int i = 0; i < n; i ++ )
        printf("%d ", a[i]);

    return 0;
}

作者：垫底抽风
链接：https://www.acwing.com/solution/content/9456/
```


利用set会自动去重的性质来做.
```
#include<iostream>
#include<set>
using namespace std;

int main()
{
    int n,size;
    int a[1010];
    cin>>n>>size;
    for(int i=0;i<n;i++) cin>>a[i];
    set<int> a1(a,&a[size]);
    cout<<a1.size()+n-size;
    return 0;
}

#include <bits/stdc++.h>

using namespace std;
const int N = 1010;
int b[N], a[N];
int n;
void reverse(int a[], int &size){

    for(int i = 0; i < size; i++){
        b[i] = a[size - i - 1];


    }
    for(int i = size; i < n; i++) b[i] = a[i];

}
int main()
{
    int size;
    cin>> n >> size;
    for(int i = 0; i< n; i++) cin>> a[i];
    //int a[n], b[n];
    reverse(a, size);

    for(int i = 0;i < n; i++) cout<<b[i] << " ";
    return 0;
}

作者：zhiling
链接：https://www.acwing.com/solution/content/2154/

#include <cstdio>

void print(char str[])
{
    printf("%s", str);
}

int main()
{
    char str[110];
    fgets(str, 101, stdin);

    print(str);

    return 0;
}

```

入门题目，根据题意写就好，注意是从 a 数组复制给 b 数组，所以，函数声明和调用函数的参数不要写错了哦~

C++ 代码
```
#include <iostream>

using namespace std;

void copy(int a[], int b[], int size)
{
    for (int i = 1; i <= size; i ++ )
        b[i] = a[i];
}

int main()
{
    int a[110], b[110], n, m, size;
    cin >> n >> m >> size;
    for (int i = 1; i <= n; i ++ )
        cin >> a[i];
    for (int i = 1; i <= m; i ++ )
        cin >> b[i];

    copy(a, b, size);

    for (int i = 1; i <= m; i ++ )
        cout << b[i] << ' ';

    return 0;
}
```
作者：liuser
链接：https://www.acwing.com/solution/content/10802/


```
#include <cstring>
#include <iostream>

using namespace std;

const int N = 110;

void copy(int a[], int b[], int size)
{
    memcpy(b, a, size * 4);
}

int main()
{
    int a[N], b[N];
    int n, m, size;
    cin >> n >> m >> size;
    for (int i = 0; i < n; i ++ ) cin >> a[i];
    for (int i = 0; i < m; i ++ ) cin >> b[i];

    copy(a, b, size);

    for (int i = 0; i < m; i ++ ) cout << b[i] << ' ';
    cout << endl;

    return 0;
}

#include <iostream>

using namespace std;

int lcm(int a, int b)
{
    for (int i = 1; i <= a * b; i ++ )
        if (i % a == 0 && i % b == 0)
            return i;
    return -1;  // 一定不会执行
}

int main()
{
    int a, b;
    cin >> a >> b;

    cout << lcm(a, b) << endl;

    return 0;
}

#include <iostream>

using namespace std;

int sum(int l, int r)
{
    int s = 0;
    for (int i = l; i <= r; i ++ ) s += i;
    return s;
}

int main()
{
    int l, r;
    cin >> l >> r;
    cout << sum(l, r) << endl;

    return 0;
}

#include <cstdio>

double add(double x, double y)
{
    return x + y;
}

int main()
{
    double x, y;
    scanf("%lf%lf", &x, &y);
    printf("%.2lf\n", add(x, y));

    return 0;
}

#include<bits/stdc++.h>
using namespace std;
double add(double x,double y){return x+y;}
int main(){
    double x,y; cin>>x>>y;
    cout<<fixed<<setprecision(2)<<add(x,y)<<endl;
    return 0;
}


#include <iostream>

using namespace std;

int abs(int x)
{
    if (x > 0) return x;
    return -x;
}

int main()
{
    int x;
    cin >> x;
    cout << abs(x) << endl;

    return 0;
}

#include <iostream>

using namespace std;

int f(int n)
{
    if (n <= 2) return 1;
    return f(n - 2) + f(n - 1);
}

int main()
{
    int n;
    cin >> n;

    cout << f(n) << endl;

    return 0;
}

//依然是一道简单的函数题。。。
#include<bits/stdc++.h>
using namespace std;
int fact(int n){//阶乘函数
    if(n==1) return 1;//递归终止条件
    else return fact(n-1)*n;//递归关系式
}
int main(){
    int n;cin>>n;
    cout<<fact(n);
    return 0;
}

#include <iostream>

using namespace std;

int fact(int n)
{
    if (n == 1) return 1;
    return n * fact(n - 1);
}

int main()
{
    int n;
    cin >> n;

    cout << fact(n) << endl;

    return 0;
}

#include<stdio.h>
int main(int n,int m,int k){
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++) scanf("%d",&k),printf("%d ",k);
        puts("");
    }
}


#include <iostream>

using namespace std;

const int N = 120;

int a[N][N];

void print(int a[][N], int c, int r)
{
    for (int i = 1; i <= c; i ++ )
    {
        for (int j = 1; j <= r; j ++ )
            cout << a[i][j] << ' ';

        cout << endl;
    }

    return ;
}

int main()
{
    int n, m;
    cin >> n >> m;


    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            cin >> a[i][j];

    print(a, n, m);

    return 0;
}




#include <iostream>

using namespace std;

void print2D(int a[][100], int row, int col)
{
    for (int i = 0; i < row; i ++ )
    {
        for (int j = 0; j < col; j ++ )
            cout << a[i][j] << ' ';
        cout << endl;
    }
}

int main()
{
    int a[100][100];

    int row, col;

    cin >> row >> col;
    for (int i = 0; i < row; i ++ )
        for (int j = 0; j < col; j ++ )
            cin >> a[i][j];

    print2D(a, row, col);

    return 0;
}

#include <bits/stdc++.h>

using namespace std;
const int N = 1010;
int a[N];
int size;
int n;
void print(int a[], int &size){
    for(int i = 0; i < size; i++) cout<< a[i] <<" ";

    cout<< endl;
}

int main()
{
    //int n;
    cin>> n >> size;
    //int a[n];
    for(int i = 0;i < n; i++) cin>>a[i];
    print(a, size);
    return 0;
}

#include <iostream>

using namespace std;

const int N = 1010;

void print(int a[], int size)
{
    for (int i = 0; i < size; i ++ )
        cout << a[i] << ' ';
    cout << endl;
}

int main()
{
    int n, size;
    int a[N];

    cin >> n >> size;
    for (int i = 0; i < n; i ++ ) cin >> a[i];

    print(a, size);

    return 0;
}

#include<iostream>
using namespace std;



int main()
{

    int x,y;
    cin>>x>>y;

    x=x^y;
    y=x^y;
    x=x^y;

    cout<<x<<" " <<y<<endl;

    return 0;

}

#include <iostream>

using namespace std;

void swap(int& x, int& y)
{
    if (x == y) return;

    int t = x;
    x = y;
    y = t;
}

int main()
{
    int x, y;
    cin >> x >> y;
    swap(x, y);

    cout << x << ' ' << y << endl;

    return 0;
}

#include<iostream>
using namespace std;
int gcd(int a,int b)
{
    if(a%b==0) return b;
    return gcd(b,a%b);
}
int main()
{
    int a,b;
    cin>>a>>b;
    cout<<gcd(a,b);
    return 0;
}

//最简单的函数题了。。。
#include<bits/stdc++.h>
using namespace std;
int maxn(int a,int b){if(a>=b) return a;else return b;}
int main(){
    int a,b; cin>>a>>b;
    cout<<maxn(a,b)<<endl;
    return 0;
}

#include <iostream>

using namespace std;

int max(int x, int y)
{
    if (x > y) return x;
    return y;
}

int main()
{
    int x, y;
    cin >> x >> y;

    cout << max(x, y) << endl;

    return 0;
}


```



题目描述
输入一个整数n，请你编写一个函数，int fact(int n)，计算并输出n的阶乘。

输入格式
共一行，包含一个整数n。

输出格式
共一行，包含一个整数表示n的阶乘的值。

数据范围
1≤n≤10

样例
输入样例：
3
输出样例：
6
算法1
C++ 代码
```
#include<iostream>

using namespace std;

int f(int n)
{
    int r = 1;
    for(int i = 1; i <= n; i ++) r *= i;
    return r;
}

int main()
{
    int n;
    cin >> n;
    cout << f(n) << endl;
    return 0;
}
```
副：由Anjor dalao提供的递归做法
```
#include<iostream>
using namespace std;

int fact(int n){
    if(n<=1)return n;
    return n*fact(n-1);
}

int main(){
    int n;
    cin>>n;
    cout<<fact(n);


    return 0;
}
```
垫底抽风提供主函数递归做法
```
#include <stdio.h>
int main(int n,bool f=true)
{
    if(f)
    {
        scanf("%d",&n);
        printf("%d",main(n,false));
        return 0;
    }
    if(n)return main(n-1,false)*n;
    return 1;
}
```
作者：cht
链接：https://www.acwing.com/solution/content/12856/
。






```
#include <iostream>

using namespace std;

const int N = 200;

int n;
string str[N];

int main()
{
    while (cin >> n, n)
    {
        int len = 1000;
        for (int i = 0; i < n; i ++ )
        {
            cin >> str[i];
            if (len > str[i].size()) len = str[i].size();
        }

        while (len)
        {
            bool success = true;
            for (int i = 1; i < n; i ++ )
            {
                bool is_same = true;
                for (int j = 1; j <= len; j ++ )
                    if (str[0][str[0].size() - j] != str[i][str[i].size() - j])
                    {
                        is_same = false;
                        break;
                    }
                if (!is_same)
                {
                    success = false;
                    break;
                }
            }

            if (success) break;
            len -- ;
        }

        cout << str[0].substr(str[0].size() - len) << endl;
    }

    return 0;
}
```
section 5 on string backwards on problems;
section 6 on function done
section 7 on struct, class, pointer done
section 8 on STL done

兔子吃窝边草，一层一层吃 bfs

树是无环联通图 -
无向图是特殊的有向图 - 双向边，所以只需考虑有向图

这道题我看了一下别人的题解，感觉都没我的容易理解。。哈哈哈。
先不论时间复杂度谁的更低。
我的大致思路如下：因为这道题是让求几次方，其实我们可以当成周期串来处理，输入一个字符串，找到他的最小周期，然后让字符串长度除以它的最小周期就是它的几次方了。
代码如下：

```
#include<bits/stdc++.h>
using namespace std;
int main()
{
    string s;
    while(cin>>s)
    {
        int flag;
        if(s[0]=='.')   break;
        int len=s.size();
        for(int i=1;i<=len;i++)
        {
            flag=0;
            if(len%i!=0)    continue;
            for(int j=0;j<len;j++)
                if(s[j]!=s[j%i])    {flag=1;break;}
            if(!flag)   {cout<<len/i<<endl;break;}
        }
        if(flag)    cout<<"-1"<<endl;
    }
}

#include <iostream>

using namespace std;

int main()
{
    string str;

    while (cin >> str, str != ".")
    {
        int len = str.size();

        for (int n = len; n; n -- )
            if (len % n == 0)
            {
                int m = len / n;
                string s = str.substr(0, m);
                string r;
                for (int i = 0; i < n; i ++ ) r += s;

                if (r == str)
                {
                    cout << n << endl;
                    break;
                }
            }
    }

    return 0;
}


```
作者：给个选择
链接：https://www.acwing.com/solution/content/7726/


一次性将该字符串连接上去，复制字符串即可

C++ 代码
```
#include <iostream>
#include <string>

using namespace std;

bool check(string a, string b)
{
    int len = a.size();
    a += a; //复制字符串并连接
    if (a.find(b) >= 0 && a.find(b) < len) return true; //判断是否包含
    return false;
}

int main()
{
    string a, b;
    cin >> a >> b;
    if (check(a, b) || check(b, a)) cout << "true";
    else cout << "false";
    return 0;
}
```
作者：Meet.
链接：https://www.acwing.com/solution/content/9160/

```
#include <bits/stdc++.h>
using namespace std;
int main()
{
    string s[10005];int i=1;
    while(cin>>s[i]) i++;
    for(int j=i-1;j>=1;j--) cout<<s[j]<<" ";
    return 0;
}

#include<cstring>
#include<iostream>
using namespace std;
string l,lm;
int len,maxx,f;
int main(){
    while(cin>>l){
        len=l.size();
        if(l[len-1]=='.'){
            len--;
            l.erase(len,1);
            f=1;
        }
        if(len>maxx){
            lm=l;
            maxx=len;
        }
        if(f) break;
    }
    cout<<lm;
    return 0;
}


```

题目描述
求一个字符串中最长的连续出现的字符，输出该字符及其出现次数，字符串中无空白字符（空格、回车和tab），如果这样的字符不止一个，则输出第一个。

输入格式
第一行输入整数N，表示测试数据的组数。

每组数据占一行，包含一个不含空白字符的字符串，字符串长度不超过200。

输出格式
共一行，输出最长的连续出现的字符及其出现次数，中间用空格隔开。

输入样例：
2
aaaaabbbbbcccccccdddddddddd
abcdefghigk
输出样例：
d 10
a 1
算法
(双指针) O(n×T)O(n×T)
使用双指针扫描每一个test case，并记录下最大长度与该长度下的字符即可。

时间复杂度
每个test case的字符串会被扫描一次，总共T个test case，所以总复杂度为O(n×T)O(n×T)。

C++ 代码
```
#include <iostream>
using namespace std;
int main()
{
    int T;
    cin >> T;
    while(T --)
    {
        int maxn = -1;//maxn记录最大长度
        string str, maxs;//maxs记录最大长度时的字符
        cin >> str;
        for(int i = 0; i < str.size(); i ++)
        {
            int j = i;
            int cnt = 0;
            while(str[j] == str[i] && j < str.size())//当指针j没有越界且与指针i的内容相同时移动
                j ++, cnt ++;
            if(cnt > maxn)//更新最大值
                maxn = cnt, maxs = str[i];
            i = j - 1;//移动指针i
        }
        cout << maxs << " " << maxn << endl;
    }
}
```
作者：P云
链接：https://www.acwing.com/solution/content/8698/


题目描述
输入一个字符串，以回车结束（字符串长度不超过100）。
该字符串由若干个单词组成，单词之间用一个空格隔开，所有单词区分大小写。
现需要将其中的某个单词替换成另一个单词，并输出替换之后的字符串。
输入格式
输入共3行。
第1行是包含多个单词的字符串 s;
第2行是待替换的单词a(长度不超过100);
第3行是a将被替换的单词b(长度不超过100)。
输出格式
共一行，输出将s中所有单词a替换成b之后的字符串。

样例
输入样例：
You want someone to help you
You
I
输出样例：
I want someone to help you
分析
直接用正则表达式替换单词即可

C++ 代码
```
#include <iostream>
#include <string>
#include <regex>
using namespace std;
int main(){
    string s,s1,s2;
    getline(cin,s);
    cin>>s1>>s2;
    cout<<regex_replace(s,regex("\\b" + s1 + "\\b"),s2)<<endl;
    return 0;
}

#include <iostream>
#include <sstream>

using namespace std;

int main()
{
    string s, a, b;

    getline(cin, s);
    cin >> a >> b;

    stringstream ssin(s);
    string str;
    while (ssin >> str)
        if (str == a) cout << b << ' ';
        else cout << str << ' ';

    return 0;
}

```
作者：昂昂累世士
链接：https://www.acwing.com/solution/content/2601/


题目描述
给定一个字符串a，请你按照下面的要求输出字符串b。

给定字符串a的第一个字符的ASCII值加第二个字符的ASCII值，得到b的第一个字符；

给定字符串a的第二个字符的ASCII值加第三个字符的ASCII值，得到b的第二个字符；

…

给定字符串a的倒数第二个字符的ASCII值加最后一个字符的ASCII值，得到b的倒数第二个字符；

给定字符串a的最后一个字符的ASCII值加第一个字符的ASCII值，得到b的最后一个字符。

输入格式
输入共一行，包含字符串a。注意字符串中可能包含空格。

数据保证字符串内的字符的ASCII值均不超过63。

输出格式
输出共一行，包含字符串b。

数据范围
2≤a的长度≤100

样例
输入样例：
1 2 3
输出样例：
QRRSd
算法1
C++ 代码
```
#include<iostream>
using namespace std;
int main()
{
    string a;
    getline(cin,a);
    string b;
    for(int i = 0; i + 1 < a.size(); i ++) b += (char)(a[i] + a[i + 1]);
    b += (char)(a[0] + a.back());
    cout << b;
}

#include <iostream>

using namespace std;

int main()
{
    string a, b;
    getline(cin, a);

    for (int i = 0; i < a.size(); i ++ ) b += a[i] + a[(i + 1) % a.size()];

    cout << b << endl;

    return 0;
}

```
作者：cht
链接：https://www.acwing.com/solution/content/12876/


方法一：

公式法：
```
#include <bits/stdc++.h>
using namespace std;
int main()
{
    string s;
    getline(cin, s);
    for (auto &c : s)
        if (c >= 'a' && c <= 'z') c = (c - 'a' + 1) % 26 + 'a';
        else if (c >= 'A' && c <= 'Z') c = (c - 'A' + 1) % 26 + 'A';
    cout << s << endl;
    return 0;
}
方法二：

自动算空格法：
注意:P用scanf不会过滤空格哦！

#include<bits/stdc++.h>
using namespace std;
int main(){
    char ch;
    while(scanf("%c",&ch)==1){
        if(ch>='a'&&ch<='z')
            ch=(ch+1-'a')%26+'a';
        else if(ch>='A'&&ch<='Z')
            ch=(ch+1-'A')%26+'A';
    cout<<ch;
    }
    return 0;
}

方法三：

不动脑子法
#include<bits/stdc++.h>
using namespace std;

int main(){
    string a;
    getline(cin,a);
    for(int i=0;i<a.size();i++){
        if(a[i]>='a'&&a[i]<'z'||a[i]>='A'&&a[i]<'Z')
        a[i]++;
        else if(a[i]=='z'||a[i]=='Z')a[i]-=25;
    }
    cout<<a<<endl;

    return 0;
}


作者：wuzgnm
链接：https://www.acwing.com/solution/content/9203/

#include <iostream>

using namespace std;

int main()
{
    string s;

    getline(cin, s);

    for (auto &c : s)
        if (c >= 'a' && c <= 'z') c = (c - 'a' + 1) % 26 + 'a';
        else if (c >= 'A' && c <= 'Z') c = (c - 'A' + 1) % 26 + 'A';

    cout << s << endl;

    return 0;
}

```


方法一：利用cin在输入时不会读入空格,tab,回车。
```
#include<bits/stdc++.h>
using namespace std;
int main()
{
    string s;
    while(cin>>s)
    {
        cout<<s<<" ";
    }
    return 0;
}
```
方法二：利用标识符flag来标记上一个输出字符是否为空格，如果是,那么本次字符如果还是空格的话不输出。
```
#include<bits/stdc++.h>
using namespace std;
int main()
{
    string s;
    getline(cin,s);
    int flag=0;
    for(int i=0;i<s.size();i++)
    {
        if(s[i]!=' ')
            { cout<<s[i];
             flag=0;}
        else if(flag==0)
            {
                cout<<" ";
                flag=1;
            }
        else
            continue;
    }
    return 0;
}

cin做法
#include <iostream>

using namespace std;

int main()
{
    string s;
    while (cin >> s) cout << s << ' ' ;

    return 0;
}
第一类双指针算法
#include <iostream>

using namespace std;

int main()
{
    string s;
    getline(cin, s);

    string r;
    for (int i = 0; i < s.size(); i ++ )
        if (s[i] != ' ') r += s[i];
        else
        {
            r += ' ';
            int j = i;
            while (j < s.size() && s[j] == ' ') j ++ ;
            i = j - 1;
        }

    cout << r << endl;

    return 0;
}
局部性判断方法
#include <iostream>

using namespace std;

int main()
{
    string s;
    getline(cin, s);

    string r;
    for (int i = 0; i < s.size(); i ++ )
        if (s[i] != ' ') r += s[i];
        else
        {
            if (!i || s[i - 1] != ' ') r += ' ';
        }

    cout << r << endl;

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/255236/

```

作者：冰语晨星
链接：https://www.acwing.com/solution/content/10733/

(暴力枚举)

```
#include <iostream>
using namespace std;

int main()
{
    string a, b;
    getline(cin, a);
    getline(cin, b);
    int t = 0, m = a.size()>b.size() ? a.size():b.size();
    for(int i = 0; i < m; i++)
    {
        if(a[i] >= 'a' && a[i] <= 'z') a[i] -= 32;
        if(b[i] >= 'a' && b[i] <= 'z') b[i] -= 32;
        if(a[i] == b[i])
            t++;
        else if(a[i] > b[i])
        {
            cout << ">" << endl;
            break;
        }
        else if(a[i] < b[i])
        {
            cout << "<" << endl;
            break;
        }
    }
    if(t == m) cout << "=" << endl;
    return 0;
}

#include <cstdio>
#include <cstring>

int main()
{
    char a[100], b[100];

    fgets(a, 100, stdin);
    fgets(b, 100, stdin);

    if (a[strlen(a) - 1] == '\n') a[strlen(a) - 1] = 0;  // 去掉末尾回车
    if (b[strlen(b) - 1] == '\n') b[strlen(b) - 1] = 0;  // 去掉末尾回车

    for (int i = 0; a[i]; i ++ )
        if (a[i] >= 'A' && a[i] <= 'Z')
            a[i] += 32;

    for (int i = 0; b[i]; i ++ )
        if (b[i] >= 'A' && b[i] <= 'Z')
            b[i] += 32;

    int t = strcmp(a, b);
    if (t == 0) puts("=");
    else if (t < 0) puts("<");
    else puts(">");

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/247391/

```
作者：有毒的Time
链接：https://www.acwing.com/solution/content/9327/

超简单字符串

```
#include <bits/stdc++.h>
using namespace std;
int main()
{
    double cnt = 0;
    double k;
    string a,b;
    cin>>k>>a>>b;
    double len = a.size();
    for(int i = 0;i<len;i++)
    {
        if(a[i] == b[i])
            cnt++;
    }
    cnt = cnt/len;
    if(k<=cnt)
    {
        cout<<"yes";
        return 0;
    }
    else cout<<"no";
    return 0;
}
#include <iostream>

using namespace std;

int main()
{
    double k;
    string a, b;
    cin >> k >> a >> b;

    int cnt = 0;
    for (int i = 0; i < a.size(); i ++ )
        if (a[i] == b[i])
            cnt ++ ;

    if ((double)cnt / a.size() >= k) puts("yes");
    else puts("no");

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/247373/

```
作者：sxc
链接：https://www.acwing.com/solution/content/5104/


题目描述
给你一个只包含小写字母的字符串。

请你判断是否存在只在字符串中出现过一次的字符。

如果存在，则输出满足条件的字符中位置最靠前的那个。

如果没有，输出”no”。
输入格式
共一行，包含一个由小写字母构成的字符串。

数据保证字符串的长度不超过100000。

输出格式
输出满足条件的第一个字符。

如果没有，则输出”no”。

样例
输入样例：
abceabcd
输出样例：
e
思路
这道题的关键在于如何把字母出现的次数记下来，这里用到一个数组s[300]，来记录各个字母出现的次数；
然后调用
for(int i=0;i<str.length();i)
{
s[str[i];
}，用数组s存储各个字母出现的次数。
再次调用for循环遍历各个字符，
if(s[str[i]]==1)
{
p=str[i];
break;
}如果遇到第一个字符出现次数为1.则把该字符赋值给c,break退出for循环。

C++ 代码
```
#include<iostream>
#include<cstdio>
#include<cstring>
using namespace std;
int main()
{
    string str;
    getline(cin,str);
    int s[300]={0};
    for(int i=0;i<str.length();i++)
    {
        s[str[i]]++;//统计各个字符出现的次数
    }
    char p=-1;
    for(int i=0;i<str.length();i++)//如果遇到第一个字符出现次数为1.则把该字符赋值给c,break退出for循环。
    {
        if(s[str[i]]==1)
        {
            p=str[i];
            break;
        }
    }
    if(p==-1) puts("no");
    else  printf("%c",p);
    return 0;
}

#include <iostream>
#include <cstring>

using namespace std;

int cnt[26];
char str[100010];

int main()
{
    cin >> str;

    for (int i = 0; str[i]; i ++ ) cnt[str[i] - 'a'] ++ ;

    for (int i = 0; str[i]; i ++ )
        if (cnt[str[i] - 'a'] == 1)
        {
            cout << str[i] << endl;
            return 0;
        }

    puts("no");

    return 0;
}

作者：yxc

```
作者：白云苍狗
链接：https://www.acwing.com/solution/content/7348/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

题目描述
合并两个字符串，把第二个字符串放到第一个字符串中最大ASSIC码的字符的后面

算法1
(直接模拟) O(n)O(n)
先找到第一个字符串中最大ASSIC码字符的位置，标记位置，然后输出第一个字符串前面的字符，输出第二个字符

接着输出后面的剩下的字符

注意题目有多组输入，只需要第一个最大的字符
时间复杂度分析：

C++ 代码
```
#include <bits/stdc++.h>

using namespace std;

int main()
{
    char str[11], substr[4];

    int n = 2;
    while(scanf("%s %s", str, substr) != EOF){
        int cnt = str[0], res = 0;
        for(int i = 0; i < strlen(str); i++){
            if(str[i] > cnt) cnt = str[i], res = i;
            else if(str[i] == cnt) continue;
        }
        for(int i = 0; i <= res; i ++){
            cout<< str[i];
        }
        for(int i = 0; i < strlen(substr); i++){
            cout<< substr[i];
        }
        for(int i = res + 1; i < strlen(str); i++){
            cout<< str[i];
        }
        puts("");
    }
    return 0;
}

#include <iostream>

using namespace std;

int main()
{
    string a, b;

    while (cin >> a >> b)
    {
        int p = 0;
        for (int i = 1; i < a.size(); i ++ )
            if (a[i] > a[p])
                p = i;

        cout << a.substr(0, p + 1) + b + a.substr(p + 1) << endl;
    }

    return 0;
}
```
作者：yxc
链接：https://www.acwing.com/activity/content/code/content/247367/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

作者：zhiling
链接：https://www.acwing.com/solution/content/2913/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```
#include <iostream>
#include <cstring>
using namespace std;

int main()
{
    string a;
    char s;
    getline(cin,a);
    cin>>s;
    for(int i = 0;i < a.size();i ++)
    {
        if(a[i] == s) cout << "#";
        else cout<<a[i];
    }
    return 0;
}



#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    char str[31];
    scanf("%s", str);

    char c;
    scanf("\n%c", &c);

    for (int i = 0; str[i]; i ++ )
        if (str[i] == c)
            str[i] = '#';

    puts(str);

    return 0;
}
给定一个字符串，在字符串的每个字符之间都加一个空格。

输出修改后的新字符串。

输入格式
共一行，包含一个字符串。注意字符串中可能包含空格。

输出格式
输出增加空格后的字符串。

数据范围
1≤字符串长度≤100

样例
输入样例：
test case
输出样例：
t e s t   c a s e
算法1
C++ 代码
#include<iostream>
using namespace std;
int main()
{
    string a;
    getline(cin, a);
    for(char &c : a) cout << c << ' '; 
}


#include <iostream>

using namespace std;

int main()
{
    string a;
    getline(cin, a);

    string b;
    for (auto c : a) b = b + c + ' ';

    b.pop_back();  // 把最后一个字符删掉

    cout << b << endl;

    return 0;
}

用string的size之差表示player1赢的三种情况
#include <iostream>

using namespace std;

int main()
{
    int n;
    cin >> n;

    string x, y;
    while (n --)
    {
        cin >> x >> y;
        int a = x.size(), b = y.size();
        if (a - b == -1 || a - b == -2 || a - b == 3)
            cout << "Player1" << endl;
        else if (a == b)
            cout << "Tie" << endl;
        else cout << "Player2" << endl;
    }

    return 0;
}

作者：黄
链接：https://www.acwing.com/solution/content/15253/
#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    int n;
    cin >> n;

    while (n -- )
    {
        string a, b;
        cin >> a >> b;

        int x, y;
        if (a == "Hunter") x = 0;
        else if (a == "Bear") x = 1;
        else x = 2;

        if (b == "Hunter") y = 0;
        else if (b == "Bear") y = 1;
        else y = 2;

        if (x == y) puts("Tie");
        else if (x == (y + 1) % 3) puts("Player1");
        else puts("Player2");
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/247346/

输入一行字符，长度不超过100，请你统计一下其中的数字字符的个数。

输入格式
输入一行字符。注意其中可能包含空格。

输出格式
输出一个整数，表示字数字字符的个数。

样例
输入样例：
I am 18 years old this year.
输出样例：
2
算法1
C++ 代码
#include<iostream>

using namespace std;

int main()
{
    string a;
    getline(cin,a);
    int ans = 0;
    for(int i = 0; i < a.size(); i ++)
    {
        if(a[i] <= '9' && a[i] >= '0')
        {
            ans ++;
        }
    }

    cout << ans << endl;

    return 0;
}

作者：cht
链接：https://www.acwing.com/solution/content/12875/

#include <cstdio>

int main()
{
    char str[101];

    fgets(str, 101, stdin);

    int cnt = 0;
    for (int i = 0; str[i]; i ++ )
        if (str[i] >= '0' && str[i] <= '9')
            cnt ++ ;

    printf("%d\n", cnt);

    return 0;
}

题目描述
给定一行长度不超过100的字符串，请你求出它的具体长度。

输入格式
输入一行，表示一个字符串。注意字符串中可能包含空格。

输出格式
输出一个整数，表示它的长度。

样例
输入样例：
I love Beijing.
输出样例：
15
算法1
C语言代码
#include<string.h>
#include<stdio.h>

int main()
{
    char s[105];
    gets(s);
    printf("%d",strlen(s));
    return 0;
}
算法2
C++ 代码
#include <iostream>
#include <cstring>

using namespace std;

int main()
{
    string a;
    getline(cin,a);
    cout<<a.size()<<endl;
    return 0;
}
算法3
C++ 代码
#include <iostream>
#include <cstring>

using namespace std;

int main()
{
    char a[105];
    cin.get(a,105);//需要注意cin.get()不会把换行符取出删除，影响下一次读入！
    cout<<strlen(a)<<endl;
    return 0;
}
算法4
C++ 代码
#include <iostream>
#include <cstring>

using namespace std;

int main()
{
    char a[105];
    cin.getline(a,105);//需要注意cin.getline()会把换行符取出删除，不影响下一次读入！
    cout<<strlen(a)<<endl;
    return 0;
}
顺带一提 cin 和 scanf读入字符串时遇到空格就停止了。

作者：Accepting
链接：https://www.acwing.com/solution/content/10357/

注意：fgets函数会把回车也读进来
#include <cstdio>

int main()
{
    char str[101];

    fgets(str, 101, stdin);

    int len = 0;
    for (int i = 0; str[i] && str[i] != '\n'; i ++ ) len ++ ;

    printf("%d\n", len);

    return 0;
}


利用 left right top bottom 四个变量 来表示 这个矩形的边界

#include <iostream>

using namespace std;
const int N = 105;

int a[N][N];
int n, m;

int main() {
    cin >> n >> m;

    int left = 0, right = m - 1, top = 0, bottom = n - 1;
    int k = 1;
    while (left <= right && top <= bottom) {
        for (int i = left ; i <= right; i ++) {
            a[top][i] = k ++;
        }
        for (int i = top + 1; i <= bottom; i ++) {
            a[i][right] = k ++;
        }
        for (int i = right - 1; i >= left && top < bottom; i --) {
            a[bottom][i] = k ++;
        }
        for (int i = bottom - 1; i > top && left < right; i --) {
            a[i][left] = k ++;
        }
        left ++, right --, top ++, bottom --;
    }
    for (int i = 0; i < n; i ++) {
        for (int j = 0; j < m; j ++) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}

作者：Yuerer
链接：https://www.acwing.com/solution/content/8007/

因为在main里面初始化的话二维数组的初始值不确定，在main外面初始化初值全部为0，和堆与栈的储存方式有关，y总上课讲过。另外需要初始化初值全部为0，因为if (a < 0 || a >= n || b < 0 || b >= m || res[a][b])，这句判断中res[a][b]在没有被填入值的时候是0（否），填入值之后才是非0（真）。如果在main里用int res[n][m]，还要加上memset(res, 0, sizeof res); 这个语句

#include <iostream>

using namespace std;

int res[100][100];

int main()
{
    int n, m;
    cin >> n >> m;

    int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0};

    for (int x = 0, y = 0, d = 0, k = 1; k <= n * m; k ++ )
    {
        res[x][y] = k;
        int a = x + dx[d], b = y + dy[d];
        if (a < 0 || a >= n || b < 0 || b >= m || res[a][b])
        {
            d = (d + 1) % 4;
            a = x + dx[d], b = y + dy[d];
        }
        x = a, y = b;
    }

    for (int i = 0; i < n; i ++ )
    {
        for (int j = 0; j < m; j ++ ) cout << res[i][j] << ' ';
        cout << endl;
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245992/

#include <iostream>

using namespace std;

int main()
{
    int n;
    while(cin >> n,n)
    {
        for(int i = 0; i < n; i ++)
        {
            for(int j = 0; j < n; j ++)
                cout << (1 << i) * (1 << j) << ' ';//两个乘数 后者控制基数 1 ~ 2^(n-1) ，前者控制倍数
            cout << endl;
        }
        cout << endl;
    }
    return 0;
}

作者：XDDX
链接：https://www.acwing.com/solution/content/7673/

#include <iostream>
#include <cstdio>

using namespace std;

int main()
{
    int n;
    while (cin >> n, n)
    {
        for (int i = 0; i < n; i ++ )
        {
            for (int j = 0; j < n; j ++ )
            {
                int v = 1;
                for (int k = 0; k < i + j; k ++ ) v *= 2;
                cout << v << ' ';
            }
            cout << endl;
        }

        cout << endl;
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245977/

如题看看就好....
找了一个比较简单的规律....

#include <iostream>
#include <algorithm>

using namespace std;

int a[100][100];
int n;

int main()
{
    while (cin >> n)
    {
        for (int i = 0; i < n; i ++ )
            for(int j = 0; j < n; j ++ )
                a[i][j] = abs(i - j) + 1;  // 规律

        for (int i = 0; i < n; i ++ )
        {
            for (int j = 0; j < n; j ++ )
                cout << a[i][j] << ' ';

            cout << endl;
        }

        if (n)
        cout << endl;
    }

    return 0;
}

作者：ltk
链接：https://www.acwing.com/solution/content/6638/

#include <iostream>

using namespace std;

int q[100][100];

int main()
{
    int n;
    while (cin >> n, n)
    {
        for (int i = 0; i < n; i ++ )
        {
            q[i][i] = 1;
            for (int j = i + 1, k = 2; j < n; j ++, k ++ ) q[i][j] = k;
            for (int j = i + 1, k = 2; j < n; j ++, k ++ ) q[j][i] = k;
        }

        for (int i = 0; i < n; i ++ )
        {
            for (int j = 0; j < n; j ++ ) cout << q[i][j] << ' ';
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245972/

数组的右方区域：j>i&&i+j>11
#include <iostream>
using namespace std;
int main()
{
    char c;
    cin>>c;
    double a,res=0;
    for(int i=0;i<12;i++)
        for(int j=0;j<12;j++)
        {
            cin>>a;
            if(j>i&&i+j>11)res+=a;
        }
    printf("%.1lf",c=='S'?res:res/30);
}

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    char t;
    cin >> t;
    double q[12][12];

    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            cin >> q[i][j];

    double s = 0, c = 0;
    for (int i = 1; i <= 5; i ++ )
        for (int j = 12 - i; j <= 11; j ++ )
        {
            s += q[i][j];
            c += 1;
        }

    for (int i = 6; i <= 10; i ++ )
        for (int j = i + 1; j <= 11; j ++ )
        {
            s += q[i][j];
            c += 1;
        }

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / c);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245957/

。

数组的下方区域：i>j&&i+j>11
#include <iostream>
using namespace std;
int main()
{
    char c;
    cin>>c;
    double a,res=0;
    for(int i=0;i<12;i++)
        for(int j=0;j<12;j++)
        {
            cin>>a;
            if(i>j&&i+j>11)res+=a;
        }
    printf("%.1lf",c=='S'?res:res/30);
}

#include <iostream>
#include <cstdio>

using namespace std;

int main()
{
    char t;
    cin >> t;

    double q[12][12];
    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            cin >> q[i][j];

    double s = 0, c = 0;
    for (int i = 7; i <= 11; i ++ )
        for (int j = 12 - i; j <= i - 1; j ++ )
        {
            s += q[i][j];
            c += 1;
        }

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / c);

    return 0;

}

数组的左下半部分：j<i,总个数66
#include <iostream>

using namespace std;

int main()
{
    double s=0;
    char op;
    cin >> op;
    for(int i=0;i<12;i++)
    {
        for(int j=0;j<12;j++)
        {
            double x;
            cin >>x;
            if(j<i) s+=x;
        }
    }
    printf("%.1lf",op=='S' ? s : s/66);
}

作者：繁花似锦
链接：https://www.acwing.com/solution/content/12432/

#include <iostream>
#include <cstdio>

using namespace std;

int main()
{
    char t;
    cin >> t;

    double q[12][12];
    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            cin >> q[i][j];

    double s = 0, c = 0;
    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j <= i - 1; j ++ )
        {
            s += q[i][j];
            c += 1;
        }

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / c);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245941/

数组的右下半部分：i+j>=12
#include <iostream>

using namespace std;

int main()
{
    double s=0;
    char op;
    cin >> op;
    for(int i=0;i<12;i++)
    {
        for(int j=0;j<12;j++)
        {
            double x;
            cin >>x;
            if(i+j>=12) s+=x;
        }
    }
    printf("%.1lf",op=='S' ? s : s/66);
}

作者：繁花似锦
链接：https://www.acwing.com/solution/content/12431/

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    char t;
    cin >> t;
    double q[12][12];

    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            cin >> q[i][j];

    double s = 0, c = 0;
    for (int i = 1; i <= 11; i ++ )
        for (int j = 12 - i; j <= 11; j ++ )
        {
            s += q[i][j];
            c += 1;
        }

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / c);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245933/

#include<iostream>
#include<cstdio>
using namespace std;
double a[15][15];
int main(){
    int c;
    cin>>c;
    char flag;
    cin>>flag;
    for(int i=1;i<=12;i++){
        for(int j=1;j<=12;j++){
            cin>>a[i][j];
        }
    }
    double sum = 0.0;
    for(int i=1;i<=12;i++) sum += a[i][c + 1];
    if(flag == 'S') printf("%.1lf",sum);
    else printf("%.1lf",sum/12);
}

作者：HalfSummer
链接：https://www.acwing.com/solution/content/12689/

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    int c;
    char t;
    double q[12][12];

    cin >> c >> t;
    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            cin >> q[i][j];

    double s = 0;
    for (int i = 0; i < 12; i ++ ) s += q[i][c];

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / 12);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245928/

#include <bits/stdc++.h>

using namespace std;

int main()
{
    int n, res = 0, re = 0;
    cin>> n;
   // cout<< n <<endl;
   int num[n];
   cin>> num[0];
    res = num[0];
    for(int i = 1; i < n; i++){
        cin>> num[i];

       // cout<< res << endl;
       // cout<< num[i]  << endl;
        if(res > num[i]){
            //int temp = res;
            res = num[i];
            //num[i] = temp;
            re = i;
             //cout<< num[i] << endl;
             //cout<< res << endl;
        }

    }
   cout<<"Minimum value: "<< res << endl;
   cout<< "Position: "<< re << endl;
    return 0;
}

作者：zhiling
链接：https://www.acwing.com/solution/content/1894/


minimum number and its position by yxc:

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    int a[1001];
    int n;

    cin >> n;
    for (int i = 0; i < n; i ++ ) cin >> a[i];

    int p = 0;
    for (int i = 1; i < n; i ++ )
        if (a[i] < a[p])
            p = i;

    printf("Minimum value: %d\n", a[p]);
    printf("Position: %d\n", p);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245921/


#include <bits/stdc++.h>

using namespace std;

long long febo(int n)
{
    long long  a = 0;
    long long b = 1;
    long long c;
    if(n == 0) c = a;
    if(n == 1) c = b;

    while (n >= 2)
    {
        c = a + b;//从第三个数开始，斐波那契数等于前两个数的和；
        a = b;//将前一个数给到a，开始下一次求值
        b = c;//将斐波那契数给b，开始下一次求值
        n--;//每求一次，n都要减一
    }
    return c;
}

int main()
{
    int t, num = 0;
    long long res = 0;
    cin>> t;
    while(t--){
        cin>> num;
        res = febo(num);
        printf("Fib(%d) = %lld\n", num, res);
    }
    return 0;
}

作者：zhiling
链接：https://www.acwing.com/solution/content/1892/

fibonacci number: yxc

注意：小心整数溢出。

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    long long f[61];
    f[0] = 0, f[1] = 1;

    for (int i = 2; i <= 60; i ++ ) f[i] = f[i - 1] + f[i - 2];

    int n;
    cin >> n;
    while (n -- )
    {
        int x;
        cin >> x;
        printf("Fib(%d) = %lld\n", x, f[x]);
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245910/

//不是很难
#include <iostream>
using namespace std;
int main()
{
    int a[20];
    for(int i=0;i<20;i++)
    {
        cin >> a[i];
    }
    for(int i=19;i>=0;i--)
    {
        printf("N[%d] = %d\n", abs(i-19), a[i]);
    }
    return 0;
}

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    int a[20], b[20];

    for (int i = 0; i < 20; i ++ ) cin >> a[i];
    for (int i = 19, j = 0; i >= 0; i --, j ++ ) b[j] = a[i];

    for (int i = 0; i < 20; i ++ ) printf("N[%d] = %d\n", i, b[i]);

    return 0;
}

#include<bits/stdc++.h>

using namespace std;

int main()
{

    int n;
    int x; //当前数值
    while(cin>>n && n!=0)
    {
        for(int i=1;i<=n;i++)
        {
            for(int j=1;j<=n;j++)
            {
                x=min(min(i,j),min(n-i+1,n-j+1));//判断当前数值   内层第一个min是判断正方形左上部分，第二个min判断正方形右下部分
                cout<<x<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }

    return 0;
}

作者：Honey
链接：https://www.acwing.com/solution/content/6806/

#include <iostream>

using namespace std;

int main()
{
    int n;
    while (cin >> n, n)
    {
        for (int i = 1; i <= n; i ++ )
        {
            for (int j = 1; j <= n; j ++ )
            {
                int up = i, down = n - i + 1, left = j, right = n - j + 1;
                cout << min(min(up, down), min(left, right)) << ' ';
            }
            cout << endl;
        }

        cout << endl;
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245898/

数组的左方区域:(i+j)<=10&&i>j
#include <iostream>
using namespace std;
int main()
{
    char a;
    cin>>a;

    double s=0;
    for(int i=0;i<12;i++)
    {
        for(int j=0;j<12;j++)
        {
            double a;
            cin>>a;
            if((i+j)<=10&&i>j) // 规律
            s+=a;
        }
    }
    printf("%.1lf", a=='S' ? s : s/30);

}

作者：繁花似锦
链接：https://www.acwing.com/solution/content/12426/

#include <cstdio>

int main()
{
    char t;
    scanf("%c", &t);

    double q[12][12];
    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            scanf("%lf", &q[i][j]);

    double s = 0, c = 0;
    for (int i = 1; i <= 5; i ++ )
        for (int j = 0; j <= i - 1; j ++ )
        {
            s += q[i][j];
            c += 1;
        }

    for (int i = 6; i <= 10; i ++ )
        for (int j = 0; j <= 10 - i; j ++ )
        {
            s += q[i][j];
            c += 1;
        }

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / c);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/245893/


数组的上方区域：j>i&&i+j<11
#include <iostream>
using namespace std;
int main()
{
    char c;
    cin>>c;
    double a,res=0;
    for(int i=0;i<12;i++)
        for(int j=0;j<12;j++)
        {
            cin>>a;
            if(j>i&&i+j<11)res+=a;
        }
    printf("%.1lf",c=='S'?res:res/30);
}

左上半部分规律：i+j<11
#include <iostream>

using namespace std;

int main()
{
    double s=0;
    char op;
    cin >> op;
    for(int i=0;i<12;i++)
    {
        for(int j=0;j<12;j++)
        {
            double x;
            cin >>x;
            if(i+j<11) s+=x;
        }
    }
    printf("%.1lf",op=='S' ? s : s/66);
}

作者：繁花似锦
链接：https://www.acwing.com/solution/content/12455/


#include <bits/stdc++.h>

using namespace std;
string DoubleToStringByStdToString(double value)
 {

 const std::string& new_val = std::to_string(value);
return new_val;
}
int main()
{
    char str;
    cin>> str;
    int n = 12;
    double m[n][n];
    double res = 0.0;
     double re = 0.0;
     double count = 0;
    for(int i =0; i < n; i++){
        for(int j = 0; j < n; j++){
                   cin >> m[i][j];
        }
    }
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
                    //cout << m[i][j] << endl; 
                    //cout<< m[i][j] << endl;
                    re = re + m[i][j];
                    res = re;


        }
    }
    //cout << res << " " << count  << endl;
     if(str == 'S') res = res;   
     if(str == 'M') res = res / 66;
     //cout<< res << endl;

     //float ress = (float)(res);
    // cout<< ress << endl;

   // printf("%.1f\n", ress);
    //cout.unsetf(ios::fixed);
    // cout<< res << endl;
    //  res = res - (abs)(res - (int)(res));
    //   cout<<(abs)(res - (int)(res)) << endl;
     //cout<< res <<endl;


     //cout.setf(ios::fixed);
       // cout<< res << endl;

        printf("%.1lf", res);
    return 0;
}

作者：zhiling
链接：https://www.acwing.com/solution/content/1906/

yxc



#include <cstdio>

int main()
{
    char t;
    scanf("%c", &t);
    double a[12][12];

    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            scanf("%lf", &a[i][j]);

    int c = 0;
    double s = 0;

    for (int i = 0; i < 12; i ++ )
        for (int j = i + 1; j < 12; j ++ )
        {
            c ++ ;
            s += a[i][j];
        }

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / c);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/238587/

#include <iostream>

using namespace std;

int main()
{
    int l;
    char op;
    cin >> l >> op;
    double s=0;
    for(int i=0;i<12;i++)
    {
        for(int j=0;j<12;j++)
        {
            double a;
            cin >> a;
            if(i==l) s+=a;
        }
    }

    printf("%.1lf",op=='S' ? s : s/12);
}

作者：繁花似锦
链接：https://www.acwing.com/solution/content/12452/

#include <cstdio>

int main()
{
    double a[12][12];

    int l;
    char t;
    scanf("%d\n%c", &l, &t);

    for (int i = 0; i < 12; i ++ )
        for (int j = 0; j < 12; j ++ )
            scanf("%lf", &a[i][j]);

    double s = 0;
    for (int i = 0; i < 12; i ++ ) s += a[l][i];

    if (t == 'S') printf("%.1lf\n", s);
    else printf("%.1lf\n", s / 12);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/238575/

#include<iostream>
using namespace std;
int main(){
    double x[100]={0};
    for(int i=0;i<100;i++) 
    {
        cin>>x[i];
        if(x[i]<=10) printf("A[%d] = %.1f\n",i,x[i]);
    }
    return 0;
}

#include<iostream>
#include<cstdio>
using namespace std;
int main()
{
    int x,i,v;
    cin>>v;
    x=v;
    for(i=0;i<=9;i++)
    {
        printf("N[%d] = %d\n",i,x);
        x=x*2;
    }
    return 0;
}

#include <iostream>
#include <cstdio>
using namespace std;
int main(){
    for(int i = 0; i < 10; i ++ ){
        int x;  cin >> x;
        printf("X[%d] = %d\n", i, x <= 0 ? 1 : x);
    }
    return 0;
}

y总优雅做法
别着急做题！先去找图形特点！这其实是个正方形！
abs(sx - i) + abs(sy - j) <= n / 2
#include <iostream>
#include <algorithm>

using namespace std;

int main()
{
    int n;
    cin >> n;

    int sx = n / 2, sy = n / 2;

    for (int i = 0; i < n ; i ++ )
    {
        for (int j = 0; j < n; j ++ )
        {
            if ( abs(sx - i) + abs(sy - j) <= n / 2 ) cout << "*";
            else cout << " ";
        }
        cout << endl;    
    }

    return 0;
}

第三次做法(依旧是陆同学的想法) 可以把上下三角形合并，直接出菱形
int x = n / 2;
for (int i = -x; i <= x; i ++ )
    {
        for (int j = 0; j < abs(i); j ++ ) cout <<' ';
        for (int j = 0; j < n - abs(i) * 2; j ++ ) cout << '*';
        puts("");
    }
第二次做法(陆同学的想法) 去找行号 和 空格&星号 的关系,输出上下三角形
int x = n / 2;

for (int i = 0; i < x; i ++)
{
    for (int j = 0; j < x - i; j ++ ) cout <<' ';
    for (int j = 0; j < 2 * i + 1; j ++  ) cout << '*';
    puts("");
}

for (int i = 0; i < n - x; i ++ )
{
    for (int j = 0; j < i; j ++ ) cout << ' ';
    for (int j = 0; j < n - 2 * i; j ++ ) cout << '*';
    puts("");
}
个人第一次做法，想老半天，我简直蠢到家了
#include<iostream>
#include<cstdio>

using namespace std;

int main()
{
    int n;
    cin >> n;
    int x = n / 2;

    char s[n][n];

    // 输入空格
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < n; j ++ )
            s[i][j] = ' ' ;

    // 上半部分
    for (int i = 0; i < x; i ++ )
        for (int j = x - i; j <= x + i; j ++ )
            s[i][j] = '*';

    // 中间一行  
    for (int j = 0; j < n; j ++ )
        s[x][j] = '*';

    // 下半部分
    for (int i = x + 1; i < n; i ++ )
        for (int j = i - x; j < n - i + x; j ++ )
            s[i][j] = '*';

    // 输出
    for (int i = 0; i < n; i ++ )
    {
        for (int j = 0; j < n; j ++ )
            cout << s[i][j] ;
        cout << endl;
    }

    return 0;
}

作者：小张同学
链接：https://www.acwing.com/solution/content/8774/


#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
bool complare(int a,int b)
{
    return a>b;
}
int main()
{
   int n;
   cin>>n;
   int x, res = 0;
   while(n--){
       cin>> x;
       int k  = (int)(sqrt(x));
        bool prime = true;
       for(int i = 2; i <= k; i++){
           if(x % i == 0){
               prime = false;
           }
       }
       if(prime == true)
        cout<< x <<" is prime" << endl;
      if(prime == false)
          cout << x <<" is not prime" << endl;
   }
  return 0;
}

作者：zhiling
链接：https://www.acwing.com/solution/content/1880/


有点数学基础的人都应该知道100000000100000000内的完全数没有几个......

数学部分
100000000100000000内的完全数有6,28,496,8128,335503366,28,496,8128,33550336.所以说多背一点数字是很有用的

既然这道题可以直接O(1)O(1)解决,我们不妨来说一下完全数的各种性质以备于各种毒瘤的算法竞赛.

完全数比较重要的几个性质
(也是我只知道的几个性质)

所有完全数都是三角形数
目前截止发现的所有完全数都以66或2828结尾
到现在为止,数学家们一共发现了4848个完全数,且4848个完全数全部是偶数
如果有人们没有找到的奇完全数,则它一定可以写成12p+112p+1或36p+936p+9的形式,而且pp是素数
奇完全数一定大于1030010300
完全数的约数的倒数之和为调和数
完全数可以表示成连续奇数的立方和
完全数可以表示成22的连续自然数的次幂之和,且这些自然数的数量必定是素数
完全数计算法
若2p−12p−1是素数(亦称其为梅森素数),则2p−1∗(2p−1)2p−1∗(2p−1)是完全数.

时间复杂度
这里数据小了一点,对于每个数据时间复杂度为O(1)O(1).
数据再大我都不怕,反正现在找到48个不如列个map然后映射一个布尔类不就好了!
假装你在屏幕前听到了bjq掌脸的声音

C++ 代码
其实这里用Python3比较好写

#include <bits/stdc++.h>

using namespace std;

int t, n;

int main() {
    cin >> t;
    while (t--) {
        cin >> n;
        if (n == 6 || n == 28 || n == 496 || n == 8128 || n == 33550336)  
            cout << n << " is perfect" << endl;
        else cout << n << " is not perfect" << endl;
    }

    return 0;
}

作者：bjq
链接：https://www.acwing.com/solution/content/10289/

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    int n;
    cin >> n;

    while (n -- )
    {
        int x;
        cin >> x;

        int s = 0;
        for (int i = 1; i * i <= x; i ++ )
            if (x % i == 0)
            {
                if (i < x) s += i;
                if (i != x / i && x / i < x) s += x / i;
            }

        if (s == x) printf("%d is perfect\n", x);
        else printf("%d is not perfect\n", x);
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/237521/


输入若干个整数对M,N，对于每个数对，输出以这两个数为最大值和最小值的公差为1的等差数列。

注意，当输入整数对中，任意一个数为0或负整数时，立即停止输入，且该组数对无需作任何处理。

样例
数据范围
M,N≤100M,N≤100
输入样例：
2 5
6 3
5 0
输出样例：
2 3 4 5 Sum=14
3 4 5 6 Sum=18
算法1
C++ 代码
#include <iostream>

using namespace std;

int main()
{
    int n, m;

    while (cin >> n >> m, n > 0 && m > 0)
    {

        if (n > m) swap(n, m);

        int s = 0;
        for (int i = n; i <= m; i ++)
        {
            cout << i << ' ';

            s += i;
        }
        printf ("Sum=%d\n", s);
    }
    return 0;
}

作者：小鑫鑫
链接：https://www.acwing.com/solution/content/7983/

#include <iostream>
#include <algorithm>

using namespace std;

int main()
{
    int n, m;
    while (cin >> n >> m, n > 0 && m > 0)
    {
        if (n > m) swap(n, m);

        int sum = 0;
        for (int i = n; i <= m; i ++ )
        {
            cout << i << ' ';
            sum += i;
        }

        cout << "Sum=" << sum << endl;
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/237512/

其实数组都不用开(炒鸡短)
#include<stdio.h>
long long n,a,b=1,c;
int main(){
    scanf("%d",&n);
    while(n--){
        printf("%d ",a);
        c=a+b;
        a=b,b=c;
    }
}
附赠高精度版一份(求点赞)
#include<stdio.h>
#include<memory.h>
#pragma GCC optimize(3)
#define M 100001
int n,a[M],b[M],c[M],la=1,lb=1;
void add(){
    memcpy(c,b,sizeof(b));
    la=lb;
    for(int i=1;i<=lb;i++){
        b[i]+=a[i];
        if(b[i]>9&&i<lb) b[i]-=10,b[i+1]++;
    }
    if(b[lb]>9) b[lb]-=10,b[++lb]++;
    memcpy(a,c,sizeof(c));
}
int main(){
    scanf("%d",&n),b[1]=1;
    while(n--){
        for(int i=la;i;i--) printf("%d",a[i]);
        printf(" ");
        add();
    }
}

作者：第一WA者金银花
链接：https://www.acwing.com/solution/content/3523/

#include <iostream>

using namespace std;

int main()
{
    int n;
    cin >> n;

    int a = 0, b = 1;

    for (int i = 0; i < n; i ++ )
    {
        cout << a << ' ';
        int c = a + b;
        a = b;
        b = c;
    }

    cout << endl;

    return 0;
}

#include<stdio.h>
int x,y,ans,t;
int main(){
    scanf("%d",&t);
    while(t--){
        ans=0;
        scanf("%d%d",&x,&y);
        for(int i=(x<y?x:y)+1;i<(x>y?x:y);i++) if(i&1) ans+=i;
        printf("%d\n",ans);
    }
}

#include<iostream>
using namespace std;
int main()
{

    int N,X,count=0;
    cin >> N;
    for(int i=0;i<N;i++)
    {
        cin >> X;
        if(X>=10&&X<=20) count++;
    }
    printf("%d in\n",count);
    printf("%d out",N-count);
    return 0;
}

#include <iostream>
using namespace std;

int main() {
    int n, x, cnt = 0;;
    cin >> n;
    for(int i = 1; i <= n; i++) {
        cin >> x;
        if(x < 10 || x > 20) continue;
        cnt++;
    }
    cout << cnt << " in " << endl;
    cout << n - cnt << " out " << endl;
    return 0;
}


在天梯模式里我手写的输出，累死了也紧张死了，现在满身的荣誉感

#include <iostream>
#include <cstdio>
using namespace std;
typedef long long ll;
ll n,a,c=0,r=0,f=0,s;
char p;
int main()
{
    scanf("%lld",&n);
    for(;n--;)
    {
        scanf("%lld %c",&a,&p);
        if(p=='C')c+=a;
        else if(p=='R')r+=a;
        else f+=a;
    }s=c+r+f;
    printf("Total: %lld animals\nTotal coneys: %lld\nTotal rats: %lld\nTotal frogs: %lld\nPercentage of coneys: %.2lf %\nPercentage of rats: %.2lf %\nPercentage of frogs: %.2lf %",s,c,r,f,c*100.0/s,r*100.0/s,f*100.0/s);
    return 0;
}

作者：皮KA丘
链接：https://www.acwing.com/solution/content/17796/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    int n;
    cin >> n;

    int c = 0, r = 0, f = 0;
    for (int i = 0; i < n; i ++ )
    {
        int k;
        char t;
        scanf("%d %c", &k, &t);  // scanf在读入字符时，不会自动过滤空格、回车、tab
        if (t == 'C') c += k;
        else if (t == 'R') r += k;
        else f += k;
    }

    int s = c + r + f;
    printf("Total: %d animals\n", s);
    printf("Total coneys: %d\n", c);
    printf("Total rats: %d\n", r);
    printf("Total frogs: %d\n", f);
    printf("Percentage of coneys: %.2lf %%\n", (double)c / s * 100);
    printf("Percentage of rats: %.2lf %%\n", (double)r / s * 100);
    printf("Percentage of frogs: %.2lf %%\n", (double)f / s * 100);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/237478/


#include<iostream>
using namespace std;
int main()
{
    int n;
    cin>>n;
    for(int i=1;i<=10;i++)
        cout<<i<<" x "<<n<<" = "<<i*n<<endl;
    return 0;
}



题目描述
读取一个整数X，输出X之后的6个奇数，如果X也是奇数，那么它也算作6个奇数之一。

样例
输入样例：
9
输出样例：
9
11
13
15
17
19
算法1
C++ 代码
#include <iostream>

using namespace std;

int main()
{
    long long x;
    cin >> x;
    if (x % 2 == 0)
    printf ("%d\n%d\n%d\n%d\n%d\n%d\n", x + 1, x + 3, x + 5, x + 7, x + 9, x + 11);
    if (x % 2 == 1)
    printf ("%d\n%d\n%d\n%d\n%d\n%d\n", x, x + 2, x + 4, x + 6, x + 8, x + 10);
    return 0;    
}
算法2
C++ 代码
#include <iostream>

using namespace std;

int main()
{
    int x;
    cin >> x;

    for (int i = x, j = 0; j < 6; i ++ )
        if (i % 2)
        {
            cout << i << endl;
            j ++ ;
        }

    return 0;
}

作者：小鑫鑫
链接：https://www.acwing.com/solution/content/6102/




#include <iostream>
using namespace std;

int main() {
    int x;
    cin >> x;
    x % 2 ? x : x += 1;
    for(int i = 1; i <= 6; i++) {
        cout << x << endl;
        x += 2;
    }
    return 0;
}


#include<stdio.h>
int n,ans=2;
int main(){
    scanf("%d",&n);
    while(ans<10000) printf("%d\n",ans),ans+=n;
}

#include <bits/stdc++.h>

using namespace std;

int main()
{
    int n, m, res  = 0, cnt = 0;
    cin >>n >> m;
    for(int i = 1; i <= n; i++){

        for(int j = 1; j <= m; j++){
            res++;


            if(res % m == 0) cout<< "PUM" << endl;
            if((res < n *m) && (res % m !=0)) cout<< res << " ";

        }
    }

    return 0;
}

作者：zhiling
链接：https://www.acwing.com/solution/content/1932/


输入一个整数N，按照从小到大的顺序输出它的全部约数。

样例
输入样例：
6
输出样例：
1
2
3
6
算法1
C++ 代码
#include <iostream>
#include <algorithm>

using namespace std;

int main()
{
    int n;

    cin >> n;

    for (int i = 1; i <= n; i ++)
    {
        if (n % i == 0)

            cout << i << endl;
    }

    return 0;
}



我刚开始居然没看懂题目，真是惭愧
#include<stdio.h>
int x,y,n;
int main(){
    while(~scanf("%d",&n)){
        if(n<=0) continue;
        if(!x) x=n;
        else{
            y=n;
            break;
        }
    }
    printf("%d",(2*x+y-1)*y/2);
}


#include<stdio.h>
int main(int n){
    while(~scanf("%d",&n),n){
        for(int i=1;i<=n;i++) printf("%d ",i);
        puts("");
    }
    return 0;
}

#include<stdio.h>
int k,n,maxx,ma;
int main(){
    while(~scanf("%d",&k)){
        n++;
        if(k>maxx){
            maxx=k;
            ma=n;
        }
    }
    printf("%d\n%d",maxx,ma);
}

#include<stdio.h>
int main(int x,int y,int ans){
    ans=0;
    scanf("%d%d",&x,&y);
    for(int i=(x<y?x:y)+1;i<(x>y?x:y);i++) if(i&1) ans+=i;
    printf("%d",ans);
}
#include<iostream>
using namespace std;
int main()
{
    int n=6,x=0;
    while(n--)
    {
        float c;
        cin>>c;
        if(c>0) x++;
    }
    cout<<x<<" positive numbers";
    return 0;
}
#include<iostream>
using namespace std;
int main()
{
    int x;
    cin>>x;
    for(int i=1;i<=x;i+=2) cout<<i<<endl;
    return 0;
}
#include<iostream>
using namespace std;
int main()
{
    for(int i=2;i<=100;i+=2) cout<<i<<endl;
    return 0;
}


section 2

#include<iostream>
using namespace std;
int main()
{
    double N1,N2,N3,N4,media;
    cin>>N1>>N2>>N3>>N4;
    media=(N1*2+N2*3+N3*4+N4*1)/10;
    printf("Media: %.1lf\n",media);
    if(media>=7) cout<<"Aluno aprovado."<<endl;
    else if(media<5) cout<<"Aluno reprovado."<<endl;
    else
    {
        cout<<"Aluno em exame."<<endl;
        double Y,Z;
        cin>>Y;
        printf("Nota do exame: %.1lf\n",Y);
        Z=(media+Y)/2;
        if(Z>=5) cout<<"Aluno aprovado."<<endl;
        else cout<<"Aluno reprovado."<<endl;
        printf("Media final: %.1lf",Z);
    }
    return 0;
}

作者：Struggle
链接：https://www.acwing.com/solution/content/13564/

#include <cstdio>
#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    double n1, n2, n3, n4;
    scanf("%lf%lf%lf%lf", &n1, &n2, &n3, &n4);

    double x = (n1 * 2 + n2 * 3 + n3 * 4 + n4) / 10;

    printf("Media: %.1lf\n", x + 1e-8);  // 为了防止出现4.8499999999这种极端情况
    if (x >= 7) printf("Aluno aprovado.\n");
    else if (x < 5) printf("Aluno reprovado.\n");
    else
    {
        printf("Aluno em exame.\n");
        double y;
        scanf("%lf", &y);
        printf("Nota do exame: %.1lf\n", y + 1e-8);
        double z = (x + y) / 2;
        if (z >= 5) printf("Aluno aprovado.\n");
        else printf("Aluno reprovado.\n");
        printf("Media final: %.1lf\n", z + 1e-8);
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/229374/

#include <cstdio>
#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    double a, b, c;
    cin >> a >> b >> c;

    double delta = b * b - 4 * a * c;
    if (delta < 0 || a == 0) printf("Impossivel calcular\n");
    else
    {
        delta = sqrt(delta);
        double x1 = (-b + delta) / (2 * a);
        double x2 = (-b - delta) / (2 * a);

        printf("R1 = %.5lf\n", x1);
        printf("R2 = %.5lf\n", x2);
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/229356/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

C++ 代码
#include<cstdio>
#include<iostream>
#include<cmath>
using namespace std;
int main()
{
    double a,b,c;
    cin>>a>>b>>c;
    if(b*b-4*a*c<0||a==0)
    {
        cout<<"Impossivel calcular";
        return 0;
    }
    else printf("R1 = %.5f\nR2 = %.5f\n",(-b+sqrt(b*b-4*a*c))/(2*a),(-b-sqrt(b*b-4*a*c))/(2*a));
    return 0;
}
本题主要要注意的是运算符的优先级，输出格式的要求，
以及对于一些特殊情况的判断

作者：wkj
链接：https://www.acwing.com/solution/content/2673/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include<cstdio>
#include<iostream>
using namespace std;
int main()
{
    int a,b,c;
    cin>>a>>b>>c;
    int x,y,z;
    x=max(a,max(b,c));
    y=min(a,min(b,c));
    z=a+b+c-x-y;
    cout<<y<<endl;
    cout<<z<<endl;
    cout<<x<<endl;
    cout<<endl;
    cout<<a<<endl;
    cout<<b<<endl;
    cout<<c<<endl;
    return 0;
}

作者：只要是你呀
链接：https://www.acwing.com/solution/content/9383/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


最近刚学了sort函数，所以这题可以用sort函数写。
上面的写法只适合数少的情况，当数多时就不再适用，
所以用sort写比较好。
sort函数在库

<algorithm>中
#include<iostream>
#include<algorithm>
using namespace std;
bool comp(int a,int b)
{
    return a<b;//按升序排列
}
int main()
{
    int a[3];
    int b[3];
    for(int i=0;i<3;i++)
    {
        cin>>a[i];
        b[i]=a[i];//方便原来的顺序输出
    }
    sort(a,a+3,comp);
    for(int i=0;i<3;i++)
        cout<<a[i]<<endl;
    cout<<endl;
    for(int i=0;i<3;i++)
        cout<<b[i]<<endl;
    return 0;
}

作者：只要是你呀
链接：https://www.acwing.com/solution/content/9383/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include <iostream>

using namespace std;

int main()
{
    int a, b, c;
    cin >> a >> b >> c;

    int x = a, y = b, z = c;

    if (b < a)
    {
        int t = a;
        a = b;
        b = t;
    }
    if (c < a)
    {
        int t = a;
        a = c;
        c = t;
    }
    if (c < b)
    {
        int t = b;
        b = c;
        c = t;
    }

    cout << a << endl << b << endl << c << endl << endl;
    cout << x << endl << y << endl << z << endl;

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/229340/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

读取一个保留两位小数的浮点数值，表示一个公民的工资。

在公民交纳个人所得税时，不同收入部分需要交纳的税率也是不同的。

请根据下表确定该公民需要交纳的个人所得税是多少。

数据范围
0≤公民工资≤5000

样例
输入样例1：
3002.00
输出样例1：
R$ 80.36
输入样例2：
1700.00
输出样例2：
Isento
输入样例3：
4520.00
输出样例3：
R$ 355.60
样例解释
对于样例1，0~2000.00部分不用缴税，2000.01~3000.00部分按8%的税率缴税，共计1000 * 8% = 80，3000.01~3002.00部分按18%的税率缴税，共计2 * 18% = 0.36，合计80.36。

对于样例2，公民收入未超过2000，所以输出Isento。

对于样例3，0~2000.00部分不用缴税，2000.01~3000.00部分按8%的税率缴税，共计1000 * 8% = 80，3000.01~4500.00部分按18%的税率缴税，共计1500 * 18% = 270，4500.01~4520部分按28%的税率缴税，共计20 * 28% = 5.60，合计355.60。
算法1
C++ 代码
#include <iostream>

using namespace std;

int main()
{
    double x;

    cin >> x;

    if(x <= 2000.00) printf("Isento");


    else if(x <=3000.00) printf("R$ %.2lf\n", (x - 2000.00) * 0.08);

    else if(x <= 4500.00) printf("R$ %.2lf\n", 80 + (x - 3000.00) * 0.18);

    else if(x > 4500.00) printf("R$ %.2lf\n", 80 + 270 + (x - 4500.00) * 0.28);

    return 0;
}

作者：小鑫鑫
链接：https://www.acwing.com/solution/content/7980/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include <cstdio>

int main()
{
    double x;
    scanf("%lf", &x);

    double sum = 0;
    if (x > 2000)
    {
        double y = 3000;
        if (x < 3000) y = x;
        sum += (y - 2000) * 0.08;
    }
    if (x > 3000)
    {
        double y = 4500;
        if (x < 4500) y = x;
        sum += (y - 3000) * 0.18;
    }
    if (x > 4500) sum += (x - 4500) * 0.28;

    if (sum == 0) printf("Isento");
    else printf("R$ %.2lf\n", sum);

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/229326/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 668. 游戏时间2    原题链接    中等
作者：    Belous ,  2020-03-17 15:11:31 ,  阅读 371

5


4
预先设定差一天后用余数处理，需要注意时间差为24小时的时候取余会变为0，需要特殊处理一下。
#include<stdio.h>
int main(void)
{
    int a,b,c,d;
    scanf("%d%d%d%d",&a,&b,&c,&d);
    b+=a*60,d+=c*60;
    a=(d-b+24*60)%(24*60);
    a=(a)?(a):(24*60);
    printf("O JOGO DUROU %d HORA(S) E %d MINUTO(S)\n",a/60,a%60);
    return 0;
}

作者：Belous
链接：https://www.acwing.com/solution/content/10093/


AcWing 666. 三角形类型    原题链接    简单
作者：    ymy ,  2020-03-03 15:28:42 ,  阅读 516

3


1
#include<iostream>
#include<algorithm>
using namespace std;

int main()
{
    double a,b,c;
    double d[3]={0};
    for(int i=0;i<3;i++)  cin>>d[i];
    sort(d,d+3);
    a = d[2];
    b = d[1];
    c = d[0];

    if(a >= b + c)  cout << "NAO FORMA TRIANGULO"<<endl;
    else
    {
        if(a*a == b*b + c*c)  cout << "TRIANGULO RETANGULO"<<endl;
        if(a*a > b*b + c*c)  cout << "TRIANGULO OBTUSANGULO"<<endl;
        if(a*a < b*b + c*c)  cout << "TRIANGULO ACUTANGULO"<<endl;
        if(a==b && b==c && a==c)  cout << "TRIANGULO EQUILATERO"<<endl;
        if((a==b && a!=c && b!=c) || (a==c && a!=b && c!=b)||(b==c && b!=a && c!=a))   cout << "TRIANGULO ISOSCELES"<<endl;
    }
    return 0;
}

作者：ymy
链接：https://www.acwing.com/solution/content/9460/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include <iostream>

using namespace std;

int main()
{
    double a, b, c;
    cin >> a >> b >> c;

    if (b > a)
    {
        double t = a;
        a = b;
        b = t;
    }
    if (c > a)
    {
        double t = a;
        a = c;
        c = t;
    }
    if (c > b)
    {
        double t = b;
        b = c;
        c = t;
    }

    if (a >= b + c) cout << "NAO FORMA TRIANGULO" << endl;
    else
    {
        if (a * a == b * b + c * c) cout << "TRIANGULO RETANGULO" << endl;
        if (a * a > b * b + c * c) cout << "TRIANGULO OBTUSANGULO" << endl;
        if (a * a < b * b + c * c) cout << "TRIANGULO ACUTANGULO" << endl;
        if (a == b && b == c) cout << "TRIANGULO EQUILATERO" << endl;
        else if (a == b || b == c || a == c) cout << "TRIANGULO ISOSCELES" << endl;
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/229300/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 662. 点的坐标    原题链接    简单
作者：    zyy1313 ,  2020-05-02 08:52:05 ,  阅读 185

3


#include<iostream>
#include<cstdio>
using namespace std;
int main(){
    double a,b;
    cin>>a>>b;
    if(a==0&&b==0) cout<<"Origem";
    else if(a==0)  cout<<"Eixo Y";
    else if(b==0)  cout<<"Eixo X";
    else if(a>0&&b>0) cout<<"Q1";
    else if(a<0&&b>0) cout<<"Q2";
    else if(a<0&&b<0) cout<<"Q3";
    else if(a>0&&b<0) cout<<"Q4";
     return 0;
}

作者：zyy1313
链接：https://www.acwing.com/solution/content/12534/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 671. DDD    原题链接    简单
作者：    zyy1313 ,  2020-05-02 08:45:50 ,  阅读 193

2





#include <iostream>

using namespace std;

int main()
{
    int x;
    cin >> x;
    if (x == 61) cout << "Brasilia" << endl;
    else if (x == 71) cout << "Salvador" << endl;
    else if (x == 11) cout << "Sao Paulo" << endl;
    else if (x == 21) cout << "Rio de Janeiro" << endl;
    else if (x == 32) cout << "Juiz de Fora" << endl;
    else if (x == 19) cout << "Campinas" << endl;
    else if (x == 27) cout << "Vitoria" << endl;
    else if (x == 31) cout << "Belo Horizonte" << endl;
    else cout << "DDD nao cadastrado" << endl;

    return 0;
}

作者：zyy1313
链接：https://www.acwing.com/solution/content/12533/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


#include<bits/stdc++.h>
using namespace std;
int main(){
    int a,b,c,d;
    cin>>a>>b>>c>>d;
    if(b>c&&d>a&&c+d>a+b&&c>0&&d>0&&a%2==0) cout<<"Valores aceitos"<<endl;
    else cout<<"Valores nao aceitos"<<endl;
    return 0;
}


AcWing 669. 加薪    原题链接    简单
作者：    高小呆 ,  2020-07-06 16:18:15 ,  阅读 147

2


1
这道题虽然简单，但是有几个需要注意的地方：
//1、对于不同的百分比设置一个参数y这样不用对每种情况总是printf;
//2、注意在【0,400】区间，只写<=400即可，不用写>=0

include[HTML_REMOVED]
include[HTML_REMOVED]
using namespace std;
int main()
{
double x;
cin>>x;
double y;//涨的薪水
if(x<=400.00) y=0.15;
else if(x<=800.00) y=0.12;
else if(x<=1200.00) y=0.1;
else if(x<=2000.00) y=0.07;
else if(x>2000.00) y=0.04;
printf(“Novo salario: %.2f\n”,xy+x);
printf(“Reajuste ganho: %.2f\n”,xy);
printf(“Em percentual: %.0f %\n”,y*100);

}

作者：高小呆
链接：https://www.acwing.com/solution/content/15810/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 670. 动物    原题链接    简单
作者：    小鑫鑫 ,  2019-10-06 21:32:22 ,  阅读 615

1


题目描述
给定你三个葡萄牙语单词，这些词将根据下表从左到右定义一个动物。

请你确定并输出这个动物的名称。

样例
输入样例：
vertebrado
mamifero
onivoro
输出样例：
homem
算法1
C++ 代码
#include <iostream>

using namespace std;

int main()
{
    string a, b, c;

    cin >> a >> b >> c;

    if (a == "vertebrado")
    {
        if (b == "ave")
        {
            if (c == "carnivoro") cout << "aguia" << endl;

            else if(c == "onivoro") cout << "pomba" << endl;
        }

        if (b == "mamifero")
        {
            if (c == "onivoro") cout << "homem" << endl;
            else if (c == "herbivoro") cout << "vaca" <<endl;
        }
    }
    if (a == "invertebrado")
    {
        if (b == "inseto")
        {
            if (c == "hematofago") cout << "pulga" << endl;
            else if (c == "herbivoro") cout <<"lagarta" << endl;
        }
        if (b == "anelideo")
        {
            if (c == "hematofago") cout << "sanguessuga" << endl;
            else if (c == "onivoro") cout << "minhoca" << endl;
        }
    }
    return 0;
}

作者：小鑫鑫
链接：https://www.acwing.com/solution/content/5137/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 667. 游戏时间    原题链接    简单
作者：    电光耗子王 ,  2020-05-31 20:23:46 ,  阅读 174

3


题目描述
blablabla

样例
blablabla
C++ 代码
#include<bits/stdc++.h>
using namespace std;
int main()
{
    int a,b;
    cin>>a>>b;
    if(b<a) b+=24;
    if(a==b) cout<<"O JOGO DUROU 24 HORA(S)"<<endl;
    else if(a<b) cout<<"O JOGO DUROU "<<b-a<<" HORA(S)"<<endl;
    return 0;
}

作者：电光耗子王
链接：https://www.acwing.com/solution/content/14003/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


#include <iostream>
using namespace std;
int main(){
    double a;
    cin>>a;
    if (a>=0&&a<=25)
    printf("Intervalo [%d,%d]",0,25);
    else if (a>25&&a<=50)
    printf("Intervalo (%d,%d]",25,50);
    else if (a>50&&a<=75)
    printf("Intervalo (%d,%d]",50,75);
    else if (a>75&&a<=100)
    printf("Intervalo (%d,%d]",75,100);
    else 
    printf("Fora de intervalo");
    return 0;

}

作者：noobcoder
链接：https://www.acwing.com/solution/content/9365/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 664. 三角形-语法题-C++    原题链接    简单
作者：    Struggle ,  2020-05-24 09:22:53 ,  阅读 217

2


#include<iostream>
#include<cmath>
using namespace std;
int main()
{
    double a,b,c;
    cin>>a>>b>>c;
    //判断三边是否可构成三角形   两边之和>第三边 两边之差的绝对值<第三边
    if(a+b>c && fabs(a-b)<c) printf("Perimetro = %.1lf",a+b+c);
    else printf("Area = %.1lf",(a+b)*c/2);
    return 0;
}

作者：Struggle
链接：https://www.acwing.com/solution/content/13592/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 665. 倍数    原题链接    简单
作者：    hevttccao ,  2020-03-01 18:58:07 ,  阅读 293

3


# include <iostream>

using namespace std;

int main ()
{
    int a,b;
    cin>>a>>b;

    if (a%b==0||b%a==0)
    cout<<"Sao Multiplos"<<endl;
    else
    cout<<"Nao sao Multiplos"<<endl;
    return 0;
}

AcWing 660. 零食    原题链接    简单
作者：    小鑫鑫 ,  2019-10-20 20:43:13 ,  阅读 358

1


题目描述
某商店出售5种零食，零食编号为1~5。

5种零食的价目表如下所示：

零食种类 价格
零食 1 R4.00零食2R4.00零食2R 4.50
零食 3 R5.00零食4R5.00零食4R 2.00
零食 5 R$ 1.50
现在给定某种零食的编号和数量，请你计算总价值是多少。

样例
输入样例：
3 2
输出样例：
Total: R$ 10.00
算法1
C++ 代码
#include <iostream>

using namespace std;

int main()
{
    int x, y;
    cin >> x >> y;
    if (x == 1) printf ("Total: R$ %.2lf\n", y * 4.00);
    else if (x == 2) printf ("Total: R$ %.2lf\n", y * 4.50);
    else if (x == 3) printf ("Total: R$ %.2lf\n", y * 5.00);
    else if (x == 4) printf ("Total: R$ %.2lf\n", y * 2.00);
    else if (x == 5) printf ("Total: R$ %.2lf\n", y * 1.50);

    return 0;
}

作者：小鑫鑫
链接：https://www.acwing.com/solution/content/5263/


AcWing 606. 平均数1--语法题-C++    原题链接    简单
作者：    Struggle ,  2020-05-23 10:31:08 ,  阅读 278

3


#include<iostream>
using namespace std;
int main()
{
    double a,b;
    cin>>a>>b;
    printf("MEDIA = %.5lf",(a*3.5+b*7.5)/11);//11=3.5+7.5
    return 0;
}

AcWing 655. 天数转换(与时间转换同理）    原题链接    简单
作者：    noobcoder ,  2020-02-27 16:28:04 ,  阅读 274


#include <iostream>
using namespace std;
int main(){
    int x,y,m,d;
    cin>>x;
    y=x/365;
    m=x%365/30;
    d=x-y*365-m*30;//也可以写成d=x%365%30
    cout<<y<<" ano(s)"<<endl;///注意格式
    cout<<m<<" mes(es)"<<endl;
    cout<<d<<" dia(s)"<<endl;
    return 0;

}

作者：noobcoder
链接：https://www.acwing.com/solution/content/9192/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



AcWing 656. 钞票和硬币    原题链接    中等
作者：    optimjie ,  2020-02-26 18:04:51 ,  阅读 674

15


3
刚才打saber被double精度坑了，所以直接 *100 变成int就能过了

#include <iostream>
#include <cstdio>

using namespace std;

int main()
{
    double n;
    cin >> n;

    int m = (int)(n * 100);

    double a[12] = {10000, 5000, 2000, 1000, 500, 200, 100, 50, 25, 10, 5, 1};
    int ans[12] = {0};

    for (int i = 0; i < 12; i++)
    {
        int cnt = 0;
        while (m >= a[i])
        {
            m -= a[i];
            cnt++;
        }
        ans[i] = cnt;
    }

    puts("NOTAS:");
    for (int i = 0; i < 6; i++)
        printf("%d nota(s) de R$ %.2lf\n", ans[i], (double)a[i] / 100);
    puts("MOEDAS:");
    for (int i = 6; i < 12; i++)
        printf("%d moeda(s) de R$ %.2lf\n", ans[i], (double)a[i] / 100);

    return 0;
}

作者：optimjie
链接：https://www.acwing.com/solution/content/9139/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

AcWing 618. 燃料消耗（超简单做法，看这个就够了）    原题链接    简单
作者：    noobcoder ,  2020-02-27 16:12:56 ,  阅读 312

2


问题分析
本题数据量很大，需要考虑到路程由行驶速度与时间乘积得到
时间和速度的上限均为10^9，那么得到的数据大小很可能特别大，因此此处应考虑到出现爆int的现象

代码如下：
#include <iostream>
#include <bits/stdc++.h>
using namespace std;  

int main(){
    long long s,t;
    cin>>s>>t;
    cout<<fixed<<setprecision(3)<<s*t/12.0<<endl;
    return 0;
}
总结：
采用setprecision实现数据精度的控制
因为数据很大，防止数据溢出，采用long long

作者：noobcoder
链接：https://www.acwing.com/solution/content/9190/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#include <cstdio>

int main()
{
    double s, t;
    scanf("%lf%lf", &s, &t);
    printf("%.3lf\n", s * t / 12);

    return 0;
}

#include <cstdio>

int main()
{
    double m;
    scanf("%lf", &m);
    int n = m * 100;

    printf("NOTAS:\n");
    printf("%d nota(s) de R$ 100.00\n", n / 10000); n %= 10000;
    printf("%d nota(s) de R$ 50.00\n", n / 5000); n %= 5000;
    printf("%d nota(s) de R$ 20.00\n", n / 2000); n %= 2000;
    printf("%d nota(s) de R$ 10.00\n", n / 1000); n %= 1000;
    printf("%d nota(s) de R$ 5.00\n", n / 500); n %= 500;
    printf("%d nota(s) de R$ 2.00\n", n / 200); n %= 200;

    printf("MOEDAS:\n");
    printf("%d moeda(s) de R$ 1.00\n", n / 100); n %= 100;
    printf("%d moeda(s) de R$ 0.50\n", n / 50); n %= 50;
    printf("%d moeda(s) de R$ 0.25\n", n / 25); n %= 25;
    printf("%d moeda(s) de R$ 0.10\n", n / 10); n %= 10;
    printf("%d moeda(s) de R$ 0.05\n", n / 5); n %= 5;
    printf("%d moeda(s) de R$ 0.01\n", n / 1); n %= 1;

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/220639/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。