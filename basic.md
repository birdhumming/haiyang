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