```
开放寻址法
#include <cstring>
#include <iostream>

using namespace std;

const int N = 200003, null = 0x3f3f3f3f;

int h[N];

int find(int x)
{
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x)
    {
        t ++ ;
        if (t == N) t = 0;
    }
    return t;
}

int main()
{
    memset(h, 0x3f, sizeof h);

    int n;
    scanf("%d", &n);

    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);
        if (*op == 'I') h[find(x)] = x;
        else
        {
            if (h[find(x)] == null) puts("No");
            else puts("Yes");
        }
    }

    return 0;
}
拉链法
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100003;

int h[N], e[N], ne[N], idx;

void insert(int x)
{
    int k = (x % N + N) % N;
    e[idx] = x;
    ne[idx] = h[k];
    h[k] = idx ++ ;
}

bool find(int x)
{
    int k = (x % N + N) % N;
    for (int i = h[k]; i != -1; i = ne[i])
        if (e[i] == x)
            return true;

    return false;
}

int main()
{
    int n;
    scanf("%d", &n);

    memset(h, -1, sizeof h);

    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);

        if (*op == 'I') insert(x);
        else
        {
            if (find(x)) puts("Yes");
            else puts("No");
        }
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/45308/
```

data structure 3 is all about hash; discretization is a very special case of hash, could have just used hash; no need for discretization; no need to have special item in basic knowledge

william lin tracking - codes save his past contests
lower_bound, upper_bound() must be on sorted vector
https://www.geeksforgeeks.org/upper_bound-and-lower_bound-for-non-increasing-vector-in-c/
https://www.geeksforgeeks.org/upper_bound-and-lower_bound-for-vector-in-cpp-stl/
reverse()
https://www.geeksforgeeks.org/stdreverse-in-c/
summary and take notes - so no need to study again!

quick sort and merge sort are only two sorts he talked about: both
divide and conquer, both recursion

prefix sum - used to get section sum s[l][r]=a[l]+...a[r] by preprocessing an additiona
array s[i]=a[0]+a[1]+...a[i], s[l][r] = s[r]-s[l-1]; prefix sum index starts
from 1 not 0. It takes O(n) time to build, but takes O(1) time to use

difference is the reverse operation of prefix sum;
a[i] to b[i]=a[i]-a[i-1];
O(n) time to get original array a[i] 
use - takes O(1) time to add a constant to a section in middle of 
a[l] to a[r]

the idea is to get prepared first, to get O(1) in use
2d application of prefix sum and difference

325 problems in li yudong book - daka, jinjie; if done IO can do

for 10^6 I/O java is a nightmare; lowered to 10^5 just for java; 
java big array easily leads to OOM	

python to do unique as in c++:     x=list(set(x))





first check data range to decide what algorithm to use - 
100 = n^3 floyd

826 problem -
Problem video list - 
https://www.acwing.com/user/myspace/video/1/14/
Code - 
https://www.acwing.com/activity/content/activity_person/content/1349/9/

Code template
https://www.acwing.com/user/myspace/blog/1/

Leetcode code-
https://www.acwing.com/activity/content/activity_person/content/29799/1/

https://github.com/oeddyo/algorithm
https://botao.hu/

https://github.com/oeddyo/algorithm/blob/master/resources/%E7%89%9B%E4%BA%BA%E8%B0%88ACM%E7%BB%8F%E9%AA%8C(%E5%8C%85%E6%8B%AC%E5%9B%BD%E5%AE%B6%E9%9B%86%E8%AE%AD%E9%98%9F%E8%AE%BA%E6%96%87)/%E5%9B%BD%E5%AE%B6%E9%9B%86%E8%AE%AD%E9%98%9F%E8%AE%BA%E6%96%87/%E5%9B%BD%E5%AE%B6%E9%9B%86%E8%AE%AD%E9%98%9F2007%E8%AE%BA%E6%96%87%E9%9B%86/day2/7.%E8%83%A1%E4%BC%AF%E6%B6%9B%E3%80%8A%E6%9C%80%E5%B0%8F%E5%89%B2%E6%A8%A1%E5%9E%8B%E5%9C%A8%E4%BF%A1%E6%81%AF%E5%AD%A6%E7%AB%9E%E8%B5%9B%E4%B8%AD%E7%9A%84%E5%BA%94%E7%94%A8%E3%80%8B.pdf

Meituan interview one problem - 
https://www.jianshu.com/p/5d10b4cf1753
https://www.jianshu.com/u/c6ad3f2ed2d6
https://mp.weixin.qq.com/s?__biz=MzUyNjQxNjYyMg==&mid=2247484350&idx=1&sn=fc88aa125f5a5269575b4c4d83774f41&chksm=fa0e6c3fcd79e5297257a05b8c75898b4059b1193956c702ff5ef3f2d8d46432bb7484bf6428&token=110841213&lang=zh_CN#rd



da ka
https://www.acwing.com/activity/content/activity_person/content/1349/7/


huo dong - activity
https://www.acwing.com/activity/content/discuss/31/1/


personal space-
https://www.acwing.com/user/myspace/video/1/14/

Cin can filter spaces; scanf %c can't filter space;
Scanf is faster than cin; 

Int/bool %d
Float %f
Double %lf
Char %c

Long long %lld

Xueqiu
https://xueqiu.com/u/1247347556

Basic ideas everyone has, it's how to use/apply that matters

Knapsack 1
https://www.acwing.com/problem/content/video/2/


https://v.douyu.com/author/PDAPVoKO3wxN
Usaco 2060
https://www.acwing.com/problem/41/

Usaco 1339
https://www.acwing.com/problem/27/


FFT in c++
https://www.bilibili.com/video/BV1Y7411W73U

OIer -

https://github.com/Tony031218
https://space.bilibili.com/171431343/album

https://github.com/GolemHerry/manim-on-macOS
http://bhowell4.com/manic-install-tutorial-for-mac/

python manim.py p/old/PeriodicTable/PeriodicTable.py
/Users/rsdata/manim

My websites:
https://sites.google.com/site/mathcounts2020
https://sites.google.com/view/procoding
https://sites.google.com/view/theknowledge-lewis
https://sites.google.com/iu.edu/programming

Douyu yxc AcWing videos - 
https://v.douyu.com/author/PDAPVoKO3wxN
2018-06-18-19pm 数论问题讲解 - euler phi, factor counting
2018-06-25/26-20pm 最小生成树算法及IndeedTokyo笔试题讲解; prim() kruskal(), indeed interview
2018-07-02-20pm 动态规划 - leetcode mostly and one toutiao interview question
2018-07-09-20pm 二分专题
2018-07-16-20pm 暑期LeetCode打卡——Week1，链表(1/2), 22pm 暑期LeetCode打卡——Week1，链表(2/2)

2018-07-23-20pm 暑期LeetCode打卡——Week2，递归(1/2), 22pm; leetcode 17 46 dfs 47 78

2018-07-30-19pm Week3 树专题+KickStart D轮讲解（1/2, 21pm Week3 树专题+KickStart D轮讲解（2/2）
2018-08-06-3 parts Week4 动态规划专题 (5/5)
2018-08-13-19pm Week5 字符串处理专题 (1/2), 21pm
2018-08-20-20pm Week6 二分与单调队列/栈专题(1/2)， 2018-08-20-22pm Week6 二分与单调队列/栈专题(2/2)
2018-08-30 20pm, 22pm Week7 哈希表专题
2018-09-3 20p Week8 动态规划专题II (1/2)
2018-09-10 20pm 22pm Week8 动态规划专题II (1/2); leetcode 专题打卡， 第一，第二期； #19- 复杂度分析

2018-09-17 20pm leetcode 第二期Week1——深搜/宽搜专题, flood fill
2018-09-24-20pm 第二期Week2——位运算专题
2018-10-01-20pm 第二期Week3——贪心专题; leetcode 2nd season has only 3 weeks
2018-10-08 outage
2018-10-15-20pm 背包九讲(1/2), 21pm 背包九讲(1/2)
2018-10-23-20pm 背包九讲(2/2) 
2018/10/23 LeetCode season 3 刷题打卡Week1题目走 2018-10-23 20点场
2018-10-30-20pm 刷题活动第三期Week2 DFS专题 2018-10-30 20点场
2018-11-05 20点场 刷题活动第三期Week3 数学专题 2018-11-05 20点场
2018-11-19 19点场 leetcode 刷题第四期Week1 Google专题 2018-11-19 19点场
2018-11-26 19点场 leetcode 刷题第四期Week2Facebook专题 2018-11-26 19点场
2018-12-03 20点场 leetcode 刷题第四期Week3FGoogle专题 2018-12-03 20点场
2018-12-17 20点场 浅谈排序算法 2018-12-17 20点场
2018-12-24 20点场 浅谈排序算法（下） 2018-12-24 20点场
2018-12-31 17点场 剑指Offer-Week1 2018-12-31 17点场
剑指Offer-Week2 2019-01-07 20点场
剑指Offer-Week3 2019-01-14 19点场
0x00基本算法--0x01 位运算 2019-01-15 19点场
0x00基本算法--0x02递推与递归 2019-01-18 19点场
0x00基本算法--0x03前缀和与差分 2019-01-20 20点场
《剑指Offer》打卡活动——Week4 2019-01-21 20点场,  《剑指Offer》打卡活动——Week4 2019-01-21 22点场
《算法竞赛进阶指南》0x04. 二分 2019-01-22 19点场
《算法竞赛进阶指南》0x05. 排序 2019-01-24 21点场
牛客寒假算法基础集训营1 2019-01-25 20点场
牛客寒假算法基础集训营1 2019-01-26 19点场
算法竞赛进阶指南 0x06 倍增 2019-01-27 21点场
牛客寒假算法基础集训营2讲解 2019-02-02 20点场, 牛客寒假算法基础集训营2讲解 2019-02-02 21点场, 牛客寒假算法基础集训营2讲解 2019-02-02 22点场
牛客寒假算法基础集训营2讲解 2019-02-07 20点场, 牛客寒假算法基础集训营2讲解 2019-02-07 21点场
剑指Offer打卡活动——Week5 2019-02-24 20点场, 剑指Offer打卡活动——Week5 2019-02-24 22点场
算法竞赛进阶指南 0x07 贪心 2019-02-26 19点场, 算法竞赛进阶指南 0x07 贪心 2019-02-26 21点场
算法竞赛进阶指南 0x08 练习 2019-03-03 20点场, 算法竞赛进阶指南 0x08 练习 2019-03-03 22点场
算法竞赛进阶指南 0x08 练习 2019-03-06 18点场, 
蓝桥杯 B 组！ 2019-03-24 21点场, 蓝桥杯 B 组！ 2019-03-24 23点场, 
算法竞赛进阶指南 第二章 总结与练习 2019-05-07 20点场, 算法竞赛进阶指南 第二章 总结与练习 2019-05-07 22点场


2019-02-02, 
19-01-24, Week7 哈希表专题; 
19-01-26 19pm 200k data will blow up stack; don't use dfs recursion; use bfs
https://v.douyu.com/show/DrwnvzygQDlWPNaX 2018-6-18
2019-11-22 has 5-bit numbering

Wfc to fido move
你在乎孩子，TA不在乎你

Week1完成情况：0/14
链表专题07/09 00:00:00 - 07/15 23:59:59
星期一
LeetCode 19. Remove Nth Node From End of List418人打卡
LeetCode 83. Remove Duplicates from Sorted List373人打卡
星期二
LeetCode 206. Reverse Linked List346人打卡
LeetCode 92. Reverse Linked List II300人打卡
星期三
LeetCode 61. Rotate List291人打卡
LeetCode 143. Reorder List252人打卡
星期四
LeetCode 21. Merge Two Sorted Lists269人打卡
LeetCode 160. Intersection of Two Linked Lists258人打卡
星期五
LeetCode 141. Linked List Cycle258人打卡
LeetCode 147. Insertion Sort List239人打卡
星期六
LeetCode 138. Copy List with Random Pointer182人打卡
LeetCode 142. Linked List Cycle II217人打卡
星期天
LeetCode 148. Sort List179人打卡
LeetCode 146. LRU Cache

作者：AcWing
链接：https://www.acwing.com/activity/content/punch_the_clock/1/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

Week2完成情况：0/14
DFS+回溯专题07/16 00:00:00 - 07/22 23:59:59
星期一
LeetCode 17. Letter Combinations of a Phone Number359人打卡
LeetCode 46. Permutations340人打卡
星期二
LeetCode 47. Permutations II311人打卡
LeetCode 78. Subsets313人打卡
星期三
LeetCode 90. Subsets II294人打卡
LeetCode 39. Combination Sum298人打卡
星期四
LeetCode 40. Combination Sum II279人打卡
LeetCode 216. Combination Sum III269人打卡
星期五
LeetCode 22. Generate Parentheses258人打卡
LeetCode 473. Matchsticks to Square215人打卡
星期六
LeetCode 52. N-Queens II184人打卡
LeetCode 37. Sudoku Solver146人打卡
星期天
LeetCode 282. Expression Add Operators102人打卡
LeetCode 301. Remove Invalid Parentheses

链接：https://www.acwing.com/activity/content/punch_the_clock/1/


平衡都是短暂的，不进则退
找不到朋友
炮兵怕碰标兵

google kickstart-
	546	糖果	42.07%	中等
547	滑翔伞	11.76%	简单
548	最有趣的单词搜索	80.00%

Week3完成情况：0/14
树专题07/23 00:00:00 - 07/29 23:59:59
星期一
LeetCode 101. Symmetric Tree307人打卡
LeetCode 104. Maximum Depth of Binary Tree306人打卡
星期二
LeetCode 145. Binary Tree Postorder Traversal266人打卡
LeetCode 105. Construct Binary Tree from Preorder and Inorder Traversal257人打卡
星期三
LeetCode 331. Verify Preorder Serialization of a Binary Tree228人打卡
LeetCode 102. Binary Tree Level Order Traversal248人打卡
星期四
LeetCode 653. Two Sum IV - Input is a BST243人打卡
LeetCode 236. Lowest Common Ancestor of a Binary Tree239人打卡
星期五
LeetCode 543. Diameter of Binary Tree224人打卡
LeetCode 124. Binary Tree Maximum Path Sum206人打卡
星期六
LeetCode 87. Scramble String134人打卡
LeetCode 117. Populating Next Right Pointers in Each Node II170人打卡
星期天
hardest one - LeetCode 99. Recover Binary Search Tree111人打卡
LeetCode 337. House Robber III

没有自己的东西
如果没见过，一般做不出来
没有见过，，所以不知道

trie tree, union find set, heap, 
jichuke - wk3 xitike