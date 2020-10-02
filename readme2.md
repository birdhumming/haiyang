https://www.acwing.com/file_system/file/content/whole/index/content/1344583/


搜索与图论
DFS
（一）全排列（回溯）
注：数据范围 n<=30；暴力搜索（全部情况走一遍）
一.基本知识
1.def：向下深搜；“执着”
2.栈stack；空间O(h)
3.结合：回溯；剪枝
二.实现
思考流程：一棵树
1.stack：只存储当前的路径
2.回溯/递归：恢复现场（面对时，所有分支是平行/等价/无先后顺序）
三.难点
1.u和i的关系：当前层包含当前点

（二）n皇后（剪枝）
注：剪枝：提前判断，去除不合法的情况，不用再继续往下搜（e.g.同一行/列/对角线）
一.实现
1.开一些布尔数组，记录状态（e.g.col/dig/dig）

BFS
（一）走迷宫（最短路）
一.思路
二.实现（模板）
（1）def bfs():
1.初始化队列
2.while queue:
定位队头（popleft）
拓展队头（e.g.向量dx/dy）
3.有时根据题意需要特判：距离min；点是否遍历过
(2)main
1.把图读入：开一个g[i][j] for _ in range（n）/(m)等
2.有时根据题意，需要读入状态/距离（e.g.-1等）

树与图的遍历
n<10^4

树与图的存储
（一）邻接矩阵
1.费空间
2.用的较少
（二）邻接表
一.思想/理解
1.给每个节点开一个链表：存储她指向关系的节点（难点，理解开的几个数组代表的含义）
二.实现（与数组–>链表方式相同）
1.开数组：
1）h[N]：N个链表（节点）的链表头（对应之前的head）
2）e[M]：每个节点的值
3）ne[M]：下一个节点
4）idx：当前边/点（e.g.b）
2）main: 初始化 h=-1
2.加入一条边的含义：即在a节点/链表后插入节点b
插入节点：add（a,b）

树与图的dfs遍历
一.准备
1.st：布尔数组，记录状态

二.dfs
1.变成遍历单链表：i=ne[i]

三.main
1.初始化：h=-1
2.找个点开始：dfs(1)

树与图的bfs遍历
一.准备
1.同上，开数组：四个
2.st：布尔数组，记录状态
3.d：记录距离

二.bfs
1.初始化队列
2.while queue:
1）定位队头
2）拓展头/队列

三.main
1.初始化：h=-1
2.找个点开始：bfs(1)

拓扑排序
一.基础知识
1.入度；出度

二.实现
1.queue <– 所有入度为0的点（难点，要理解的地方）
2.结合bfs模板：
while queue:
1)t定位队头
2）枚举t的所有出边 t–>j
删掉t–>j d[j]–
if d[j]==0:
queue <– j


https://www.acwing.com/file_system/file/content/whole/index/content/1344519/

1、strlen是O(n)的复杂度，所以类似for(int i=0;i<strlen(s);i++){}之类的语句，时间复杂度是O(n2)。但是，string对象的size函数的复杂度是O(1)，要注意区别。
2、gets函数已经被noip禁用，可用fgets函数替代，注意：fgets会把最后的回车包含到目标数组。
3、string对象可以进行加法运算，只+两边有string对象，就会全部转为string再连接。注意：如果+两边没有string对象，都是char字符串，那么就会报错，比如：string s="hello"+"world";，正确写法是string s1="hello"; string s2=s1+"world";，这样写也是错的：string s2="dahua" + "hao" + s1;因为先计算第一个+号，但这时两边全是char字符串。


make_pair(a, b); 作用是返回一个pair,两个元素分别为a和b

至于第二个问题，设pair[HTML_REMOVED] p;

则p.first是p的第一个元素，p.second是第二个元素

用户头像
CYa想提高工程力   4小时前     回复
正解! pair<int, int> p (markdown格式以避免与html层源码冲突噢)
看y总经常命名(即定义)为PII 。即typedef pair<int,int> PII;

用户头像
楚天   2小时前     回复
pair是将2个数据组合成一组数据，当需要这样的需求时就可以使用pair，这句话怎么理解

用户头像
楚天   2小时前    回复了 CYa想提高工程力 的评论     回复
pair是将2个数据组合成一组数据，当需要这样的需求时就可以使用pair，这句话怎么理解

用户头像
jvruo   2小时前    回复了 楚天 的评论     回复
给一个例子：

struct node {
int first, second;
};

这就是pair的原型。不过pair使用了模板,可以用pair[HTML_REMOVED]使first类型为int,second类型为double。比如有pair[HTML_REMOVED] p, 则可以使用p.first = 1; printf(“%d”, p.first);这样的语句。

当然如果你实在不习惯也可以自己手写结构体，另外pair定义了小于大于等运算符，是先比较first大小，相同再比较second的大小。

用户头像
jvruo   2小时前    回复了 jvruo 的评论     回复
pair[HTML_REMOVED]应为pair<int, double>





https://www.acwing.com/file_system/file/content/whole/index/content/1343929/