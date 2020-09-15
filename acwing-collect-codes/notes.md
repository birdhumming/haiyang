```
https://www.acwing.com/file_system/file/content/whole/index/content/1291096/

https://www.rankred.com/useful-c-cheat-sheets/



https://www.acwing.com/blog/content/3122/

cht讲算法——干货满满！c++STL与手写STL（chtSTL）详解！
作者：    cht ,  2020-07-15 21:22:35 ,  阅读 244

17


17
本文长度恐怖
建议收藏后慢慢食用。
请大家不要快速移动滑动条！
知识点都在中间！！！
请大家不要快速移动滑动条！
知识点都在中间！！！
请大家不要快速移动滑动条！
知识点都在中间！！！
重要的事情说3遍
（所以看在up主这么辛苦的面子上给一个赞吧5555）
c++STL与手写STL（chtSTL）
STL这个东西
很好
但我很恨ta，
而且大家看我天梯都是不用的
CSP不能用！
最后就是：

'vector' was not declared in this scope
'pair' was not declared in this scope
'queue' was not declared in this scope
'deque' was not declared in this scope
……
满纸荒唐言，一把辛酸泪……

零、简介
拥有巨多STL的讲解，后面手写了两个STL。
制作不易，望3连。

一、STL是个好东西吗？
(1)可爱的vector
vector 等价于一个很高大上的冥纸：
变 长 数 组
我给它起了个别名：
贪 吃 蛇 数 组
为啥？
你每次给vector插入元素的时候，它都会自动变长……
那ta到底是怎么实现的呢？
其实非常简单，如果不够长，就像内存申请把size*=2……
所以其实就是一个倍增的思路吗……
具体操作：

push_back():插入
size():查看元素个数，O(1)
clear():清空
empty():是否为空
pop_back():弹出
stack
就是一个栈。
栈拥有先进后出的特点。
可以手写，但第一部分只介绍STL。
具体用法：

stack<int> stk;//定义
stk.push(1);//从栈顶插入
stk.pop();//从栈顶弹出
stk.empty();//是否为空
stk.front();//访问栈顶
queue
就是队列。
队列拥有先进先出的性质。
具体用法：

queue<int> q;//定义一个队列q
q.push(1);//像队尾插入1
q.pop();//弹出队头
q.empty();//返回队列是否为空
q.front();//返回队头
deque
相当于双端队列。
可以进行两端的操作。
这个STL非常新蜜蜂的！

deque<int> dq;//定义一个双端队列
deque.size();//返回长度
dq.push_front(1);//向队头插入
dq.push_back(1);//向队尾插入
dq.pop_front();//删除队头
dq.pop_back();//删除队尾
dq.front();//访问第一个元素
dq.back();//访问最后一个元素
cout << dq[0] << endl;//下标访问
dq.begin();//开头指针迭代器
dq.end();//末尾指针迭代器
dq.clear();//清空
dq.empty();//是否为空
priority_queue
注：需要的库函数是:

#include<queue>
#include<vector>
优先队列，可以理解成堆。
具体操作：

priority_queue<int> heapone;//定义大根堆
priority_queue<int, vector<int>, greater<int>> heaptwo;//定义小根堆
heaptwo.top();//返回堆顶元素
heaptwo.push(1);//向堆插入一个数。
heaptwo.pop();//弹出堆顶元素
heaptwo.empty();//判断堆是否为空
heaptwo.size();//返回元素个数
string
c++ 字符串。
具体操作：

string line;//定义字符串
getline(cin, line);//读入字符串方式1
cin >> line;//读入字符串方式2
//推荐cin读入字符串
line[1];//访问元素
line.substr(x, y);//取出子串
line.size();//返回元素个数
line.length();//和size相同
line.empty();//判断是否为空
ling.clear();//清空
line.push_back();//插入字符
pair
pair相当于存储一个二元组。
排序时会按第二个元素排序。
可以嵌套。
具体使用方法：

pair<int, int> tone;//定义二元组
pair<int, pair<int, int>> ttwo;//定义三元组
pair<int, int> tthree[10086];//定义pair数组
cout << tone.first << endl;//查看tone的第1项。
cout << tone.second << endl;//查看tone的第二项
cout << ttwo.second.second << endl;//循环嵌套
map
这个就很牛了。
具体操作：

map<int, string> Map;//定义一个（前方高能）以int为索引，拥有指向string的指针。
pair<int, string> in = {000, "cht"};
Map.insert(in);//插入
Map.find(000);//返回元素位置
Map.erase(000);//删除元素（至此cht被踢出了Map）
Map.size();//返回大小
Map.begin();//头部迭代器
Map.end();//尾部迭代器
Map.clear();//清空
Map.count();//返回元素个数
Map.empty();//返回map是否为空
Map[0];//重磅！下标访问
set
我已经写腻了，因为这个的操作和map几乎一样啊……
具体操作：

set<int> S;
S.begin();//返回第一个元素
S.end();//返回最后一个元素
S.clear();//清空
S.empty();//是否为空
S.size();//返回长度
S.count(1);//返回1的个数
S.find(1);//寻找
S.erase(1);//删除
S.insert(1);//插入
支持lower_bound()和upper_bound();
还有吗？
当然……

multiset/multimap
就是支持重复元素的set和map，
这里不重复说了。

unordered_set/unordered_multiset/unordered_map/unordered_multimap
无序的set和支持重复元素的无序的set。
无序的map和支持重复元素的无序的map。
与上面类似，支持增删改查。
但不支持

lower_bound()
和

upper_bound();
也不支持迭代器的++，–操作。
因为他们内部是无序的。

list
相当于双向链表。
需要

#include<list>
功能巨多
具体操作：

list<int> L;
L.push_back(1);//插入
L.empty();//判断是否为空
L.begin();//返回头结点
L.end();//返回尾部节点
L.insert(L.begin(), 1);//在指定位置插入
L.erase(L.begin());//删除
L.erase(L.begin(), L.end());//区间删除
L.clear();//清空list
L.push_front();//插入
L.pop_front();//删除
L.pop_back();//删除
L.size();//元素个数
诶大家别打我，我承认这个参考了好吧。
list太难，还是看看官网的介绍吧……

bitset
压位新蜜蜂！
（其实压位我也不太会
压位所做的事情就是：

减 少 内 存
具体使用方法如下：

bitset<10086> B;
~B;//取反
|,&,^;//和一般的与或异或一样。
>>,<<;//移位笑
==,!=;//bitset表情包上线
B[1];//访问元素
B.count();//返回1的个数
B.any();//是否存在1
B.none();//是否为空（全为0）
//bitset英语学习器上线。
B.set();//把所有位搞成1
B.set(1, 1);//将第1位变成1
B.reset();//把所有位变成0
B.flip();//从名字来，取反所有位
B.flip(1);//取反第一位
c++STL完结撒花！
二、手写STL
手写list
就是链表啊，单链表双链表直接上代码了。
具体可以康康窝以前的分享。
单链表：

#include<bits/stdc++.h>
using namespace std;
const int N = 100010;
int m, e[N], ne[N], idx, head;
void init()
{
    head = -1;
    idx = 0;
}
void add_to_head(int x)
{
    e[idx] = x, ne[idx] = head, head = idx ++;
}
void add(int k, int x)
{
    e[idx] = x, ne[idx] = ne[k - 1], ne[k - 1] = idx ++;
}
void del(int k)
{
    if(!k) head = ne[head];
    ne[k - 1] = ne[ne[k - 1]];
}
void print_list()
{
    for(int i = head; i != -1; i = ne[i]) cout << e[i] << ' ';
    cout << endl;
}
int main()
{
    cin >> m;
    init();
    while(m --)
    {
        char op;
        cin >> op;
        int k, x;
        if(op == 'H')
        {
            cin >> x;
            add_to_head(x);
        }
        else if(op == 'I')
        {
            cin >> k >> x;
            add(k, x);
        }
        else if(op == 'D')
        {
            cin >> k;
            del(k);
        }
        else if(op == 'P')
        {
            print_list();
        }
    }
    return 0;
}
双链表：
这个稍微麻烦那么亿点点。

#include<bits/stdc++.h>
using namespace std;
const int N = 100010;
int m, e[N], l[N], r[N], idx;
void init()
{
    l[1] = 0, r[0] = 1;
    idx = 2;
}
void insert(int a, int x)
{
    e[idx] = x;
    l[idx] = a, r[idx] = r[a];
    l[r[a]] = idx, r[a] = idx ++;
}
void remove(int a)
{
    l[r[a + 1]] = l[a + 1];
    r[l[a + 1]] = r[a + 1];
}
void print_list()
{
    for(int i = r[0]; i != 1; i = r[i] ) cout << e[i] << ' ';
    cout << endl;
}
void add_to_begin(int x)
{
    insert(0, x);
}
void add_to_end(int x)
{
    insert(l[1], x);
}
void add_left(int k, int x)
{
    insert(l[k + 1], x);
}
void add_right(int k, int x)
{
    insert(k + 1, x);
}
int main()
{
    cin >> m;
    init();
    while(m --)
    {
        string op;
        cin >> op;
        int k, x;
        if(op == "L")
        {
            cin >> x;
            add_to_begin(x);
        }
        else if(op == "R")
        {
            cin >> x;
            add_to_end(x);
        }
        else if(op == "D")
        {
            cin >> k;
            remove(k);
        }
        else if(op == "IL")
        {
            cin >> k >> x;
            add_left(k, x);
        }
        else{
            cin >> k >> x;
            add_right(k, x);
        }
    }
    print_list();
    return 0;
}
手写stack
模拟栈的话见我的分享……
这里指给代码了。

#include<iostream>
using namespace std;
const int N = 100010;
int n, tt, stk[N];
void push(int x)
{
    stk[ ++ tt] = x;
}
void pop()
{
    tt --;
}
bool empty()
{
    return (tt ? true : false);
}
int top()
{
    return stk[tt];
}
int main()
{
    cin >> n;
    while(n --)
    {
        string op;
        cin >> op;
        int x;
        if(op == "push")
        {
            cin >> x;
            push(x);
        }
        else if(op == "pop") pop();
        else if(op == "empty") cout << (empty() ? "NO" : "YES") << endl;
        else cout << top() << endl;
    }
    return 0;
}
手写queue
#include<bits/stdc++.h>
using namespace std;
const int N = 100010;
int n, q[N], hh, tt = -1;
void push(int x)
{
    q[ ++ tt] = x;
}
void pop()
{
    hh ++;
}
bool empty()
{
    return (hh <= tt ? true : false);
}
int front()
{
    return q[hh];
}
int main()
{
    cin >> n;
    while(n --)
    {
        string op;
        cin >> op;
        int x;
        if(op == "push")
        {
            cin >> x;
            push(x);
        }
        else if(op == "pop") pop();
        else if(op == "empty") cout << (empty() ? "NO" : "YES") << endl;
        else cout << front() << endl;
    }
}
手写priority_queue
手写堆呗。

#include<bits/stdc++.h>
using namespace std;
const int N = 1000010;
int h[N],ph[N],hp[N],cnt,m;
void head_swap(int a, int b)
{
    swap(ph[hp[a]], ph[hp[b]]);
    swap(hp[a],hp[b]);
    swap(h[a],h[b]);
}
void down(int u)
{
    int t=u;
    if(u * 2 <= cnt && h[u*2]<h[t]) t=u*2;
    if(u*2 + 1<= cnt && h[u * 2 + 1] < h[t]) t=u*2 + 1;
    if(u != t)
    {
        head_swap(u,t);
        down(t);
    }
}
void up(int u)
{
    while(u/2 && h[u] < h[u/2])
    {
        head_swap(u,u/2);
        u >>= 1;
    }
}
void insert(int x)
{
    cnt ++ ;
    m ++ ;
    ph[m] = cnt, hp[cnt] = m;
    h[cnt] = x;
    up(cnt);
}
int top()
{
    return h[1];
}
void erase_top()
{
    head_swap(1, cnt);
    cnt -- ;
    down(1);
}
void erase(int k)
{
    k = ph[k];
    head_swap(k, cnt);
    cnt -- ;
    up(k);
    down(k);
}
void change(int k, int x)
{
    k = ph[k];
    h[k] = x;
    up(k);
    down(k);    
}
int main()
{
    int n;
    cin >> n;
    while(n--)
    {
        char op[5];
        int k,x;
        cin >> op;
        if(!strcmp(op, "I"))
        {
            cin >> x;
            insert(x);
        }
        else if(!strcmp(op, "PM"))
        {
            cout << top() << endl;
        }
        else if(!strcmp(op, "DM"))
        {
            erase_top();
        }
        else if(!strcmp(op, "D"))
        {
            cin >> k;
            erase(k);
        }
        else{
            cin >> k >> x;
            change(k, x);
        }
    }
    return 0;
}
好像也就那么100行……

手写unordered系列（功能不全）
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
    h[k] = idx ++;
}

bool find(int x)
{
    int k = (x % N + N) % N;
    for(int i = h[k]; i != -1; i = ne[i])
    {
        if(e[i] == x)
        {
            return true;
        }
    }
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
手写set（功能及其不全）
#include<bits/stdc++.h>
using namespace std;
const int N = 1000010;
int n, son[N][26], cnt[N], idx;
char str[N];
void add(char *str)
{
    int p = 0;
    for(int i = 0; str[i]; i ++)
    {
        int u = str[i] - 'a';
        if(!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++;
}
int count(char *str)
{
    int p = 0;
    for(int i = 0; str[i] ; i ++)
    {
        int u = str[i] - 'a';
        if(!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}
int main()
{
    cin >> n;
    while(n --)
    {
        char op[2];
        scanf("%s%s", op, str);
        if(*op == 'I') add(str);
        else cout << count(str) << endl;
    }
    return 0;
}
好像以后可以写成AC自动机……

以下为大佬们手写的STL：
平衡树：
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010, INF = 1e8;

int n;
struct Node
{
    int l, r;
    int key, val;
    int cnt, size;
}tr[N];

int root, idx;

void pushup(int p)
{
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
}

int get_node(int key)
{
    tr[ ++ idx].key = key;
    tr[idx].val = rand();
    tr[idx].cnt = tr[idx].size = 1;
    return idx;
}

void zig(int &p)    // 右旋
{
    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
    pushup(tr[p].r), pushup(p);
}

void zag(int &p)    // 左旋
{
    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
    pushup(tr[p].l), pushup(p);
}

void build()
{
    get_node(-INF), get_node(INF);
    root = 1, tr[1].r = 2;
    pushup(root);

    if (tr[1].val < tr[2].val) zag(root);
}


void insert(int &p, int key)
{
    if (!p) p = get_node(key);
    else if (tr[p].key == key) tr[p].cnt ++ ;
    else if (tr[p].key > key)
    {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val) zig(p);
    }
    else
    {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val) zag(p);
    }
    pushup(p);
}

void remove(int &p, int key)
{
    if (!p) return;
    if (tr[p].key == key)
    {
        if (tr[p].cnt > 1) tr[p].cnt -- ;
        else if (tr[p].l || tr[p].r)
        {
            if (!tr[p].r || tr[tr[p].l].val > tr[tr[p].r].val)
            {
                zig(p);
                remove(tr[p].r, key);
            }
            else
            {
                zag(p);
                remove(tr[p].l, key);
            }
        }
        else p = 0;
    }
    else if (tr[p].key > key) remove(tr[p].l, key);
    else remove(tr[p].r, key);

    pushup(p);
}

int get_rank_by_key(int p, int key)    // 通过数值找排名
{
    if (!p) return 0;   // 本题中不会发生此情况
    if (tr[p].key == key) return tr[tr[p].l].size + 1;
    if (tr[p].key > key) return get_rank_by_key(tr[p].l, key);
    return tr[tr[p].l].size + tr[p].cnt + get_rank_by_key(tr[p].r, key);
}

int get_key_by_rank(int p, int rank)   // 通过排名找数值
{
    if (!p) return INF;     // 本题中不会发生此情况
    if (tr[tr[p].l].size >= rank) return get_key_by_rank(tr[p].l, rank);
    if (tr[tr[p].l].size + tr[p].cnt >= rank) return tr[p].key;
    return get_key_by_rank(tr[p].r, rank - tr[tr[p].l].size - tr[p].cnt);
}

int get_prev(int p, int key)   // 找到严格小于key的最大数
{
    if (!p) return -INF;
    if (tr[p].key >= key) return get_prev(tr[p].l, key);
    return max(tr[p].key, get_prev(tr[p].r, key));
}

int get_next(int p, int key)    // 找到严格大于key的最小数
{
    if (!p) return INF;
    if (tr[p].key <= key) return get_next(tr[p].r, key);
    return min(tr[p].key, get_next(tr[p].l, key));
}

int main()
{
    build();

    scanf("%d", &n);
    while (n -- )
    {
        int opt, x;
        scanf("%d%d", &opt, &x);
        if (opt == 1) insert(root, x);
        else if (opt == 2) remove(root, x);
        else if (opt == 3) printf("%d\n", get_rank_by_key(root, x) - 1);
        else if (opt == 4) printf("%d\n", get_key_by_rank(root, x + 1));
        else if (opt == 5) printf("%d\n", get_prev(root, x));
        else printf("%d\n", get_next(root, x));
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/168876/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
线段树：
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 200010;

int m, p;
struct Node
{
    int l, r;
    int v;  // 区间[l, r]中的最大值
}tr[N * 4];

void pushup(int u)  // 由子节点的信息，来计算父节点的信息
{
    tr[u].v = max(tr[u << 1].v, tr[u << 1 | 1].v);
}

void build(int u, int l, int r)
{
    tr[u] = {l, r};
    if (l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
}

int query(int u, int l, int r)
{
    if (tr[u].l >= l && tr[u].r <= r) return tr[u].v;   // 树中节点，已经被完全包含在[l, r]中了

    int mid = tr[u].l + tr[u].r >> 1;
    int v = 0;
    if (l <= mid) v = query(u << 1, l, r);
    if (r > mid) v = max(v, query(u << 1 | 1, l, r));

    return v;
}

void modify(int u, int x, int v)
{
    if (tr[u].l == x && tr[u].r == x) tr[u].v = v;
    else
    {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid) modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}


int main()
{
    int n = 0, last = 0;
    scanf("%d%d", &m, &p);
    build(1, 1, m);

    int x;
    char op[2];
    while (m -- )
    {
        scanf("%s%d", op, &x);
        if (*op == 'Q')
        {
            last = query(1, n - x + 1, n);
            printf("%d\n", last);
        }
        else
        {
            modify(1, n + 1, (last + x) % p);
            n ++ ;
        }
    }

    return 0;
}

作者：yxc
链接：https://www.acwing.com/activity/content/code/content/167554/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
自己手写的树状数组(注意我不是大佬，只是看的y总的代码后手写的一个！！所以这里大佬指的是y总：
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
using namespace std;
int tr[2000010], n, m;
int lowbit(int x)
{
    return x & -x;
}
void add(int x, int c)
{
    for(int i = x; i <= n; i += lowbit(i)) tr[i] += c; 
}
int sum(int x)
{
    int res=  0;
    for(int i = x; i ; i -= lowbit(i)) res += tr[i];
    return res;
}
int main()
{
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; i ++)
    {
        int a;
        scanf("%d", &a);
        add(i, a);
    }
    for(int i = 1; i <= m; i ++)
    {
        int op, x, k;
        scanf("%d%d%d", &op, &x, &k);
        if(op == 1) add(x, k);
        if(op == 2) cout << sum(k) - sum(x - 1) << endl;
    }
    return 0;
}
P云大佬手写的betset（bint）：
#include <bits/stdc++.h>
using namespace std;
struct bint:vector<int>
{
    void format();
    bint(int n)
    {
        do push_back(n % 10), n /= 10; while (n);
    }
    bint(long long n)
    {
        do push_back(n % 10), n /= 10; while (n);
    }
    bint(string s)
    {
        for (int i = s.size() - 1; i >= 0; i --) push_back(s[i] - '0');
    }
    bint()
    {

    }
};
istream& operator>>(istream& in, bint& n);
ostream& operator<<(ostream& out, bint n);
bool operator<(bint a, bint b);
bool operator<=(bint a, bint b);
bool operator>(bint a, bint b);
bool operator>=(bint a, bint b);
bool operator==(bint a, bint b);
bool operator!=(bint a, bint b);
bint operator+(bint a, bint b);
bint operator-(bint a, bint b);
bint operator*(bint a, bint b);
bint operator/(bint a, bint b);
bint operator%(bint a, bint b);
template<typename T>
bint operator*(bint a, T b);
template<typename T>
bint operator/(bint a, T b);
template<typename T>
bint operator%(bint a, T b);
template<typename T>
bint operator*(T a, bint b);
bint divmode(bint& a, bint b);
template<typename T>
bint divmode(bint a, T b, T& r);
template<typename T>
void operator+=(T& a, T b);
template<typename T>
void operator-=(T& a, T b);
template<typename T>
void operator*=(T& a, T b);
template<typename T>
void operator/=(T& a, T b);
template<typename T>
void operator%=(T& a, T b);
void operator--(bint& a);
void operator++(bint& a);
void bint::format()
{
    while(size() > 1 && back() == 0) pop_back();
}
istream& operator>>(istream& in, bint& n)
{
    string s;
    in >> s;
    n.clear();
    for (int i = s.size() - 1; i >= 0; i --) n.push_back(s[i] - '0');
    return in;
}
ostream& operator<<(ostream& out, bint n)
{
    for (int i = n.size() - 1; i >= 0; i --) out << n[i];
    return out;
}
bool operator<(bint a, bint b)
{
    if (a.size() != b.size()) return a.size() < b.size();
    for (int i = a.size() - 1; i >= 0; i --)
        if (a[i] != b[i])
            return a[i] < b[i];
    return false;
}
bool operator<=(bint a, bint b)
{
    return a < b || a == b;
}
bool operator>(bint a, bint b)
{
    return !(a <= b);
}
bool operator>=(bint a, bint b)
{
    return !(a < b);
}
bool operator==(bint a, bint b)
{
    if (a.size() != b.size()) return false;
    for (int i = a.size() - 1; i >= 0; i --)
        if (a[i] != b[i])
            return false;
    return true;
}
bool operator!=(bint a, bint b)
{
    return !(a == b);
}
bint operator+(bint a, bint b)
{
    int t = 0;
    bint c;
    for (int i = 0; i < a.size() || i < b.size(); i ++)
    {
        if (i < a.size()) t += a[i];
        if (i < b.size()) t += b[i];
        c.push_back(t % 10);
        t /= 10;
    }
    if (t) c.push_back(t);
    return c;
}
bint operator-(bint a, bint b)
{
    if (b > a)
    {
        cerr << "Error occurs at BigInteger operator-(BigInteger, BigInteger)" << endl;
        cerr << "A negative result is produced" << endl;
        return a;
    }
    int t = 0;
    bint c;
    for (int i = 0; i < a.size() ; i ++)
    {
        t += a[i];
        if (i < b.size()) t -= b[i];
        if (t < 0) c.push_back(t + 10), t = -1;
        else c.push_back(t), t = 0;
    }
    c.format();
    return c;
}
bint operator*(bint a, bint b)
{
    bint c;
    c.assign(a.size() + b.size() - 1, 0);
    for(int i = 0; i < a.size(); i ++)
        for(int j = 0; j < b.size(); j ++)
            c[i + j] += a[i] * b[j];
    for (int i = 0; i < c.size() - 1; i ++)
        if (c[i] >= 10)
            c[i + 1] += c[i] / 10, c[i] %= 10;
    if (c[c.size() - 1] >= 10) c.push_back(c[c.size() - 1] / 10), c[c.size() - 2] %= 10;
    c.format();
    return c;
}
bint operator/(bint a, bint b)
{
    return divmode(a, b);
}
bint operator%(bint a, bint b)
{
    divmode(a, b);
    return a;
}
template<typename T>
bint operator*(bint a, T b)
{
    bint c;
    T t = 0;
    for (int i = 0; i < a.size() || t; i ++)
    {
        if (i < a.size()) t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }
    c.format();
    return c;
}
template<typename T>
bint operator*(T a, bint b)
{
    return b * a;
}
template<typename T>
bint operator/(bint a, T b)
{
    T r = 0;
    return divmode(a, b, r);
}
template<typename T>
T operator%(bint a, T b)
{
    T r;
    divmode(a, b, r);
    return r;
}
bint divmode(bint& a, bint b)
{
    if (b == 0)
    {
        cerr << "Error occurs at BigInteger operator/(BigInteger, BigInteger)" << endl;
        cerr << "Divided by zero" << endl;
        return a;
    }
    bint c, d, e;
    for (int i = a.size() - b.size(); a >= b; i --)
    {
        d.clear(), d.assign(i + 1, 0), d.back() = 1;
        int l = 0, r = 9, m;
        while (l < r)
        {
            m = l + r + 1 >> 1;
            e = b * d * m;
            if (e <= a) l = m;
            else r = m - 1;
        }
        a -= b * d * l, c += d * l;
    }
    return c;
}
template<typename T>
bint divmode(bint a, T b, T& r)
{
    bint c;
    r = 0;
    for (int i = a.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + a[i];
        c.push_back(r / b);
        r %= b;
    }
    reverse(c.begin(), c.end());
    c.format();
    return c;
}
template<typename T>
void operator+=(T& a, T b)
{
    a = a + b;
}
template<typename T>
void operator-=(T& a, T b)
{
    a = a - b;
}
template<typename T>
void operator*=(T& a, T b)
{
    a = a * b;
}
template<typename T>
void operator/=(T& a, T b)
{
    a = a / b;
}
template<typename T>
void operator%=(T& a, T b)
{
    a = a % b;
}
void operator--(bint& a)
{
    a -= bint(1);
}
void operator++(bint& a)
{
    a += bint(1);
}
int main()
{
    bint a, b;
    cin >> a >> b;
    cout << a + b;
    return 0;
}

作者：P云
链接：https://www.acwing.com/blog/content/2186/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
STO……

我的STL（chtSTL）
1、arg。
就是集合的意思。
其实就是并查集啦！
基本操作：

find();//查找这个点的祖宗节点
merge();//合并a和b所在的集合
New();//初始化1个点
init();//初始化所有点
smae();//返回两个点是否在同一个集合。
代码：

#include<bits/stdc++.h>
using namespace std;
int p[10010], n;
int find(int x)
{
    if(p[x] != x) p[x] = find(p[x]);
    return p[x];
}
void New(int k)
{
    p[k] = k;
}
void init()
{
    for(int i = 1; i <= n; i ++) p[i] = i;
}
void merge(int a, int b)
{
    int pa = find(a), pb = find(b);
    a = pa, b = pb;
    p[a] = b;
}
bool query(int a, int b)
{
    if(find(a) == find(b)) return true;
    return false;
}
int main()
{
    int n;
    cin >> n;
    return 0;
}
2、pos
坐标系。
支持各种坐标系操作。

empty();//是否为空
clear();//清空
add();//添加
ordered();//序列化
find();//查找
erase();//删除
Swap();//交换
print();//输出
具体代码：

#include<bits/stdc++.h>
using namespace std;
int n, m;
pair<int, int> pos[1010][1010];
void add(int x, int y, int a, int b)
{
    pos[x][y].first = a;
    pos[x][y].second = b;
}
void print()
{
    for(int i = 1; i <= n; i ++, cout << endl)
        for(int j = 1; j <= m; j ++)
            cout << "x=" << pos[i][j].first << ";y=" << pos[i][j].second << ' ';
}
void ordered()
{
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j ++)
        {
            if(pos[i][j].first > n || pos[i][j].second > m)
            {
                cout << "错误的请求！" << endl;
                return;
            }
            if(pos[i][j].first <= 0 || pos[i][j].second <= 0) continue;
            int sx = pos[i][j].first, sy = pos[i][j].second;
            if(sx != i || sy != j)swap(pos[sx][sy], pos[i][j]);
        }
}
void erase(int x, int y)
{
    pos[x][y].first = 0;
    pos[x][y].second = 0;
}
pair<int, int> find(int x, int y)
{
    for(int i = 1; i <= n; i ++)
        for(int j = 1; j <= m; j ++)
            if(x == pos[i][j].first && y == pos[i][j].second)
            {
                pair<int, int> ans = {i, j};
                return ans;
            }
}
int Swap(int x1, int y1, int x2, int y2)
{
    swap(pos[x1][y1], pos[x2][y2]);
}
bool empty()
{
    for(int i = 1; i <= n; i ++)
        for(int j = 1; j <= m; j ++)
            if(pos[i][j].first != 0 || pos[i][j].second != 0)  
                return false;
    return true;
}
void clear()
{
    for(int i = 1; i <= n;i  ++)
        for(int j = 1; j <= m; j++)
            pos[i][j] = {0, 0};
}
int main()
{
    cin >> n >> m;
    return 0;
}
好了今天的分享就到这里了。
本文耗时：4h
1482行，制作不易。
望3连！
这是我的全部分享
bye~

作者：cht
链接：https://www.acwing.com/blog/content/3122/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

C++ STL 简介
作者：    zmj2008 ,  2020-03-07 10:49:13 ,  阅读 593

9


12
C++ STL 简介
unique, 将数组中重复的元素放到了最后
sort, 将数组中的元素排序
vector, 变长数组，倍增的思想
    size()  返回元素个数
    empty()  返回是否为空
    clear()  清空
    front()/back()
    push_back()/pop_back()
    begin()/end()
    []
    支持比较运算，按字典序
pair<int, int>
    first, 第一个元素
    second, 第二个元素
    支持比较运算，以first为第一关键字，以second为第二关键字（字典序）
string，字符串
    size()/length()  返回字符串长度
    empty()
    clear()
    substr(起始下标，(子串长度))  返回子串
    c_str()  返回字符串所在字符数组的起始地址
queue, 队列
    size()
    empty()
    push()  向队尾插入一个元素
    front()  返回队头元素
    back()  返回队尾元素
    pop()  弹出队头元素
priority_queue, 优先队列，默认是大根堆
    push()  插入一个元素
    top()  返回堆顶元素
    pop()  弹出堆顶元素
    定义成小根堆的方式：priority_queue<int, vector<int>, greater<int>> q;
stack, 栈
    size()
    empty()
    push()  向栈顶插入一个元素
    top()  返回栈顶元素
    pop()  弹出栈顶元素
deque, 双端队列
    size()
    empty()
    clear()
    front()/back()
    push_back()/pop_back()
    push_front()/pop_front()
    begin()/end()
    []
set, map, multiset, multimap, 基于平衡二叉树（红黑树），动态维护有序序列
    size()
    empty()
    clear()
    begin()/end()
    ++, -- 返回前驱和后继，时间复杂度 O(logn)

    set/multiset
        insert()  插入一个数
        find()  查找一个数
        count()  返回某一个数的个数
        erase()
            (1) 输入是一个数x，删除所有x   O(k + logn)
            (2) 输入一个迭代器，删除这个迭代器
        lower_bound()/upper_bound()
            lower_bound(x)  返回大于等于x的最小的数的迭代器
            upper_bound(x)  返回大于x的最小的数的迭代器
    map/multimap
        insert()  插入的数是一个pair
        erase()  输入的参数是pair或者迭代器
        find()
        []  注意multimap不支持此操作。 时间复杂度是 O(logn)
        lower_bound()/upper_bound()
unordered_set, unordered_map, unordered_multiset, unordered_multimap, 哈希表
    和上面类似，增删改查的时间复杂度是 O(1)
    不支持 lower_bound()/upper_bound()， 迭代器的++，--
bitset, 圧位
    bitset<10000> s;
    ~, &, |, ^
    >>, <<
    ==, !=
    []

    count()  返回有多少个1

    any()  判断是否至少有一个1
    none()  判断是否全为0

    set()  把所有位置成1
    set(k, v)  将第k位变成v
    reset()  把所有位变成0
    flip()  等价于~
    flip(k) 把第k位取反

作者：zmj2008
链接：https://www.acwing.com/blog/content/1846/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


https://www.acwing.com/blog/content/844/



https://www.acwing.com/blog/content/3684/

https://www.acwing.com/blog/content/3538/

https://www.acwing.com/blog/content/1846/

https://www.acwing.com/blog/content/3122/
https://www.acwing.com/blog/content/3153/
https://www.acwing.com/blog/content/3801/

```