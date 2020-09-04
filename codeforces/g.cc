
 1 /*1966    Accepted    300K    63MS    C++    2366B    2012-06-14 22:09:21*/
 2 #include <cstdio>
 3 #include <cstring>
 4 #include <cmath>
 5 #include <iostream>
 6 #include <algorithm>
 7 #include <vector>
 8 using namespace std;
 9 
10 #define mpair make_pair
11 #define pii pair<int,int>
12 #define MM(a,b) memset(a,b,sizeof(a));
13 typedef long long lld;
14 typedef unsigned long long u64;
15 template<class T> bool up_max(T& a,const T& b){return b>a? a=b,1 : 0;}
16 template<class T> bool up_min(T& a,const T& b){return b<a? a=b,1 : 0;}
17 #define maxn 110
18 const int inf= 2100000000;
19 
20 int n,m, ST=0, ED, NV;
21 int map[maxn][maxn];
22 int g[maxn][maxn];
23 
24 int pre[maxn], que[maxn];
25 bool vis[maxn];
26 bool bfs(){
27     MM( vis, 0 );
28     int head= 0, tail= 0;
29     que[tail++]= ST;
30     vis[ST]= 1;
31     while( head<tail ){
32         int u= que[head++];
33         for(int v=0;v<NV;++v){
34             if( g[u][v]>0 && !vis[v] ){
35                 pre[v]= u;
36                 if( v==ED ) return 1;
37                 que[tail++]= v;
38                 vis[v]= 1;
39             }
40         }
41     }
42     return 0;
43 }
44 
45 int Edmond_karp(){
46     int ret= 0;
47     while( bfs() ){
48         int t= inf;
49         for(int i=ED;i!=ST;i=pre[i]){
50             up_min( t, g[pre[i]][i] );
51         }
52         ret+= t;
53         for(int i=ED;i!=ST;i=pre[i]){
54             g[pre[i]][i]-= t;
55             g[i][pre[i]]+= t;
56         }
57     }
58     return ret;
59 }
60 
61 int main()
62 {
63     //freopen("poj1966.in","r",stdin);
64     while( cin>>n>>m ){
65         NV= n+n;
66         MM( map, 0 );
67         for(int i=0;i<n;++i) map[i][i+n]= 1;
68         /*while( ch= getchar(), ch!='\n' ){  /// Wrong!!!
69             if( ch == '(' ){
70                 int x,y;
71                 scanf("%d,%d", &x, &y);
72                 map[x+n][y]= inf;
73                 map[y+n][x]= inf;
74             }
75         }*/
76         while(m--){
77             int x,y;
78             scanf(" (%d,%d)", &x, &y); /// my god;~!!!
79             map[x+n][y]= inf;
80             map[y+n][x]= inf;  ///!!!
81         }
82         if( n<2 ){ printf("%d\n", n); continue; }
83 
84         int ans= inf;
85         for(ST=0; ST<n; ++ST){
86             for(ED=ST+1; ED<n; ++ED){
87                 for(int k=0;k<NV;++k)for(int k1=0;k1<NV;++k1)g[k][k1]= map[k][k1];
88                 g[ST][ST+n]= inf;
89                 int t= Edmond_karp();
90                 up_min( ans, t );
91             }
92         }
93         if( ans==inf ) ans= n;
94         cout<< ans <<endl;
95     }
96 }
