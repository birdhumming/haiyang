#include<cstdio>
 2 #include<cstdlib>
 3 #include<cstring>
 4 #include<iostream>
 5 #include<algorithm>
 6 using namespace std;
 7 #define Maxn 2010
 8 #define Maxm 1000010
 9 
10 int n,m;
11 int first[Maxn];
12 bool mark[Maxn];
13 int s[Maxn],sl;
14 
15 struct node
16 {
17     int x,y,next;
18 }t[Maxm];int len;
19 
20 void ins(int x,int y)
21 {
22     t[++len].x=x;t[len].y=y;
23     t[len].next=first[x];first[x]=len;
24 }
25 
26 bool dfs(int x)
27 {
28     if(mark[x^1]) return 0;
29     if(mark[x]) return 1;
30     mark[x]=1;
31     s[++sl]=x;
32     for(int i=first[x];i;i=t[i].next)
33     {
34         if(!dfs(t[i].y)) return 0;
35     }
36     return 1;
37 }
38 
39 bool ffind()
40 {
41     memset(mark,0,sizeof(mark));
42     mark[0]=1;
43     if(!dfs(0)) return 0;
44     for(int i=1;i<n;i++)
45     {
46         if(mark[i*2]||mark[i*2+1]) continue;
47         sl=0;
48         if(!dfs(i*2))
49         {
50             while(sl) mark[s[sl--]]=0;
51             if(!dfs(i*2+1)) return 0;
52         }
53     }
54     return 1;
55 }
56 
57 int main()
58 {
59     while(1)
60     {
61         scanf("%d%d",&n,&m);
62         if(n==0&&m==0) break;
63         getchar();len=0;
64         memset(first,0,sizeof(first));
65         bool p;
66         for(int i=1;i<=m;i++)
67         {
68             int x,y,p1,p2;
69             char c;
70             scanf("%d%c",&x,&c);getchar();
71             p1=(c=='h'?0:1);
72             scanf("%d%c",&y,&c);//getchar();
73             p2=(c=='h'?0:1);
74             getchar();
75             if(p1==0&&p2==0) p=1;
76             ins(x*2+p1,y*2+1-p2);//ins(y*2+1-p2,x*2+p1);
77             ins(y*2+p2,x*2+1-p1);//ins(x*2+1-p1,y*2+p2);
78         }
79         if(!ffind()) printf("bad luck\n");
80         else
81         {
82             for(int i=1;i<n;i++)
83             {
84                 if(i!=1) printf(" ");
85                 if(mark[i*2]) printf("%dw",i);
86                 else printf("%dh",i);
87             }
88         }printf("\n");
89     }
90     return 0;
91 }
