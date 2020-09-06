#include<iostream>
#include<cstdio>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<set>
#include<map>
#include<vector>
#include<ctime>
#include<queue>
#include<iomanip>
#define ll long long
#define N 300005

using namespace std;
inline int Get() {int x=0,f=1;char ch=getchar();while(ch<'0'||ch>'9') {if(ch=='-') f=-1;ch=getchar();}while('0'<=ch&&ch<='9') {x=(x<<1)+(x<<3)+ch-'0';ch=getchar();}return x*f;}

ll n,m,q;
int cnt;
int nxt[N];
const int lx=1,rx=N<<1;
int rt[N];
int tag[N*70],ls[N*70],rs[N*70],mysize[N*70];
ll id[N*70];
void update(int v) {mysize[v]=mysize[ls[v]]+mysize[rs[v]];}

void build(int &v,int l,int r) {
	if(!v) v=++cnt;
	mysize[v]=r-l+1;
	if(l==r) id[v]=l;
	tag[v]=1;
}

void pre(int &v,int l,int r,int lx,int rx) {
	if(!v) v=++cnt;
	if(l>rx||r<lx) return ;
	if(l<=lx&&rx<=r) {build(v,lx,rx);return ;}
	int mid=lx+rx>>1;
	pre(ls[v],l,r,lx,mid);
	pre(rs[v],l,r,mid+1,rx);
	update(v);
}

void Find(int v,int lx,int rx,int k,ll &pos,ll &g) {
	if(lx==rx) {
		pos=lx;
		g=id[v];
		return ;
	}
	int mid=lx+rx>>1;
	if(tag[v]) {
		build(ls[v],lx,mid);
		build(rs[v],mid+1,rx);
		tag[v]=0;
	}
	if(k>mysize[ls[v]]) {
		Find(rs[v],mid+1,rx,k-mysize[ls[v]],pos,g);
	} else {
		Find(ls[v],lx,mid,k,pos,g);
	}
}

void Delete(int v,int lx,int rx,int pos) {
	mysize[v]--;
	if(lx==rx) return ;
	int mid=lx+rx>>1;
	if(tag[v]) {
		build(ls[v],lx,mid);
		build(rs[v],mid+1,rx);
		tag[v]=0;
	}
	if(pos<=mid) Delete(ls[v],lx,mid,pos);
	else Delete(rs[v],mid+1,rx,pos);
}

void Insert(int &v,int lx,int rx,int pos,ll id) {
	if(!v) v=++cnt;
	mysize[v]++;
	if(lx==rx) {
		::id[v]=id;
		return ;
	}
	int mid=lx+rx>>1;
	if(tag[v]) {
		build(ls[v],lx,mid);
		build(rs[v],mid+1,rx);
		tag[v]=0;
	}
	if(pos<=mid) Insert(ls[v],lx,mid,pos,id);
	else Insert(rs[v],mid+1,rx,pos,id);
}

ll pos,g;

void Get_out(ll &nx,ll &ny,ll x,ll y) {
	if(y<m) {
		Find(rt[x],lx,rx,y,pos,g);
		if(pos<=m-1) {
			nx=x;
			ny=pos;
		} else {
			nx=(g-1)/m+1;
			ny=(g-1)%m+1;
		}
		Delete(rt[x],lx,rx,pos);
	} else {
		Find(rt[n+1],lx,rx,x,pos,g);
		if(pos<=n) {
			nx=pos;
			ny=m;
		} else {
			nx=(g-1)/m+1;
			ny=(g-1)%m+1;
		}
		Delete(rt[n+1],lx,rx,pos);
	}
}

int main() {
	n=Get(),m=Get(),q=Get();
	if(m>1) {
		for(int i=1;i<=n;i++) {
			nxt[i]=m;
			pre(rt[i],1,m-1,lx,rx);
		}
	}
	pre(rt[n+1],1,n,lx,rx);
	nxt[n+1]=n+1;
	int x,y;
	ll sx,sy;
	ll tx,ty;
	while(q--) {
		x=Get(),y=Get();
		if(y<m) {
			Get_out(sx,sy,x,y);
			Get_out(tx,ty,x,m);
			Insert(rt[x],lx,rx,nxt[x],(tx-1)*m+ty);
			Insert(rt[n+1],lx,rx,nxt[n+1],(sx-1)*m+sy);
			nxt[x]++,nxt[n+1]++;
		} else {
			Get_out(sx,sy,x,y);
			Insert(rt[n+1],lx,rx,nxt[n+1],(sx-1)*m+sy);
			nxt[n+1]++;
		}
		cout<<(sx-1)*m+sy<<"\n";
	}
	return 0;
}
