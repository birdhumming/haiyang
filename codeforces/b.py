

def calc(a,b,x,y,n):
    t=min(a-x, n)
    a-=t
    n-=t
    t=min(b-y, n)
    b-=t
    return a*b

t=int(input())

for x in range(0,t):
    a,b,x,y,n=map(int, input().split())
    #print(a,b)
    c=min(calc(a,b,x,y,n),calc(b,a,y,x,n))
    print(c)