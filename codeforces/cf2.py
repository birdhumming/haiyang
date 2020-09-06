import sys


def calc(a,b,x,y,n):
    if (a-x+b-y) <= n:
        cmin=x*y
    else:
        cmin=1e19

        #if(a>b):
        #    mid=(a-b+n)/2
        #else:
        #    mid=(b-a+n)/2
        
        #if ((a-mid)>=x and (b-n+mid)>=y):
        #    cmin=(a-mid)*(b-n+mid)

        for i in range(0,n+1):
            if ((a-i)>=x and (b-n+i)>=y):
                
                cmin=min(cmin, (a-i)*(b-n+i))
                #print(cmin)
            
                


    return cmin

t=int(input())

for x in range(0,t):
    a,b,x,y,n=map(int, input().split())
    #print(a,b)
    c=calc(a,b,x,y,n)
    print(c)