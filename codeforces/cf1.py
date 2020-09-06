import sys

def calc(a,b):
    if a>b:
        c=(a-b)//10 +  (0 if((a-b)%10==0) else 1)
    elif a==b:
        c=0
    else:
        c=(b-a)//10 + (0 if((b-a)%10==0) else 1)
    return c

t=int(input())

for x in range(0,t):
    a,b=map(int, input().split())
    #print(a,b)
    c=calc(a,b)
    print(c)