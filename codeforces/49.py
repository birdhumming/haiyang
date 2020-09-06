# 1 to 49 select 3 numbers, number of ways for one of them is the average of
#the other two 576

#for i in range (1, 11):
	#print(i)

import itertools 
  
def findsubsets(s, n): 
    return list(itertools.combinations(s, n)) 
  
# Driver Code 
#s = {1, 2, 3} 
s = list(range(1,50))
n = 3
  
#print(findsubsets(s, n)) 
a=findsubsets(s, n)

#print(a)

count=0
for x in a:
	if (x[0]+x[2])==2*x[1]: 
		print(x)
		count = count+1
print(count)
#	print(x)
#	if (sum(x)==11) print(list(x))

