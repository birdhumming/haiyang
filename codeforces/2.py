#for i in range (1, 11):
	#print(i)

import itertools 
  
def findsubsets(s, n): 
    return list(itertools.combinations(s, n)) 
  
# Driver Code 
#s = {1, 2, 3} 
s = list(range(1,11))
n = 2
  
#print(findsubsets(s, n)) 
a=findsubsets(s, n)

for x in a:
	if (sum(x)==11): 	
		print(list(x))


n = 3

#print(findsubsets(s, n))
b=findsubsets(s, n)

for x in b:
        if (sum(x)==11):        
                print(list(x))

