//arbitrary precision PEMDAS using python is simply easy!

a=int(input())
b=int(input())

print(a+b) //arbitrary precision add

print(a-b) //arbitrary precision sub

print(a*b) //arbitrary precision multiply


print(a//b) //arbitrary precision division, note for large integers needs to use // not /

print(a%b) // modulo for remainder

// in python integer type is unlimited (as memory allows); 
// however python float is limited, a/10 will return a float which has limits sys.float_info.max
// and it may overflow; double // division should be used to get the right integer division done
//https://stackoverflow.com/questions/27946595/how-to-manage-division-of-huge-numbers-in-python/27946741#:~:text=Unlike%20floats%2C%20int%20values%20can,3000%20%2F%2F%2010%20123023192216111717693155881327...


//python can run faster in many online judges now since it's optimized

//pemdas

print(a**b) // a to power of b

print(a^b) // a xor b