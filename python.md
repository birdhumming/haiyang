
lamda is anonymous function

from time import sleep

with open("large", "r") as file:

	for word in file.readline():

		print(f"Checking {word.rstrip()} ...")

		sleep(.1)

rstrip()



indentation is import, indicating scope

no semicolon is needed

f-string=format string print(f"hello, {name}") where name is a variable



def func(n)

python ^ is XOR not exponential as in C

int(input("what's your name?"))



python integer has no  upper bound limit!!! no overflow errors

web programming is much easier in python than C

print("Output: ", end="") to supress the \n newline



for c in s:

	print(c,end="")



command-line arguments



from sys import argv, exit

for arg in argv:

    print(arg)

for i in range(len(argv)): 

	print(argv[i])



names=["emma", "tom", "john"]

if "emma" in names:

	print("found")

	exit(0)

print("not found")

exit(1)



hashtable = dict in python = key-value pair = associative array



if str1 == str2:

	print("same")



x=1

y=2

print (f"x is {x}, y is {y}")

x,y=y,x

print (f"x is {x}, y is {y}")

file I/O

import csv

file=open("phonebook.csv", "a")

with open("phonebook.csv") as file: // no need to close with

name=get_string("Name: ")

number=get_string("Number: ")

write = csv.writer(file)

writer.writerow((name,number))

//file.close()



python has regular expression; C doesn't

import re

if re.search("yes|y", s, re.IGNORECASE):  // "^y(es)?$" or s.lower()



https://www.geeksforgeeks.org/speech-recognition-in-python-using-google-speech-api/



python eval() method

https://www.programiz.com/python-programming/methods/built-in/eval



there is no char in python, only string as str

python map(function, iterable) applies the function to each item of a given iterable and return the results:

# Return double of n

def addition(n):

    return n + n 

# We double all numbers using map()

numbers = (1, 2, 3, 4)

result = map(addition, numbers)

print(list(result))





It can be done with lamda expressions also: lamda is anonymous function



# Double all numbers using map and lambda

numbers = (1, 2, 3, 4)

result = map(lambda x: x + x, numbers)

print(list(result))

# Add two lists using map and lambda

numbers1 = [1, 2, 3]

numbers2 = [4, 5, 6]

result = map(lambda x, y: x + y, numbers1, numbers2)

print(list(result))





# List of strings

l = ['sat', 'bat', 'cat', 'mat']

  

# map() can listify the list of strings individually

test = list(map(list, l))

print(test)

map() can listify the list of strings individually as shown above



A lambda function is a small anonymous function.

A lambda function can take any number of arguments, but can only have one expression.

lambda arguments : expression

The expression is executed and the result is returned:



A lambda function that adds 10 to the number passed in as an argument, and print the result:

x = lambda a : a + 10

print(x(5))

Lambda functions can take any number of arguments:

x = lambda a, b : a * b

print(x(5, 6))

A lambda function that sums argument a, b, and c and print the result:

x = lambda a, b, c : a + b + c

print(x(5, 6, 2))



The power of lambda is better shown when you use them as an anonymous function inside another function.

Say you have a function definition that takes one argument, and that argument will be multiplied with an unknown number:

def myfunc(n):

  return lambda a : a * n

Use that function definition to make a function that always doubles the number you send in:





def myfunc(n):

  return lambda a : a * n

mydoubler = myfunc(2)

print(mydoubler(11))



Or, use the same function definition to make a function that always triples the number you send in:





def myfunc(n):

  return lambda a : a * n

mytripler = myfunc(3)

print(mytripler(11))

Or, use the same function definition to make both functions, in the same program:





def myfunc(n):

  return lambda a : a * n



mydoubler = myfunc(2)

mytripler = myfunc(3)



print(mydoubler(11))

print(mytripler(11))





comments start with #;  can also use multiline string/triple quotes docstring style

"""

This is a comment

written in

more than just one line 

can be "hello work" with double quotes inside docstring

single quotes ok too 'ok single quotes'

"""

print("HELLO World.") #this is a comment

Indentation
Python uses indentation for blocks, instead of curly braces. Both tabs and spaces are supported, but the standard indentation requires standard Python code to use four spaces. For example:

x = 1

if x == 1:

    # indented four spaces

    print("x is 1.")

Variables and Types
You do not need to declare variables before using them, or declare their type. 

Every variable in Python is an object. Python is completely object oriented, and not "statically typed".

Numbers
Python supports two types of numbers - integers and floating point numbers. (It also supports complex numbers, more later).

To define an integer, use the following syntax:

myint = 7

print(myint)

To define a floating point number, you may use one of the following notations:

myfloat = 7.0

print(myfloat)

or

myfloat = float(7) #casting

print(myfloat)

Strings
Strings are defined either with a single quote or a double quotes. The difference between the two is that using double quotes makes it easy to include apostrophes (whereas these would terminate the string if using single quotes).

mystring = 'hello'

print(mystring)

mystring = "hello"

print(mystring)

mystring = "Don't worry about apostrophes" #this is allowed

print(mystring)



Simple operators can be executed on numbers and strings:

one = 1

two = 2

three = one + two #addition

print(three)

hello = "hello"

world = "world"

helloworld = hello + " " + world #concatenation of string

print(helloworld)

Assignments can be done on more than one variable "simultaneously" on the same line like this

a, b = 3, 4

print(a,b)

Mixing operators between numbers and strings is not supported:

# This will not work!

one = 1

two = 2

hello = "hello"

print(one + two + hello)

with errors:

Traceback (most recent call last):

  File "<stdin>", line 6, in <module>

    print(one + two + hello)

TypeError: unsupported operand type(s) for +: 'int' and 'str'

code sample:

# example code with if and ==

mystring = "hello"

myfloat = 10.0

myint = 20



# testing code

if mystring == "hello":

    print("String: %s" % mystring)

if isinstance(myfloat, float) and myfloat == 10.0:

    print("Float: %f" % myfloat)

if isinstance(myint, int) and myint == 20:

    print("Integer: %d" % myint)



Lists [ ]


Lists can contain any type of variable, and they can contain as many variables as you wish. Lists can also be iterated over in a very simple manner. Here is an example of how to build a list.

mylist = []

mylist.append(1)

mylist.append(2)

mylist.append(3)

print(mylist[0]) # prints 1

print(mylist[1]) # prints 2

print(mylist[2]) # prints 3



# prints out 1,2,3

for x in mylist:

    print(x)

Accessing an index which does not exist generates an exception (an error).

mylist = [1,2,3]

print(mylist[10]) 

# gives error below

Traceback (most recent call last):

  File "<stdin>", line 2, in <module>

    print(mylist[10])

IndexError: list index out of range

sample code



numbers = []

strings = []

names = ["John", "Eric", "Jessica"]



# write your code here

numbers.append(1)

numbers.append(2)

numbers.append(3)



strings.append("hello")

strings.append("world")



second_name = names[1]



# this code should write out the filled arrays and the second name in the # names list (Eric).

print(numbers)

print(strings)

print("The second name on the names list is %s" % (second_name,))

print("The second name on the names list is %s %s" % (second_name, strings[1]))





The main difference between a list and an array is the functions that you can perform to them. For example, you can divide an array by 3, and each number in the array will be divided by 3 and the result will be printed if you request it. If you try to divide a list by 3, Python will tell you that it can't be done, and an error will be thrown.

It does take an extra step to use arrays because they have to be declared while lists don't because they are part of Python's syntax, so lists are generally used more often between the two, which works fine most of the time. However, if you're going to perform arithmetic functions to your lists, you should really be using arrays instead. Additionally, arrays will store your data more compactly and efficiently, so if you're storing a large amount of data, you may consider using arrays as well.

Basic Operators
Arithmetic Operators
Just as any other programming languages, the addition, subtraction, multiplication, and division operators can be used with numbers. They follow PEMDAS order.  Another operator available is the modulo (%) operator, which returns the integer remainder of the division. dividend % divisor = remainder.

number = 1 + 2 * 3 / 4.0

print(number)

remainder = 11 % 3

print(remainder)

Using two multiplication symbols makes a power relationship.

squared = 7 ** 2

cubed = 2 ** 3

print(squared)

print(cubed)

Using Operators with Strings
Python supports concatenating strings using the addition operator. Python also supports multiplying strings to form a string with a repeating sequence:

helloworld = "hello" + " " + "world"

print(helloworld)

lotsofhellos = "hello" * 10

print(lotsofhellos)

Using Operators with Lists
Lists can be joined with the addition operators. Just as in strings, Python supports forming new lists with a repeating sequence using the multiplication operator:

even_numbers = [2,4,6,8]

odd_numbers = [1,3,5,7]

all_numbers = odd_numbers + even_numbers

print(all_numbers)

print([1,2,3] * 3)

x = object()

y = object()



"""

The target of this exercise is to create two lists called x_list and y_list, which contain 10 instances of the variables x and y, respectively. You are also required to create a list called big_list, which contains the variables x and y, 10 times each, by concatenating the two lists you have created.

"""



x_list = [x] * 10

y_list = [y] * 10

big_list = x_list + y_list



print("x_list contains %d objects" % len(x_list))

print("y_list contains %d objects" % len(y_list))

print("big_list contains %d objects" % len(big_list))



# testing code

if x_list.count(x) == 10 and y_list.count(y) == 10:

    print("Almost there...")

if big_list.count(x) == 10 and big_list.count(y) == 10:

    print("Great!")

String Formatting


Python uses C-style string formatting to create new, formatted strings. The "%" operator is used to format a set of variables enclosed in a "tuple" (a fixed size list), together with a format string, which contains normal text together with "argument specifiers", special symbols like "%s" and "%d".

Let's say you have a variable called "name" with your user name in it, and you would then like to print(out a greeting to that user.)

To use two or more argument specifiers, use a tuple (parentheses):

# This prints out "Hello, John!"

name = "John"

print("Hello, %s!" % name)

# This prints out "John is 23 years old."

name = "John"

age = 23

print("%s is %d years old." % (name, age))

Any object which is not a string can be formatted using the %s operator as well. The string which returns from the "repr" method of that object is formatted as the string. For example:

# This prints out: A list: [1, 2, 3]

mylist = [1,2,3]

print("A list: %s" % mylist)

Here are some basic argument specifiers you should know:

%s - String (or any object with a string representation, like numbers)

%d - Integers

%f - Floating point numbers

%.<number of digits>f - Floating point numbers with a fixed amount of digits to the right of the dot.

%x/%X - Integers in hex representation (lowercase/uppercase)

data = ("John", "Doe", 53.44)

format_string = "Hello %s %s. Your current balance is $%s."



print(format_string % data)



Basic String Operations


Strings are bits of text. They can be defined as anything between quotes.

As you can see, the first thing you learned was printing a simple sentence. This sentence was stored by Python as a string. However, instead of immediately printing strings out, we will explore the various things you can do to them. You can also use single quotes to assign a string. However, you will face problems if the value to be assigned itself contains single quotes.For example to assign the string in these bracket(single quotes are ' ') you need to use double quotes only like this

astring = "Hello world!"

astring2 = 'Hello world!'

astring = "Hello world!"

print("single quotes are ' '")

print(len(astring))

That prints out 12, because "Hello world!" is 12 characters long, including punctuation and spaces.

astring = "Hello world!"

print(astring.index("o"))

That prints out 4, because the location of the first occurrence of the letter "o" is 4 characters away from the first character. Notice how there are actually two o's in the phrase - this method only recognizes the first.

But why didn't it print out 5? Isn't "o" the fifth character in the string? To make things more simple, Python (and most other programming languages) start things at 0 instead of 1. So the index of "o" is 4.

astring = "Hello world!"

print(astring.count("l"))

For those of you using silly fonts, that is a lowercase L, not a number one. This counts the number of l's in the string. Therefore, it should print 3.

astring = "Hello world!"

print(astring[3:7])

This prints a slice of the string, starting at index 3, and ending at index 6. But why 6 and not 7? Again, most programming languages do this - it makes doing math inside those brackets easier.

If you just have one number in the brackets, it will give you the single character at that index. If you leave out the first number but keep the colon, it will give you a slice from the start to the number you left in. If you leave out the second number, it will give you a slice from the first number to the end.

You can even put negative numbers inside the brackets. They are an easy way of starting at the end of the string instead of the beginning. This way, -3 means "3rd character from the end".

astring = "Hello world!"

print(astring[3:7:2])

This prints the characters of string from 3 to 7 skipping one character. This is extended slice syntax. The general form is [start:stop:step].

This prints the characters of string from 3 to 7 skipping one character. This is extended slice syntax. The general form is [start:stop:step].

astring = "Hello world!"

print(astring[3:7])

print(astring[3:7:1])

Note that both of them produce same output

There is no function like strrev in C to reverse a string. But with the above mentioned type of slice syntax you can easily reverse a string like this

astring = "Hello world!"

print(astring[::-1])

#These make a new string with all letters converted to uppercase and lowercase, respectively.

astring = "Hello world!"

print(astring.upper())

print(astring.lower())

"""

This is used to determine whether the string starts with something or ends with something, respectively. The first one will print True, as the string starts with "Hello". The second one will print False, as the string certainly does not end with "asdfasdfasdf".

"""

astring = "Hello world!"

print(astring.startswith("Hello"))

print(astring.endswith("asdfasdfasdf"))

"""

This splits the string into a bunch of strings grouped together in a list. Since this example splits at a space, the first item in the list will be "Hello", and the second will be "world!".

"""

astring = "Hello world!"

afewwords = astring.split(" ")



## exercise below

s = "Strings are awesome!"

# Length should be 20

print("Length of s = %d" % len(s))



# First occurrence of "a" should be at index 8

print("The first occurrence of the letter a = %d" % s.index("a"))



# Number of a's should be 2

print("a occurs %d times" % s.count("a"))



# Slicing the string into bits

print("The first five characters are '%s'" % s[:5]) # Start to 5

print("The next five characters are '%s'" % s[5:10]) # 5 to 10

print("The thirteenth character is '%s'" % s[12]) # Just number 12

print("The characters with odd index are '%s'" %s[1::2]) #(0-based indexing)

print("The last five characters are '%s'" % s[-5:]) # 5th-from-last to end



# Convert everything to uppercase

print("String in uppercase: %s" % s.upper())



# Convert everything to lowercase

print("String in lowercase: %s" % s.lower())



# Check how a string starts

if s.startswith("Str"):

    print("String starts with 'Str'. Good!")



# Check how a string ends

if s.endswith("ome!"):

    print("String ends with 'ome!'. Good!")



# Split the string into three separate strings,

# each containing only a word

print("Split the words of the string: %s" % s.split(" "))

Conditions


Python uses boolean variables to evaluate conditions. The boolean values True and False are returned when an expression is compared or evaluated. 

Notice that variable assignment is done using a single equals operator "=", whereas comparison between two variables is done using the double equals operator "==". The "not equals" operator is marked as "!=".

Boolean operators
The "and" and "or" boolean operators allow building complex boolean expressions, for example:

x = 2

print(x == 2) # prints out True

print(x == 3) # prints out False

print(x < 3) # prints out True

name = "John"

age = 23

if name == "John" and age == 23:

    print("Your name is John, and you are also 23 years old.")

if name == "John" or name == "Rick":

    print("Your name is either John or Rick.")

The "in" operator
The "in" operator could be used to check if a specified object exists within an iterable object container, such as a list:

name = "John"

if name in ["John", "Rick"]:

    print("Your name is either John or Rick.")

Python uses indentation to define code blocks, instead of brackets. The standard Python indentation is 4 spaces, although tabs and any other space size will work, as long as it is consistent. Notice that code blocks do not need any termination.

Here is an example for using Python's "if" statement using code blocks

statement = False

another_statement = True

if statement is True:

    # do something

    pass

elif another_statement is True: # else if

    # do something else

    pass

else:

    # do another thing

    pass



x = 2

if x == 2:

    print("x equals two!")

else:

    print("x does not equal to two.")

A statement is evaulated as true if one of the following is correct: 1. The "True" boolean variable is given, or calculated using an expression, such as an arithmetic comparison. 2. An object which is not considered "empty" is passed.

Here are some examples for objects which are considered as empty: 1. An empty string: "" 2. An empty list: [] 3. The number zero: 0 4. The false boolean variable: False

The 'is' operator
Unlike the double equals operator "==", the "is" operator does not match the values of the variables, but the instances themselves. For example:

x = [1,2,3]

y = [1,2,3]

print(x == y) # Prints out True

print(x is y) # Prints out False

The "not" operator
Using "not" before a boolean expression inverts it:

print(not False) # Prints out True

print((not False) == (False)) # Prints out False



# Exercise, prints 1 2 3 4 5 6

number = 16

second_number = 0

first_array = [1,2,3]

second_array = [1,2]

if number > 15:

    print("1")

if first_array:

    print("2")

if len(second_array) == 2:

    print("3")

if len(first_array) + len(second_array) == 5:

    print("4")

if first_array and first_array[0] == 1:

    print("5")

if not second_number:

    print("6")



Loops


There are two types of loops in Python, for and while.

The "for" loop
For loops iterate over a given sequence. 

For loops can iterate over a sequence of numbers using the "range" and "xrange" functions. The difference between range and xrange is that the range function returns a new list with numbers of that specified range, whereas xrange returns an iterator, which is more efficient. (Python 3 uses the range function, which acts like xrange). Note that the range function is zero based.

"while" loops
While loops repeat as long as a certain boolean condition is met. 

"break" and "continue" statements
break is used to exit a for loop or a while loop, whereas continue is used to skip the current block, and return to the "for" or "while" statement. 

unlike languages like C,CPP.. we can use else for loops. When the loop condition of "for" or "while" statement fails then code part in "else" is executed. If break statement is executed inside for loop then the "else" part is skipped. Note that "else" part is executed even if there is a continue statement

# Prints out 0,1,2,3,4 and then it prints "count value reached 5"



count=0

while(count<5):

    print(count)

    count +=1

else:

    print("count value reached %d" %(count))



# Prints out 1,2,3,4

for i in range(1, 10):

    if(i%5==0):

        break

    print(i)

else:

    print("this is not printed because for loop is terminated because of break but not due to fail in condition")



###

primes = [2, 3, 5, 7]

for prime in primes:

    print(prime)



# Prints out the numbers 0,1,2,3,4

for x in range(5):

    print(x)



# Prints out 3,4,5

for x in range(3, 6):

    print(x)



# Prints out 3,5,7

for x in range(3, 8, 2):

    print(x)





# Prints out 0,1,2,3,4



count = 0

while count < 5:

    print(count)

    count += 1  # This is the same as count = count + 1





# Prints out 0,1,2,3,4



count = 0

while True:

    print(count)

    count += 1

    if count >= 5:

        break



# Prints out only odd numbers - 1,3,5,7,9

for x in range(10):

    # Check if x is even

    if x % 2 == 0:

        continue

    print(x)



"""

Loop through and print out all even numbers from the numbers list in the same order they are received. Don't print any numbers that come after 237 in the sequence.

"""

numbers = [

    951, 402, 984, 651, 360, 69, 408, 319, 601, 485, 980, 507, 725, 547, 544,

    615, 83, 165, 141, 501, 263, 617, 865, 575, 219, 390, 984, 592, 236, 105, 942, 941,

    386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345,

    399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217,

    815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717,

    958, 609, 842, 451, 688, 753, 854, 685, 93, 857, 440, 380, 126, 721, 328, 753, 470,

    743, 527

]



# your code goes here

for number in numbers:

    if number == 237:

        break



    if number % 2 == 1:

        continue



    print(number)







Functions


What are Functions?
Functions are a convenient way to divide your code into useful blocks, allowing us to order our code, make it more readable, reuse it and save some time. Also functions are a key way to define interfaces so programmers can share their code.

How do you write functions in Python?
As we have seen on previous tutorials, Python makes use of blocks.

A block is a area of code of written in the format 

Where a block line is more Python code (even another block), and the block head is of the following format: block_keyword block_name(argument1,argument2, ...) Block keywords you already know are "if", "for", and "while".

Functions in python are defined using the block keyword "def", followed with the function's name as the block's name. For example:

Functions may also receive arguments (variables passed from the caller to the function). For example:

Functions may return a value to the caller, using the keyword- 'return' . For example:

How do you call functions in Python?
Simply write the function's name followed by (), placing any required arguments within the brackets. For example, lets call the functions written above (in the previous example):

# Define our 3 functions

def my_function():

    print("Hello From My Function!")



def my_function_with_args(username, greeting):

    print("Hello, %s , From My Function!, I wish you %s"%(username, greeting))



def sum_two_numbers(a, b):

    return a + b



# print(a simple greeting)

my_function()



#prints - "Hello, John Doe, From My Function!, I wish you a great year!"

my_function_with_args("John Doe", "a great year!")



# after this line x will hold the value 3!

x = sum_two_numbers(1,2)





####

block_head:

    1st block line

    2nd block line

    ...



###

def my_function():

    print("Hello From My Function!")

###

def my_function_with_args(username, greeting):

    print("Hello, %s , From My Function!, I wish you %s"%(username, greeting))

####

def sum_two_numbers(a, b):

    return a + b

###









In this exercise you'll use an existing function, and while adding your own to create a fully functional program.

Add a function named list_benefits() that returns the following list of strings: "More organized code", "More readable code", "Easier code reuse", "Allowing programmers to share and connect code together"

Add a function named build_sentence(info) which receives a single argument containing a string and returns a sentence starting with the given string and ending with the string " is a benefit of functions!"

Run and see all the functions work together!



# Modify this function to return a list of strings as defined above

def list_benefits():

    return "More organized code", "More readable code", "Easier code reuse", "Allowing programmers to share and connect code together"



# Modify this function to concatenate to each benefit - " is a benefit of functions!"

def build_sentence(benefit):

    return "%s is a benefit of functions!" % benefit





def name_the_benefits_of_functions():

    list_of_benefits = list_benefits()

    for benefit in list_of_benefits:

        print(build_sentence(benefit))



name_the_benefits_of_functions()







Classes and Objects


Objects are an encapsulation of variables and functions into a single entity. Objects get their variables and functions from classes. Classes are essentially a template to create your objects.

A very basic class would look something like this:

We'll explain why you have to include that "self" as a parameter a little bit later. First, to assign the above class(template) to an object you would do the following:

Now the variable "myobjectx" holds an object of the class "MyClass" that contains the variable and the function defined within the class called "MyClass".

Accessing Object Variables
To access the variable inside of the newly created object "myobjectx" you would do the following:



So for instance the below would output the string "blah":

You can create multiple different objects that are of the same class(have the same variables and functions defined). However, each object contains independent copies of the variables defined in the class. For instance, if we were to define another object with the "MyClass" class and then change the string in the variable above:

Accessing Object Functions
To access a function inside of an object you use notation similar to accessing a variable:

The above would print out the message, "This is a message inside the class."



class MyClass:

    variable = "blah"



    def function(self):

        print("This is a message inside the class.")



###

class MyClass:

    variable = "blah"



    def function(self):

        print("This is a message inside the class.")



myobjectx = MyClass()



####



class MyClass:

    variable = "blah"



    def function(self):

        print("This is a message inside the class.")



myobjectx = MyClass()



myobjectx.variable



####

class MyClass:

    variable = "blah"



    def function(self):

        print("This is a message inside the class.")



myobjectx = MyClass()



print(myobjectx.variable)



####

class MyClass:

    variable = "blah"



    def function(self):

        print("This is a message inside the class.")



myobjectx = MyClass()

myobjecty = MyClass()



myobjecty.variable = "yackity"



# Then print out both values

print(myobjectx.variable)

print(myobjecty.variable)



####

class MyClass:

    variable = "blah"



    def function(self):

        print("This is a message inside the class.")



myobjectx = MyClass()



myobjectx.function()



"""

We have a class defined for vehicles. Create two new vehicles called car1 and car2. Set car1 to be a red convertible worth $60,000.00 with a name of Fer, and car2 to be a blue van named Jump worth $10,000.00.

"""

# define the Vehicle class

class Vehicle:

    name = ""

    kind = "car"

    color = ""

    value = 100.00

    def description(self):

        desc_str = "%s is a %s %s worth $%.2f." % (self.name, self.color, self.kind, self.value)

        return desc_str



# your code goes here

car1 = Vehicle()

car1.name = "Fer"

car1.color = "red"

car1.kind = "convertible"

car1.value = 60000.00



car2 = Vehicle()

car2.name = "Jump"

car2.color = "blue"

car2.kind = "van"

car2.value = 10000.00



# test code

print(car1.description())

print(car2.description())

Dictionaries


A dictionary is a data type similar to arrays, but works with keys and values instead of indexes. Each value stored in a dictionary can be accessed using a key, which is any type of object (a string, a number, a list, etc.) instead of using its index to address it.

For example, a database of phone numbers could be stored using a dictionary like this:



Alternatively, a dictionary can be initialized with the same values in the following notation:



Iterating over dictionaries
Dictionaries can be iterated over, just like a list. However, a dictionary, unlike a list, does not keep the order of the values stored in it. To iterate over key value pairs, use the following syntax:

Removing a value
To remove a specified index, use either one of the following notations:



phonebook = {}

phonebook["John"] = 938477566

phonebook["Jack"] = 938377264

phonebook["Jill"] = 947662781

print(phonebook)

phonebook = {

    "John" : 938477566,

    "Jack" : 938377264,

    "Jill" : 947662781

}

print(phonebook)



phonebook = {"John" : 938477566,"Jack" : 938377264,"Jill" : 947662781}

for name, number in phonebook.items():

    print("Phone number of %s is %d" % (name, number))



phonebook = {

   "John" : 938477566,

   "Jack" : 938377264,

   "Jill" : 947662781

}

del phonebook["John"]

print(phonebook)



phonebook = {

   "John" : 938477566,

   "Jack" : 938377264,

   "Jill" : 947662781

}

phonebook.pop("John")

print(phonebook)



Add "Jake" to the phonebook with the phone number 938273443, and remove Jill from the phonebook.





phonebook = {

    "John" : 938477566,

    "Jack" : 938377264,

    "Jill" : 947662781

}



# write your code here

phonebook["Jake"] = 938273443

del phonebook["Jill"]



# testing code

if "Jake" in phonebook:

    print("Jake is listed in the phonebook.")

if "Jill" not in phonebook:

    print("Jill is not listed in the phonebook.")

Modules and Packages


In programming, a module is a piece of software that has a specific functionality. For example, when building a ping pong game, one module would be responsible for the game logic, and

another module would be responsible for drawing the game on the screen. Each module is a different file, which can be edited separately.

Writing modules
Modules in Python are simply Python files with a .py extension. The name of the module will be the name of the file. A Python module can have a set of functions, classes or variables defined and implemented. In the example above, we will have two files, we will have:

The Python script game.py will implement the game. It will use the function draw_game from the file draw.py, or in other words, thedraw module, that implements the logic for drawing the game on the screen.

Modules are imported from other modules using the import command. In this example, the game.py script may look something like this:

The draw module may look something like this:



In this example, the game module imports the draw module, which enables it to use functions implemented in that module. The main function would use the local function play_game to run the game, and then draw the result of the game using a function implemented in the draw module called draw_game. To use the function draw_game from the draw module, we would need to specify in which module the function is implemented, using the dot operator. To reference the draw_game function from the game module, we would need to import the draw module and only then call draw.draw_game().

When the import draw directive will run, the Python interpreter will look for a file in the directory which the script was executed from, by the name of the module with a .py prefix, so in our case it will try to look for draw.py. If it will find one, it will import it. If not, he will continue to look for built-in modules.

You may have noticed that when importing a module, a .pyc file appears, which is a compiled Python file. Python compiles files into Python bytecode so that it won't have to parse the files each time modules are loaded. If a .pyc file exists, it gets loaded instead of the .py file, but this process is transparent to the user.

Importing module objects to the current namespace
We may also import the function draw_game directly into the main script's namespace, by using the from command.

You may have noticed that in this example, draw_game does not precede with the name of the module it is imported from, because we've specified the module name in the import command.

The advantages of using this notation is that it is easier to use the functions inside the current module because you don't need to specify which module the function comes from. However, any namespace cannot have two objects with the exact same name, so the import command may replace an existing object in the namespace.

Importing all objects from a module
We may also use the import * command to import all objects from a specific module, like this:

This might be a bit risky as changes in the module might affect the module which imports it, but it is shorter and also does not require you to specify which objects you wish to import from the module.

Custom import name
We may also load modules under any name we want. This is useful when we want to import a module conditionally to use the same name in the rest of the code.

For example, if you have two draw modules with slighty different names - you may do the following:

Module initialization
The first time a module is loaded into a running Python script, it is initialized by executing the code in the module once. If another module in your code imports the same module again, it will not be loaded twice but once only - so local variables inside the module act as a "singleton" - they are initialized only once.

This is useful to know, because this means that you can rely on this behavior for initializing objects. For example:

Extending module load path
There are a couple of ways we could tell the Python interpreter where to look for modules, aside from the default, which is the local directory and the built-in modules. You could either use the environment variable PYTHONPATH to specify additional directories to look for modules in, like this:

This will execute game.py, and will enable the script to load modules from the foo directory as well as the local directory.

Another method is the sys.path.append function. You may execute it before running an import command:

This will add the foo directory to the list of paths to look for modules in as well.

Exploring built-in modules
Check out the full list of built-in modules in the Python standard library here.

Two very important functions come in handy when exploring modules in Python - the dir and help functions.

If we want to import the module urllib, which enables us to create read data from URLs, we simply import the module:

We can look for which functions are implemented in each module by using the dir function:



When we find the function in the module we want to use, we can read about it more using the help function, inside the Python interpreter:



Writing packages
Packages are namespaces which contain multiple packages and modules themselves. They are simply directories, but with a twist.

Each package in Python is a directory which MUST contain a special file called __init__.py. This file can be empty, and it indicates that the directory it contains is a Python package, so it can be imported the same way a module can be imported.

If we create a directory called foo, which marks the package name, we can then create a module inside that package called bar. We also must not forget to add the __init__.py file inside the foo directory.

To use the module bar, we can import it in two ways:

In the first method, we must use the foo prefix whenever we access the module bar. In the second method, we don't, because we import the module to our module's namespace.

The __init__.py file can also decide which modules the package exports as the API, while keeping other modules internal, by overriding the __all__ variable, like so:

__init__.py:



__all__ = ["bar"]



mygame/

mygame/game.py

mygame/draw.py



# game.py

# import the draw module

import draw



def play_game():

    ...



def main():

    result = play_game()

    draw.draw_game(result)



# this means that if this script is executed, then 

# main() will be executed

if __name__ == '__main__':

    main()





# draw.py



def draw_game():

    ...



def clear_screen(screen):

    ...



# game.py

# import the draw module

from draw import draw_game



def main():

    result = play_game()

    draw_game(result)





# game.py

# import the draw module

from draw import *



def main():

    result = play_game()

    draw_game(result)







# game.py

# import the draw module

if visual_mode:

    # in visual mode, we draw using graphics

    import draw_visual as draw

else:

    # in textual mode, we print out text

    import draw_textual as draw



def main():

    result = play_game()

    # this can either be visual or textual depending on visual_mode

    draw.draw_game(result)









# draw.py



def draw_game():

    # when clearing the screen we can use the main screen object initialized in this module

    clear_screen(main_screen)

    ...



def clear_screen(screen):

    ...



class Screen():

    ...



# initialize main_screen as a singleton

main_screen = Screen()









PYTHONPATH=/foo python game.py







sys.path.append("/foo")





# import the library

import urllib



# use it

urllib.urlopen(...)







>>> import urllib

>>> dir(urllib)

['ContentTooShortError', 'FancyURLopener', 'MAXFTPCACHE', 'URLopener', '__all__', '__builtins__', 

'__doc__', '__file__', '__name__', '__package__', '__version__', '_ftperrors', '_get_proxies', 

'_get_proxy_settings', '_have_ssl', '_hexdig', '_hextochr', '_hostprog', '_is_unicode', '_localhost', 

'_noheaders', '_nportprog', '_passwdprog', '_portprog', '_queryprog', '_safe_map', '_safe_quoters', 

'_tagprog', '_thishost', '_typeprog', '_urlopener', '_userprog', '_valueprog', 'addbase', 'addclosehook', 

'addinfo', 'addinfourl', 'always_safe', 'basejoin', 'c', 'ftpcache', 'ftperrors', 'ftpwrapper', 'getproxies', 

'getproxies_environment', 'getproxies_macosx_sysconf', 'i', 'localhost', 'main', 'noheaders', 'os', 

'pathname2url', 'proxy_bypass', 'proxy_bypass_environment', 'proxy_bypass_macosx_sysconf', 'quote', 

'quote_plus', 'reporthook', 'socket', 'splitattr', 'splithost', 'splitnport', 'splitpasswd', 'splitport', 

'splitquery', 'splittag', 'splittype', 'splituser', 'splitvalue', 'ssl', 'string', 'sys', 'test', 'test1', 

'thishost', 'time', 'toBytes', 'unquote', 'unquote_plus', 'unwrap', 'url2pathname', 'urlcleanup', 'urlencode', 

'urlopen', 'urlretrieve']







help(urllib.urlopen)









import foo.bar



from foo import bar





In this exercise, you will need to print an alphabetically sorted list of all functions in the re module, which contain the word find.





import re



# Your code goes here

find_members = []

for member in dir(re):

    if "find" in member:

        find_members.append(member)



print(sorted(find_members))

Numpy Arrays


Getting started
Numpy arrays are great alternatives to Python Lists. Some of the key advantages of Numpy arrays are that they are fast, easy to work with, and give users the opportunity to perform calculations across entire arrays.

In the following example, you will first create two Python lists. Then, you will import the numpy package and create numpy arrays out of the newly created lists.

# Create 2 new lists height and weight

height = [1.87,  1.87, 1.82, 1.91, 1.90, 1.85]

weight = [81.65, 97.52, 95.25, 92.98, 86.18, 88.45]



# Import the numpy package as np

import numpy as np



# Create 2 numpy arrays from height and weight

np_height = np.array(height)

np_weight = np.array(weight)

#Print out the type of np_height
print(type(np_height)) 



Element-wise calculations
Now we can perform element-wise calculations on height and weight. For example, you could take all 6 of the height and weight observations above, and calculate the BMI for each observation with a single equation. These operations are very fast and computationally efficient. They are particularly helpful when you have 1000s of observations in your data.

# Calculate bmi

bmi = np_weight / np_height ** 2



# Print the result

print(bmi)

Subsetting
Another great feature of Numpy arrays is the ability to subset. For instance, if you wanted to know which observations in our BMI array are above 23, we could quickly subset it to find out.

# For a boolean response

bmi > 23



# Print only those observations above 23

bmi[bmi > 23]

Exercise
First, convert the list of weights from a list to a Numpy array. Then, convert all of the weights from kilograms to pounds. Use the scalar conversion of 2.2 lbs per kilogram to make your conversion. Lastly, print the resulting array of weights in pounds.

weight_kg = [81.65, 97.52, 95.25, 92.98, 86.18, 88.45]



import numpy as np



# Create a numpy array np_weight_kg from weight_kg

np_weight_kg = np.array(weight_kg)



# Create np_weight_lbs from np_weight_kg

np_weight_lbs = np_weight_kg * 2.2



# Print out np_weight_lbs

print(np_weight_lbs)

Pandas Basics


Pandas DataFrames
Pandas is a high-level data manipulation tool developed by Wes McKinney. It is built on the Numpy package and its key data structure is called the DataFrame. DataFrames allow you to store and manipulate tabular data in rows of observations and columns of variables.

There are several ways to create a DataFrame. One way way is to use a dictionary. For example:

dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],

       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],

       "area": [8.516, 17.10, 3.286, 9.597, 1.221],

       "population": [200.4, 143.5, 1252, 1357, 52.98] }



import pandas as pd

brics = pd.DataFrame(dict)

print(brics)



As you can see with the new brics DataFrame, Pandas has assigned a key for each country as the numerical values 0 through 4. If you would like to have different index values, say, the two letter country code, you can do that easily as well.

# Set the index for brics

brics.index = ["BR", "RU", "IN", "CH", "SA"]



# Print out brics with new index values

print(brics)

Another way to create a DataFrame is by importing a csv file using Pandas. Now, the csv cars.csv is stored and can be imported using pd.read_csv:



# Import pandas as pd

import pandas as pd



# Import the cars.csv data: cars

cars = pd.read_csv('cars.csv')



# Print out cars

print(cars)

Indexing DataFrames
There are several ways to index a Pandas DataFrame. One of the easiest ways to do this is by using square bracket notation.

In the example below, you can use square brackets to select one column of the cars DataFrame. You can either use a single bracket or a double bracket. The single bracket with output a Pandas Series, while a double bracket will output a Pandas DataFrame.

# Import pandas and cars.csv

import pandas as pd

cars = pd.read_csv('cars.csv', index_col = 0)



# Print out country column as Pandas Series

print(cars['cars_per_cap'])



# Print out country column as Pandas DataFrame

print(cars[['cars_per_cap']])



# Print out DataFrame with country and drives_right columns

print(cars[['cars_per_cap', 'country']])



Square brackets can also be used to access observations (rows) from a DataFrame. For example:



# Import cars data

import pandas as pd

cars = pd.read_csv('cars.csv', index_col = 0)



# Print out first 4 observations

print(cars[0:4])



# Print out fifth and sixth observation

print(cars[4:6])



You can also use loc and iloc to perform just about any data selection operation. loc is label-based, which means that you have to specify rows and columns based on their row and column labels. iloc is integer index based, so you have to specify rows and columns by their integer index like you did in the previous exercise.



# Import cars data

import pandas as pd

cars = pd.read_csv('cars.csv', index_col = 0)



# Print out observation for Japan

print(cars.iloc[2])



# Print out observations for Australia and Egypt

print(cars.loc[['AUS', 'EG']])

Generators


Generators are very easy to implement, but a bit difficult to understand.

Generators are used to create iterators, but with a different approach. Generators are simple functions which return an iterable set of items, one at a time, in a special way.

When an iteration over a set of item starts using the for statement, the generator is run. Once the generator's function code reaches a "yield" statement, the generator yields its execution back to the for loop, returning a new value from the set. The generator function can generate as many values (possibly infinite) as it wants, yielding each one in its turn.

Here is a simple example of a generator function which returns 7 random integers:

import random



def lottery():

    # returns 6 numbers between 1 and 40

    for i in range(6):

        yield random.randint(1, 40)



    # returns a 7th number between 1 and 15

    yield random.randint(1,15)



for random_number in lottery():

       print("And the next number is... %d!" %(random_number))



This function decides how to generate the random numbers on its own, and executes the yield statements one at a time, pausing in between to yield execution back to the main for loop.

Exercise
Write a generator function which returns the Fibonacci series. They are calculated using the following formula: The first two numbers of the series is always equal to 1, and each consecutive number returned is the sum of the last two numbers. Hint: Can you use only two variables in the generator function? Remember that assignments can be done simultaneously. The code



a = 1

b = 2

a, b = b, a

print(a,b)



will simultaneously switch the values of a and b.

# fill in this function

def fib():

    a, b = 1, 1

    while 1:

        yield a

        a, b = b, a + b



# testing code

import types

if type(fib()) == types.GeneratorType:

    print("Good, The fib function is a generator.")



    counter = 0

    for n in fib():

        print(n)

        counter += 1

        if counter == 10:

            break

List Comprehensions


List Comprehensions is a very powerful tool, which creates a new list based on another list, in a single, readable line.

For example, let's say we need to create a list of integers which specify the length of each word in a certain sentence, but only if the word is not the word "the".

sentence = "the quick brown fox jumps over the lazy dog"

words = sentence.split()

word_lengths = []

for word in words:

      if word != "the":

          word_lengths.append(len(word))

print(words)

print(word_lengths)



Using a list comprehension, we could simplify this process to this notation:

sentence = "the quick brown fox jumps over the lazy dog"

words = sentence.split()

word_lengths = [len(word) for word in words if word != "the"]

print(words)

print(word_lengths)



Exercise
Using a list comprehension, create a new list called "newlist" out of the list "numbers", which contains only the positive numbers from the list, as integers.

numbers = [34.6, -203.4, 44.9, 68.3, -12.2, 44.6, 12.7]

newlist = [int(x) for x in numbers if x > 0]

print(newlist)

Multiple Function Arguments


Every function in Python receives a predefined number of arguments, if declared normally, like this:

def myfunction(first, second, third):

    # do something with the 3 variables

It is possible to declare functions which receive a variable number of arguments, using the following syntax:

def foo(first, second, third, *therest):

    print("First: %s" % first)

    print("Second: %s" % second)

    print("Third: %s" % third)

    print("And all the rest... %s" % list(therest))





The "therest" variable is a list of variables, which receives all arguments which were given to the "foo" function after the first 3 arguments. So calling foo(1,2,3,4,5) will print out:

def foo(first, second, third, *therest):

    print("First: %s" %(first))

    print("Second: %s" %(second))

    print("Third: %s" %(third))

    print("And all the rest... %s" %(list(therest)))



foo(1,2,3,4,5)

It is also possible to send functions arguments by keyword, so that the order of the argument does not matter, using the following syntax. The following code yields the following output: The sum is: 6 Result: 1

def bar(first, second, third, **options):

    if options.get("action") == "sum":

        print("The sum is: %d" %(first + second + third))



    if options.get("number") == "first":

        return first



result = bar(1, 2, 3, action = "sum", number = "first")

print("Result: %d" %(result))



The "bar" function receives 3 arguments. If an additional "action" argument is received, and it instructs on summing up the numbers, then the sum is printed out. Alternatively, the function also knows it must return the first argument, if the value of the "number" parameter, passed into the function, is equal to "first".

Exercise
Fill in the foo and bar functions so they can receive a variable amount of arguments (3 or more) The foo function must return the amount of extra arguments received. The bar must return True if the argument with the keyword magicnumber is worth 7, and False otherwise.



# edit the functions prototype and implementation

def foo(a, b, c, *args):

    return len(args)



def bar(a, b, c, **kwargs):

    return kwargs["magicnumber"] == 7





# test code

if foo(1,2,3,4) == 1:

    print("Good.")

if foo(1,2,3,4,5) == 2:

    print("Better.")

if bar(1,2,3,magicnumber = 6) == False:

    print("Great.")

if bar(1,2,3,magicnumber = 7) == True:

    print("Awesome!")

Regular Expressions


Regular Expressions (sometimes shortened to regexp, regex, or re) are a tool for matching patterns in text. In Python, we have the re module. The applications for regular expressions are wide-spread, but they are fairly complex, so when contemplating using a regex for a certain task, think about alternatives, and come to regexes as a last resort.

An example regex is r"^(From|To|Cc).*?python-list@python.org" Now for an explanation: the caret ^ matches text at the beginning of a line. The following group, the part with (From|To|Cc) means that the line has to start with one of the words that are separated by the pipe |. That is called the OR operator, and the regex will match if the line starts with any of the words in the group. The .*? means to un-greedily match any number of characters, except the newline \n character. The un-greedy part means to match as few repetitions as possible. The . character means any non-newline character, the * means to repeat 0 or more times, and the ? character makes it un-greedy.

So, the following lines would be matched by that regex: From: python-list@python.org To: !asp]<,. python-list@python.org

A complete reference for the re syntax is available at the python docs.

As an example of a "proper" email-matching regex (like the one in the exercise), see this

# Example: 

import re

pattern = re.compile(r"\[(on|off)\]") # Slight optimization

print(re.search(pattern, "Mono: Playback 65 [75%] [-16.50dB] [on]"))

# Returns a Match object!

print(re.search(pattern, "Nada...:-("))

# Doesn't return anything.

# End Example



# Exercise: make a regular expression that will match an email

def test_email(your_pattern):

    pattern = re.compile(your_pattern)

    emails = ["john@example.com", "python-list@python.org", "wha.t.`1an?ug{}ly@email.com"]

    for email in emails:

        if not re.match(pattern, email):

            print("You failed to match %s" % (email))

        elif not your_pattern:

            print("Forgot to enter a pattern!")

        else:

            print("Pass")

pattern = r"" # Your pattern here!

test_email(pattern)



##### solution

# Exercise: make a regular expression that will match an email

import re

def test_email(your_pattern):

    pattern = re.compile(your_pattern)

    emails = ["john@example.com", "python-list@python.org", "wha.t.`1an?ug{}ly@email.com"]

    for email in emails:

        if not re.match(pattern, email):

            print("You failed to match %s" % (email))

        elif not your_pattern:

            print("Forgot to enter a pattern!")

        else:

            print("Pass")

# Your pattern here!

pattern = r"\"?([-a-zA-Z0-9.`?{}]+@\w+\.\w+)\"?"

test_email(pattern)

Exception Handling


When programming, errors happen. It's just a fact of life. Perhaps the user gave bad input. Maybe a network resource was unavailable. Maybe the program ran out of memory. Or the programmer may have even made a mistake!

Python's solution to errors are exceptions. You might have seen an exception before.

print(a)



#error

Traceback (most recent call last):

  File "<stdin>", line 1, in <module>

NameError: name 'a' is not defined

</module></stdin>



Oops! Forgot to assign a value to the 'a' variable.

But sometimes you don't want exceptions to completely stop the program. You might want to do something special when an exception is raised. This is done in a try/except block.

Here's a trivial example: Suppose you're iterating over a list. You need to iterate over 20 numbers, but the list is made from user input, and might not have 20 numbers in it. After you reach the end of the list, you just want the rest of the numbers to be interpreted as a 0. Here's how you could do that:

def do_stuff_with_number(n):

    print(n)



def catch_this():

    the_list = (1, 2, 3, 4, 5)



    for i in range(20):

        try:

            do_stuff_with_number(the_list[i])

        except IndexError: # Raised when accessing a non-existing index of a list

            do_stuff_with_number(0)



catch_this()



There, that wasn't too hard! You can do that with any exception. For more details on handling exceptions, look no further than the Python Docs

Exercise
Handle all the exception! Think back to the previous lessons to return the last name of the actor.

actor = {"name": "John Cleese", "rank": "awesome"}



def get_last_name():

    return actor["name"].split()[1]



get_last_name()

print("All exceptions caught! Good job!")

print("The actor's last name is %s" % get_last_name())

Sets


Sets are lists with no duplicate entries. Let's say you want to collect a list of words used in a paragraph:

print(set("my name is Eric and Eric is my name".split()))

This will print out a list containing "my", "name", "is", "Eric", and finally "and". Since the rest of the sentence uses words which are already in the set, they are not inserted twice.

Sets are a powerful tool in Python since they have the ability to calculate differences and intersections between other sets. For example, say you have a list of participants in events A and B:

a = set(["Jake", "John", "Eric"])

print(a)

b = set(["John", "Jill"])

print(b)



To find out which members attended both events, you may use the "intersection" method:



a = set(["Jake", "John", "Eric"])

b = set(["John", "Jill"])



print(a.intersection(b))

print(b.intersection(a))



To find out which members attended only one of the events, use the "symmetric_difference" method:



a = set(["Jake", "John", "Eric"])

b = set(["John", "Jill"])



print(a.symmetric_difference(b))

print(b.symmetric_difference(a))

To find out which members attended only one event and not the other, use the "difference" method:



a = set(["Jake", "John", "Eric"])

b = set(["John", "Jill"])



print(a.difference(b))

print(b.difference(a))

To receive a list of all participants, use the "union" method:



a = set(["Jake", "John", "Eric"])

b = set(["John", "Jill"])



print(a.union(b))



In the exercise below, use the given lists to print out a set containing all the participants from event A which did not attend event B.



a = ["Jake", "John", "Eric"]

b = ["John", "Jill"]



A = set(a)

B = set(b)



print(A.difference(B))



Serialization


Python provides built-in JSON libraries to encode and decode JSON.

In Python 2.5, the simplejson module is used, whereas in Python 2.7, the json module is used. Since this interpreter uses Python 2.7, we'll be using json.

In order to use the json module, it must first be imported:

import  json

There are two basic formats for JSON data. Either in a string or the object datastructure. The object datastructure, in Python, consists of lists and dictionaries nested inside each other. The object datastructure allows one to use python methods (for lists and dictionaries) to add, list, search and remove elements from the datastructure. The String format is mainly used to pass the data into another program or load into a datastructure.

To load JSON back to a data structure, use the "loads" method. This method takes a string and turns it back into the json object datastructure:

import  json 

print(json.loads(json_string))

To encode a data structure to JSON, use the "dumps" method. This method takes an object and returns a String:



import json

json_string = json.dumps([1, 2, 3, "a", "b", "c"])

print(json_string)



Python supports a Python proprietary data serialization method called pickle (and a faster alternative called cPickle).

You can use it exactly the same way.

import pickle

pickled_string = pickle.dumps([1, 2, 3, "a", "b", "c"])

print(pickle.loads(pickled_string))



The aim of this exercise is to print out the JSON string with key-value pair "Me" : 800 added to it.



import json



# fix this function, so it adds the given name

# and salary pair to salaries_json, and return it

def add_employee(salaries_json, name, salary):

    salaries = json.loads(salaries_json)

    salaries[name] = salary



    return json.dumps(salaries)



# test code

salaries = '{"Alfred" : 300, "Jane" : 400 }'

new_salaries = add_employee(salaries, "Me", 800)

decoded_salaries = json.loads(new_salaries)

print(decoded_salaries["Alfred"])

print(decoded_salaries["Jane"])

print(decoded_salaries["Me"])





Partial functions


You can create partial functions in python by using the partial function from the functools library.

Partial functions allow one to derive a function with x parameters to a function with fewer parameters and fixed values set for the more limited function.

Import required:

from functools import partial

This code will return 8.



from functools import partial



def multiply(x,y):

        return x * y



# create a new function that multiplies by 2

dbl = partial(multiply,2)

print(dbl(4))

An important note: the default values will start replacing variables from the left. The 2 will replace x. y will equal 4 when dbl(4) is called. It does not make a difference in this example, but it does in the example below.

Exercise
Edit the function provided by calling partial() and replacing the first three variables in func(). Then print with the new partial function using only one input variable so that the output equals 60.

from functools import partial

def func(u,v,w,x):

    return u*4 + v*3 + w*2 + x



p = partial(func,5,6,7)

print(p(8))



Code Introspection


Code introspection is the ability to examine classes, functions and keywords to know what they are, what they do and what they know.

Python provides several functions and utilities for code introspection.

help()

dir() 

hasattr() 

id() 

type() 

repr() 

callable() 

issubclass() 

isinstance() 

__doc__ 

__name__

Often the most important one is the help function, since you can use it to find what other functions do.

Exercise
Print a list of all attributes of the given Vehicle object.

# Use the help function to see what each function does.

# Delete this when you are done.

help(dir)

help(hasattr)

help(id)



# Define the Vehicle class.

class Vehicle:

    name = ""

    kind = "car"

    color = ""

    value = 100.00

    def description(self):

        desc_str = "%s is a %s %s worth $%.2f." % (self.name, self.color, self.kind, self.value)

        return desc_str



# Print a list of all attributes of the Vehicle class.

# Your code goes here



### solution

# Define the Vehicle class

class Vehicle:

    name = ""

    kind = "car"

    color = ""

    value = 100.00

    def description(self):

        desc_str = "%s is a %s %s worth $%.2f." % (self.name, self.color, self.kind, self.value)

        return desc_str



# Print a list of all attributes of the Vehicle class.

print(dir(Vehicle))





Closures


A Closure is a function object that remembers values in enclosing scopes even if they are not present in memory. Let us get to it step by step

Firstly, a Nested Function is a function defined inside another function. It's very important to note that the nested functions can access the variables of the enclosing scope. However, at least in python, they are only readonly. However, one can use the "nonlocal" keyword explicitly with these variables in order to modify them.

For example:

def transmit_to_space(message):

    "This is the enclosing function"

    def data_transmitter():

        "The nested function"

        print(message)



    data_transmitter()



print(transmit_to_space("Test message"))



This works well as the 'data_transmitter' function can access the 'message'. To demonstrate the use of the "nonlocal" keyword, consider this



def print_msg(number):

    def printer():

        "Here we are using the nonlocal keyword"

        nonlocal number

        number=3

        print(number)

    printer()

    print(number)



print_msg(9)



Without the nonlocal keyword, the output would be "3 9", however, with its usage, we get "3 3", that is the value of the "number" variable gets modified.

Now, how about we return the function object rather than calling the nested function within. (Remember that even functions are objects. (It's Python.))

def transmit_to_space(message):

    "This is the enclosing function"

    def data_transmitter():

        "The nested function"

        print(message)

    return data_transmitter



And we call the function as follows:



def transmit_to_space(message):

  "This is the enclosing function"

  def data_transmitter():

      "The nested function"

      print(message)

  return data_transmitter



fun2 = transmit_to_space("Burn the Sun!")

fun2()



Even though the execution of the "transmit_to_space()" was completed, the message was rather preserved. This technique by which the data is attached to some code even after end of those other original functions is called as closures in python

ADVANTAGE : Closures can avoid use of global variables and provides some form of data hiding.(Eg. When there are few methods in a class, use closures instead).

Also, Decorators in Python make extensive use of closures.

Exercise
Make a nested loop and a python closure to make functions to get multiple multiplication functions using closures. That is using closures, one could make functions to create multiply_with_5() or multiply_with_4() functions using closures.

def multiplier_of(n):

    def multiplier(number):

        return number*n

    return multiplier



multiplywith5 = multiplier_of(5)

print(multiplywith5(9))

Decorators


Decorators allow you to make simple modifications to callable objects like functions, methods, or classes. We shall deal with functions for this tutorial. The syntax

@decorator

def functions(arg):

    return "value"

Is equivalent to:



def function(arg):

    return "value"

function = decorator(function) # this passes the function to the decorator, and reassigns it to the functions



As you may have seen, a decorator is just another function which takes a functions and returns one. For example you could do this:



def repeater(old_function):

    def new_function(*args, **kwds): # See learnpython.org/en/Multiple%20Function%20Arguments for how *args and **kwds works

        old_function(*args, **kwds) # we run the old function

        old_function(*args, **kwds) # we do it twice

    return new_function # we have to return the new_function, or it wouldn't reassign it to the value



This would make a function repeat twice.



>>> @repeater

def multiply(num1, num2):

    print(num1 * num2)



>>> multiply(2, 3)

6

6

You can also make it change the output



def double_out(old_function):

    def new_function(*args, **kwds):

        return 2 * old_function(*args, **kwds) # modify the return value

    return new_function



change input



def double_Ii(old_function):

    def new_function(arg): # only works if the old function has one argument

        return old_function(arg * 2) # modify the argument passed

    return new_function

and do checking.



def check(old_function):

    def new_function(arg):

        if arg < 0: raise (ValueError, "Negative Argument") # This causes an error, which is better than it doing the wrong thing

        old_function(arg)

    return new_function



Let's say you want to multiply the output by a variable amount. You could define the decorator and use it as follows:



    def multiply_generator(old_function):

        def new_function(*args, **kwds):

            return multiplier * old_function(*args, **kwds)

        return new_function

    return multiply_generator # it returns the new generator



# Usage

@multiply(3) # multiply is not a generator, but multiply(3) is

def return_num(num):

    return num



# Now return_num is decorated and reassigned into itself

return_num(5) # should return 15



You can do anything you want with the old function, even completely ignore it! Advanced decorators can also manipulate the doc string and argument number. For some snazzy decorators, go to http://wiki.python.org/moin/PythonDecoratorLibrary.

Exercise
Make a decorator factory which returns a decorator that decorates functions with one argument. The factory should take one argument, a type, and then returns a decorator that makes function should check if the input is the correct type. If it is wrong, it should print("Bad Type") (In reality, it should raise an error, but error raising isn't in this tutorial). Look at the tutorial code and expected output to see what it is if you are confused (I know I would be.) Using isinstance(object, type_of_object) or type(object) might help.

def type_check(correct_type):

    #put code here



@type_check(int)

def times2(num):

    return num*2



print(times2(2))

times2('Not A Number')



@type_check(str)

def first_letter(word):

    return word[0]



print(first_letter('Hello World'))

first_letter(['Not', 'A', 'String'])



Map, Filter, Reduce


Map, Filter, and Reduce are paradigms of functional programming. They allow the programmer (you) to write simpler, shorter code, without neccessarily needing to bother about intricacies like loops and branching.

Essentially, these three functions allow you to apply a function across a number of iterables, in one full swoop. map and filter come built-in with Python (in the __builtins__ module) and require no importing. reduce, however, needs to be imported as it resides in the functools module. Let's get a better understanding of how they all work, starting with map.

Map

The map() function in python has the following syntax:

map(func, *iterables)

Where func is the function on which each element in iterables (as many as they are) would be applied on. Notice the asterisk(*) on iterables? It means there can be as many iterables as possible, in so far func has that exact number as required input arguments. Before we move on to an example, it's important that you note the following:

In Python 2, the map() function returns a list. In Python 3, however, the function returns a map object which is a generator object. To get the result as a list, the built-in list() function can be called on the map object. i.e. list(map(func, *iterables))

The number of arguments to func must be the number of iterables listed.

Let's see how these rules play out with the following examples.

Say I have a list (iterable) of my favourite pet names, all in lower case and I need them in uppercase. Traditonally, in normal pythoning, I would do something like this:

my_pets = ['alfred', 'tabitha', 'william', 'arla']

uppered_pets = []



for pet in my_pets:

    pet_ = pet.upper()

    uppered_pets.append(pet_)



print(uppered_pets)



Which would then output ['ALFRED', 'TABITHA', 'WILLIAM', 'ARLA']

With map() functions, it's not only easier, but it's also much more flexible. I simply do this:

# Python 3

my_pets = ['alfred', 'tabitha', 'william', 'arla']



uppered_pets = list(map(str.upper, my_pets))



print(uppered_pets)

Which would also output the same result. Note that using the defined map() syntax above, func in this case is str.upper and iterables is the my_pets list -- just one iterable. Also note that we did not call the str.upper function (doing this: str.upper()), as the map function does that for us on each element in the my_pets list.

What's more important to note is that the str.upper function requires only one argument by definition and so we passed just one iterable to it. So, if the function you're passing requires two, or three, or n arguments, then you need to pass in two, three or n iterables to it. Let me clarify this with another example.

Say I have a list of circle areas that I calculated somewhere, all in five decimal places. And I need to round each element in the list up to its position decimal places, meaning that I have to round up the first element in the list to one decimal place, the second element in the list to two decimal places, the third element in the list to three decimal places, etc. With map() this is a piece of cake. Let's see how.

Python already blesses us with the round() built-in function that takes two arguments -- the number to round up and the number of decimal places to round the number up to. So, since the function requires two arguments, we need to pass in two iterables.

# Python 3



circle_areas = [3.56773, 5.57668, 4.00914, 56.24241, 9.01344, 32.00013]



result = list(map(round, circle_areas, range(1,7)))



print(result)

See the beauty of map()? Can you imagine the flexibility this evokes?

The range(1,7) function acts as the second argument to the round function (the number of required decimal places per iteration). So as map iterates through circle_areas, during the first iteration, the first element of circle_areas, 3.56773 is passed along with the first element of range(1,7), 1 to round, making it effectively become round(3.56773, 1). During the second iteration, the second element of circle_areas, 5.57668 along with the second element of range(1,7), 2 is passed to round making it translate to round(5.57668, 2). This happens until the end of the circle_areas list is reached.

I'm sure you're wondering: "What if I pass in an iterable less than or more than the length of the first iterable? That is, what if I pass range(1,3) or range(1, 9999) as the second iterable in the above function". And the answer is simple: nothing! Okay, that's not true. "Nothing" happens in the sense that the map() function will not raise any exception, it will simply iterate over the elements until it can't find a second argument to the function, at which point it simply stops and returns the result.

So, for example, if you evaluate result = list(map(round, circle_areas, range(1,3))), you won't get any error even as the length of circle_areas and the length of range(1,3) differ. Instead, this is what Python does: It takes the first element of circle_areas and the first element of range(1,3) and passes it to round. round evaluates it then saves the result. Then it goes on to the second iteration, second element of circle_areas and second element of range(1,3), round saves it again. Now, in the third iteration (circle_areas has a third element), Python takes the third element of circle_areas and then tries to take the third element of range(1,3) but since range(1,3) does not have a third element, Python simply stops and returns the result, which in this case would simply be [3.6, 5.58].

Go ahead, try it.



# Python 3



circle_areas = [3.56773, 5.57668, 4.00914, 56.24241, 9.01344, 32.00013]



result = list(map(round, circle_areas, range(1,3)))



print(result)

The same thing happens if circle_areas is less than the length of the second iterable. Python simply stops when it can't find the next element in one of the iterables.

To consolidate our knowledge of the map() function, we are going to use it to implement our own custom zip() function. The zip() function is a function that takes a number of iterables and then creates a tuple containing each of the elements in the iterables. Like map(), in Python 3, it returns a generator object, which can be easily converted to a list by calling the built-in list function on it. Use the below interpreter session to get a grip of zip() before we create ours with map()

# Python 3



my_strings = ['a', 'b', 'c', 'd', 'e']

my_numbers = [1,2,3,4,5]



results = list(zip(my_strings, my_numbers))



print(results)

As a bonus, can you guess what would happen in the above session if my_strings and my_numbers are not of the same length? No? try it! Change the length of one of them.

Onto our own custom zip() function!

# Python 3



my_strings = ['a', 'b', 'c', 'd', 'e']

my_numbers = [1,2,3,4,5]



results = list(map(lambda x, y: (x, y), my_strings, my_numbers))



print(results)



Just look at that! We have the same result as zip.

Did you also notice that I didn't even need to create a function using the def my_function() standard way? That's how flexible map(), and Python in general, is! I simply used a lambda function. This is not to say that using the standard function definition method (of def function_name()) isn't allowed, it still is. I simply preferred to write less code (be "Pythonic").

That's all about map. Onto filter()

Filter

While map() passes each element in the iterable through a function and returns the result of all elements having passed through the function, filter(), first of all, requires the function to return boolean values (true or false) and then passes each element in the iterable through the function, "filtering" away those that are false. It has the following syntax:

filter(func, iterable)

The following points are to be noted regarding filter():

Unlike map(), only one iterable is required.

The func argument is required to return a boolean type. If it doesn't, filter simply returns the iterable passed to it. Also, as only one iterable is required, it's implicit that func must only take one argument.

filter passes each element in the iterable through func and returns only the ones that evaluate to true. I mean, it's right there in the name -- a "filter".

Let's see some examples

The following is a list (iterable) of the scores of 10 students in a Chemistry exam. Let's filter out those who passed with scores more than 75...using filter.



# Python 3

scores = [66, 90, 68, 59, 76, 60, 88, 74, 81, 65]



def is_A_student(score):

    return score > 75



over_75 = list(filter(is_A_student, scores))



print(over_75)



The next example will be a palindrome detector. A "palindrome" is a word, phrase, or sequence that reads the same backwards as forwards. Let's filter out words that are palindromes from a tuple (iterable) of suspected palindromes.



# Python 3

dromes = ("demigod", "rewire", "madam", "freer", "anutforajaroftuna", "kiosk")



palindromes = list(filter(lambda word: word == word[::-1], dromes))



print(palindromes)

Which should output ['madam', 'anutforajaroftuna'].

Pretty neat huh? Finally, reduce()

Reduce

reduce applies a function of two arguments cumulatively to the elements of an iterable, optionally starting with an initial argument. It has the following syntax:

reduce(func, iterable[, initial])

Where func is the function on which each element in the iterable gets cumulatively applied to, and initial is the optional value that gets placed before the elements of the iterable in the calculation, and serves as a default when the iterable is empty. The following should be noted about reduce(): 1. func requires two arguments, the first of which is the first element in iterable (if initial is not supplied) and the second the second element in iterable. If initial is supplied, then it becomes the first argument to func and the first element in iterable becomes the second element. 2. reduce "reduces" (I know, forgive me) iterable into a single value.

As usual, let's see some examples.

Let's create our own version of Python's built-in sum() function. The sum() function returns the sum of all the items in the iterable passed to it.

# Python 3

from functools import reduce



numbers = [3, 4, 6, 9, 34, 12]



def custom_sum(first, second):

    return first + second



result = reduce(custom_sum, numbers)

print(result)

The result, as you'll expect is 68.

So, what happened?

As usual, it's all about iterations: reduce takes the first and second elements in numbers and passes them to custom_sum respectively. custom_sum computes their sum and returns it to reduce. reduce then takes that result and applies it as the first element to custom_sum and takes the next element (third) in numbers as the second element to custom_sum. It does this continuously (cumulatively) until numbers is exhausted.

Let's see what happens when I use the optional initial value.

# Python 3

from functools import reduce



numbers = [3, 4, 6, 9, 34, 12]



def custom_sum(first, second):

    return first + second



result = reduce(custom_sum, numbers, 10)

print(result)

The result, as you'll expect, is 78 because reduce, initially, uses 10 as the first argument to custom_sum.

That's all about Python's Map, Reduce, and Filter. Try on the below exercises to help ascertain your understanding of each function.

Exercise
In this exercise, you'll use each of map, filter, and reduce to fix broken code.

from functools import reduce 



# Use map to print the square of each numbers rounded

# to two decimal places

my_floats = [4.35, 6.09, 3.25, 9.77, 2.16, 8.88, 4.59]



# Use filter to print only the names that are less than 

# or equal to seven letters

my_names = ["olumide", "akinremi", "josiah", "temidayo", "omoseun"]



# Use reduce to print the product of these numbers

my_numbers = [4, 6, 9, 23, 5]



# Fix all three respectively.

map_result = list(map(lambda x: x, my_floats))

filter_result = list(filter(lambda name: name, my_names, my_names))

reduce_result = reduce(lambda num1, num2: num1 * num2, my_numbers, 0)



print(map_result)

print(filter_result)

print(reduce_result)



#solution

#### Map

from functools import reduce 



my_floats = [4.35, 6.09, 3.25, 9.77, 2.16, 8.88, 4.59]

my_names = ["olumide", "akinremi", "josiah", "temidayo", "omoseun"]

my_numbers = [4, 6, 9, 23, 5]



map_result = list(map(lambda x: round(x ** 2, 3), my_floats))

filter_result = list(filter(lambda name: len(name) <= 7, my_names))

reduce_result = reduce(lambda num1, num2: num1 * num2, my_numbers)



print(map_result)

print(filter_result)

print(reduce_result)



# code clip - parity check

for i in range(int(input())):

    if input().count('1') % 2 == 0:

        print('N')

    else:

        print('Y')

python classes and objects

https://drive.google.com/file/d/1j4sgqYVkCQifkyXk3lvvTGTyz1KTTNYN/view?usp=sharing
