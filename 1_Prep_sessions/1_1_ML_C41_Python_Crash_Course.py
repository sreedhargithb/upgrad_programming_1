#!/usr/bin/env python
# coding: utf-8

# ## Contents:
# 
# 1. [Session_1](#Session_1)<br>
# 2. [Session_2](#Session_2)<br>
# 3. [Session_3](#Session_3)<br>
# 4. [Session_4](#Session_4)<br>

# <a id='Session_1'>Session-1</a>

# In[27]:


get_ipython().system('pip install flask')


# In[28]:


from IPython.display import display,HTML

## Print statement

print("Welcome to ML C41!!" + " Python Crash course")
print(str(4)+"Saturday")

# child process outputs are coming in the Anaconda terminal
import subprocess
import sys
list_files_1 = subprocess.run(["python","-c",'''
import numpy;
import flask;
print("Hi from child process");
print(2+5%5);
import subprocess
import sys
list_files_1 = subprocess.run(["python","-c",\'''
import numpy;
import flask;
print("Hi from nested child process");
print(3+5%5);
\''']);
print("The exit code was: %d" % list_files_1.returncode);
print(list_files_1.stdout);
print(list_files_1.stderr);
''']);
print("The exit code was: %d" % list_files_1.returncode);
print(list_files_1.stdout);
print(list_files_1.stderr);


# In[29]:


# Variables
    # 1. Variable name can have only alphanumeric or underscore
    # 2. Variable can start only with alphabet or underscore
    # 3. Variable names are case-sensitive
    # 4. In Python, the highest possible length of an identifier is 79 characters.
display(HTML('<h3>Variables</h3>'))   
var, Var, _var = 3,4,5
print(var, Var, _var)
__var__ = 34
__var__


# In[30]:


# Data types:
display(HTML('<h3>Data types</h3>'))
# https://www.w3schools.com/python/python_datatypes.asp

num =8; print(type(num))
num =8.0; print(type(num))
a: int = 6; print(a)  # for type declaration in Python
b: str = "Hello"; print(b*3)

display(HTML('''
<h4>Boolean</h4>
'''))
print(type(True))


# In[31]:


# Arithmetic operations
display(HTML('<h3>Arithmetic operations</h3>'))
from math import ceil, floor, trunc

x = 3
y = 6
print(x+y)  
print(x-y)  
print(x*y)
print(x/y)  # float type division
print(x//y) # integer type division
print(x%y)  # modulus (remainder)
print(x**y) # exponent
print(int(str(x)+str(y))) # string concatenation
print(round(x/y))

print(ceil(x/y))
print(floor(x/y))
print(trunc(x/y)) #equivalent to math.floor for positive numbers, equivalent to math.ceil for negative numbers


# In[32]:


# Data type conversion
display(HTML('<h3>Data type conversion</h3>'))

var = 4
print(str(var)*int(var))


# In[33]:


# ASCII values
display(HTML('<h3>ASCII values</h3>'))
print(ord("3"))
print(chr(97))
#print(chr("3")) # TypeError: an integer is required (got type str)
print(chr(96))   # does not get printed (if the error above is uncommented)


# In[34]:


#String operations
display(HTML('<h3>String operations</h3>'))

a = "python class"
b = print("b is:",a)
str = "this is "
b
print(str + a)
del str          # to delete a variable
#print(str + a)  # error (because "str" is now a datatype and not variable)

# isalpha, isnum, isalnum
print(a.isalpha())
print("python class".isalpha())  # also works
#print(43.isnum())  # invalid syntax
print((a*2).isalnum())

#Replace, swapcase, islower, isupper, lower, upper
print(a.replace(" ","").islower()) # every character has to be lower
print(a.replace(" ","").isupper()) # every character has to be upper
swap = a.swapcase()
print(swap)  # PYTHON CLASS
print(swap.lower())
print(swap.upper())


# title vs capitalize
print(a.capitalize())
print(a.title())

# split
print(a.split())
print(a.split(" "))
print(a.split("p"))


# In[35]:


# not working for subprocess input

######################## START: Taking Input in Python  ########################
# import subprocess
# import sys
# list_files_1 = subprocess.run(["python3","-c",'''
# print(input("Enter a number:"))
# ''']);
# print("The exit code was: %d" % list_files_1.returncode);
# print(list_files_1.stdout);
# print(list_files_1.stderr);

######################## END: Taking Input in Python  ########################


# <a id='Session_2'>Session-2</a>

# In[36]:


from IPython.display import display,HTML
import copy

# Data structures

display(HTML('''
<h3>Topics covered today:</h3>
<h4>Data structures</h4>
<ul>
    <li>List</li>
    <li>Tuple</li>
    <li>Dictionary</li>
    <li>Set</li>
</ul>
<h4>Input in Python</h4>
<ul>
    <li>ast - Abstract Syntax Tree</li>
</ul>
'''))



# Note:
'''
1. jovian:

import jovian
jovian.commit()
'''

display(HTML('''
<h3>List</h3>
'''))

list = [1,2,3,4,5]
list = [i for i in"123445"]
print(list)

#Normal copy (both copies pointing to same location in memory)
lis = list
print(lis)

display(HTML('''
<h4>Normal copy</h4>
'''))
lis[0] = 0
print("Normal copy:",lis)
print("Original of normal copy:",list)
# both ids are same
print('ID of New normal-copied List:', id(lis)) 
print('ID of original List of normal copying:', id(list)) 
lis[0] = -2
print("Copy affecting original:",lis,list)

#################################################################
#Shallow copy
display(HTML('''
<h4>Shallow copy</h4>
'''))
li3 = copy.copy(list)
print(list)
# both ids are different for the lists, but same for the list items (before updating)
print('ID of New shallow-copied List:', id(li3)) 
print('ID of original List of shallow copying:', id(list)) 
print('ID of New shallow-copied List item:', id(li3[0])) 
print('ID of item of original List of shallow copying:', id(list[0])) 
li3[0] = -3
print("Copy not affecting original:",li3,list)

#################################################################
#Normal copy - 2
display(HTML('''
<h4>Normal copy - 2</h4>
'''))
li = [i for i in list]
print("Normal copied:",li)
print("Original of Normal copy:",list)
# both ids are different for the lists, but same for the list items (before updating)
print('ID of New Normal-copied List:', id(li)) 
print('ID of original List of Normal copying:', id(list)) 
print('ID of New Normal-copied List item:', id(li[0])) 
print('ID of item of original List of Normal copying:', id(list[0])) 
li[0] = -33
print("Copy not affecting original:",li,list)

#################################################################
# Deepcopy (no change in original)
display(HTML('''
<h4>Deep copy</h4>
'''))
li4 = copy.deepcopy(list)
print("Deep copied:",li4)
print("Original of deep copy:",list)
# both ids are different for the lists, but same for the list items (before updating)
print('ID of New deep-copied List:', id(li4)) 
print('ID of original List of deep copying:', id(list))
print('ID of New deep-copied List item:', id(li4[0])) 
print('ID of item of original List of deep copying:', id(list[0])) 
li4[0] = -22
print("Copy not affecting original:",li4,list)

display(HTML('''
<h3>List (continued -> append, extend, insert, del, remove)</h3>
'''))

lis = [1,2,3,4,5]
lis.append(2) #single element at a time
lis.extend([2,3,4])  # multiple added elements
print(lis)

lis.insert(11,34)  # (index, element) (if index is more than size, element is appended in the end)
lis.insert(13,[4,5,6])
print(lis)

# delete by value
lis.remove(34)  # remove(element_value)
print(lis)

# delete by index
del lis[-1]
print(lis)

display(HTML('''
<h3>List (continued -> slicing)</h3>
'''))
lis = [1,2,3,4,5,6,7]
print(lis[-9:])


display(HTML('''
<h3>Dictionary</h3>
<h5>Mutable</h5>
'''))

keyval = dict()
for i in range(10):
    keyval[i] = str(i)
print(keyval)
print(keyval.keys())
print(keyval.values())
del list
print(list(keyval.values()))


display(HTML('''
<h3>Tuple</h3>
<h5>Immutable</h5>
'''))

tup = (1,2,3,4)
tup1 = tuple([1,2,3,4])
print(tup==tup1)  # true

display(HTML('''
<h3>Set</h3>
<h5>Mutable, and not ordered</h5>
'''))

set_ = set([1,1,1,3,4,2])
print(set_)
# since not ordered, set does not support indexing
# set / list cannot be used as key in dictionary (because it is mutable)

set_.add(30)
set1_ = set_
print(set1_) # {1, 2, 3, 4, 30}




display(HTML('''
<h3>Input in Python</h3>
'''))

######################## START: Taking Input in Python  ########################

# import ast,sys
# input_list = sys.stdin.readline()    # for input passed separately
# input_list = input()  
# # Write code to remove 'SPSS'
# # Write code to append 'SPARK'
# #input_list[input_list.index('SPSS')] = 'SPARK'
# print(input_list)
######################## END: Taking Input in Python  ########################


# <a id='Session_3'>Session-3</a>

# In[37]:


# older - from IPython.display import display,HTML
from IPython.display import display,HTML
import copy

# Data structures

display(HTML('''
<h3>Topics covered today:</h3>
<ul>
    <li>Range</li>
    <li> List Comprehension</li>
</ul>
'''))


li = []
print([(i**2) for i in range(1,4,-2)]) 
print([(i**-2) for i in range(1,4,2)]) 
print([(i**-0.2) for i in range(1,4,2)]) 
print(((-1)**0.5).real)
print(((-1)**0.5).imag)
print(((-1)**-0.5))


'''
Answers:
[]
[1.0, 0.1111111111111111]
[1.0, 0.8027415617602307]
6.123233995736766e-17
1.0

'''


# <a id='Session_4'>Session-4</a>

# In[38]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display,HTML
import copy

# Problem solving

display(HTML('''
<h3>Problem solving</h3>
'''))

# https://jovian.ai/shvmgrg98/ml-c41-python-crash-course-session-4


# In[2]:

######################## START: Taking Input in Python  ########################
# principal = float(input("Enter the principal amount:"))
# rate = float(input("Enter rate in percentage:"))
# time = float(input("Enter time in years:"))
# print("Simple Interest:",principal*rate*time/100)
######################## END: Taking Input in Python  ########################

# In[3]:


li = []
for i in range(2000,3201):
    if(i%7==0):
        if(i%5!=0):
            li.append(str(i))
for i in li[:-1]:
    print(i,end=",")
print(li[-1])
print(",".join(li))  #also correct


print(",".join([str(i) for i in range(2000,3201) if(i%7==0 and i%5!=0)]))



# In[4]:

######################## START: Taking Input in Python  ########################
# factorial
# #Method 1:
# # factorial = 1
# # num = int(input("Enter the number:"))
# # if(num==0 or num==1):
# #     print(1)
# # else:
# #     for i in range(2,num+1):
# #         factorial*=i
# #     print(factorial)

# # Method 2:
# n_list = range(1,int(input("Enter a number:"))+1)
# from functools import reduce
# factorial = reduce(lambda x,y:x*y,n_list)
# print(factorial)
######################## END: Taking Input in Python  ########################

# In[5]:


import sys
sys.version

######################## START: Taking Input in Python  ########################
# # In[6]:


# string = input("Enter a string:").split(",")
# print(",".join(sorted(string)))


# # In[7]:


# string = list(input("Enter a string:"))
# print("No of alphabets:",len(list(filter(lambda x: x.isalpha(), string))))
# print("No of digits:",len(list(filter(lambda x: x.isdecimal(), string))))
# print("No of characters:",len(list(map(lambda x: x.isdecimal(), string))))
# print("Truth value of characters:",(list(map(lambda x: x.isdecimal(), string))))
# print("All digits from input string:",(list(filter(lambda x: x.isdecimal(), string))))

# li = input("Enter the list:").split(",")
# print([int(i)**2 for i in li if(int(i)%2!=0)])


######################## END: Taking Input in Python  ########################

# In[8]:


import sys
sys.stdout.write("hello\n")
sys.stdout.write("hello")


# In[9]:


#print(",".join([str(i**2) for i in [int(i) for i in input("Enter the list:").split(",")] if(i%2!=0)]))


# In[10]:


li = [1,2,3,4,5,6,7]
print(type(li[0]))

# method 1 (looping)
for i in li:
    print(str(i),end=",")    
print()

# method 2 (list comprehension)
print([str(i) for i in li])

# method 3 (lambda)
print([i() for i in (lambda x=x:str(x) for x in li)])


# In[11]:


# yield vs return

def simpleGeneratorFun():
    yield 1
    return 2   # execution stops before return
    yield 3

# Driver code to check above generator function
for value in simpleGeneratorFun(): 
    print(value)



# In[12]:


def print_():
    return 5
display(print_())
print(print_())
display(3)


# In[39]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




