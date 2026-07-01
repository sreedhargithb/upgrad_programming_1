#!/usr/bin/env python
# coding: utf-8

# ## Contents:
# 
# 0. [Installing_packages](#Installing_packages)<br>
# 1. [Data_structures](#Data_structures)<br>
# 2. [OOP](#OOP)<br>
# 3. [Control_structures_and_loops](#Control_structures_and_loops)<br>
# 4. [Python_basics__variables_data_types_functions](#Python_basics__variables_data_types_functions)<br>

# <a id='Installing_packages'>To install pip / pip3 packages</a>

# In[76]:


import subprocess
import sys
# import sklearn

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    #subprocess.run(["pip","install","scikit-learn==0.22.2","--user"])
    #print(sklearn.__version__) # 1.1.3

install("scikit-learn")
install("tensorflowjs")
'''
Notes:
- Use "imbalanced-learn" instead of imblearn
- Use "tensor-sensor" instead of tsensor
'''

subprocess.run(["pip","list"])


# <a id='Data_structures'>Data_structures</a>
# 
# In the last session, we had a look at the String data type which could be used to store character values. Now let's try building an application which records a customer's detail and assign it a unique user id number generated randomly.

# In[ ]:





# In[ ]:





# In[77]:


from random import seed
from random import randint

# generate a three digit random integers for user id
value = randint(100, 999)
print(value)


# In[78]:


######################## START: Taking Input in Python  ########################
#We would ask the user for his name and mobile number
# name= input("What's your name ")
# number= input("What's your number ")

# #Now if I want to store these values together I can create a list data type
# name= input("What's your name ")
# number= input("What's your number ")
# value = randint(100, 999)
######################## END: Taking Input in Python  ########################


# In[ ]:





# Hmm, that works. But what if I would like to add some security and make user id un-changeable or as we call it in programming - immutable, in that case, we'll use a tuple.
# 
# So in this session, we are going to learn more about another new word or a data type actually which is called "Tuple"
# 

# <h1 style="color:Brown"> Tuples </h1>
# 
# - Tuples are an ordered sequence of mixed data types.
# - Tuples are written as comma-separated  elements within parenthesis
# 

# In[79]:


t = ("disco", 12, 4.5)
print(t)


# In[80]:


print(type(t))


# <h3> Tuples can be defined is various ways </h3>

# #### A tuple can be defined without using parenthesis

# In[81]:


sample_tuple = 1,2,3,4

print(sample_tuple)


# #### Single value tuple

# In[82]:


sample_tuple = 1,
print(sample_tuple)


# In[83]:


sample_tuple = (1,)
print(sample_tuple)


# In[84]:


sample_tuple1 = 1   # This is not a tuple


# In[85]:


sample_tuple2 = (1) # This is not a tuple


# ### Indexing in tuples

# In[86]:


t = ("Mumbai", 84, "Python",)


# In[87]:


# gives the element at index location 1
print(t[1])


# In[88]:


# gives the last element from tuple
print(t[-1])


# In[ ]:





# ### Slicing

# In[89]:


# Slicing first 3 elements from t

t = ("Seattle", 84, "Python", 5, 2, 1)

print(t[0:3])


# In[90]:


# Slicing last 2 elements from t

t = ("Seattle", 84, "Python", 5, 2, 1)

print(t[-2:])


# In[91]:


# no. of elements in tuple t
t = ("Seattle", 84, "Python", 5, 2, 1)
print(len(t))


# #### Cancatenating tuples

# In[92]:


tup1 = ("This", "is", "Session", 2)
tup2 = ("on", "Tuples")

# Adding contents of tup2 to tup1 and storing in tup3

tup3 = tup1 + tup2

print(tup3)


# #### sum() - min() - max()

# In[93]:


t = (2, 4, 3, 5, 7)
print(sum(t))


# In[94]:


t = (2, 4, 3, 5, 7)
print(min(t))


# In[95]:


t = (2, 4, 3, 5, 7)
print(max(t))


# ### Immutability of tuples

# In[96]:


# t = ("USA", 4, 3, "Disco", 7.5)

# t[3] = "Hard Rock"
# # shows error
# #TypeError: 'tuple' object does not support item assignment


# In[97]:


new_t = t[0:3] + ("Hard Rock",) + t[4:]
print(new_t)


# ### Sorting a tuple

# In[98]:


t = (2,3,6,4,8,5)

print(sorted(t))


# In[99]:


x = sorted(t)

print(tuple(x))


# In[100]:


# Nested Tuples

t = (1,5,"Disco", ("Python", "Java"))

# Access "Java" from the nested tuple
print(t[3][1])


# ### Packing and Unpacking In Tuples

# In[101]:


t = (1,2,3,4)     # Packing
(a,b,c,d) = t     # Unpacking
print (a)


# #### dir() - to view the attributes or methods of an object

# In[102]:


t = ()
print(dir(t))


# <h1 style="color:Brown"> Lists </h1>
# 
# - Lists are an ordered sequnce of mixed data types.
# - Lists are written as comma-separated  elements within square brackets

# In[103]:


L = ["USA", 23, 6, "New York"]

print(L)


# In[104]:


# Nested List
L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]

print(L)


# In[105]:


# Indexing
L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]

print(L[0])


# In[106]:


print(L[-3])


# In[107]:


print(L[-1][-1])


# In[108]:


# Slicing

print(L[0:3])


# In[109]:


#Membership in Lists

L1 = [1,2,3,4]
print(1 in L1)  # True
print (8 not in L1)  # True


# In[110]:


# List Concatanetion

L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]
new_L = L + [5, 8]
print(new_L)


# In[111]:


# Replace "Biology" with "Physics"

L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]

L[1] = "Physics"

print(L)


# In[112]:


# extend()

L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]
L.extend([5, 8])
print(L)


# In[113]:


# append()

L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]
L.append([5, 8])
print(L)


# In[114]:


# del Command

L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]
del L[0]
print(L)


# In[115]:


del L[0:2]

print(L)


# In[116]:


# pop()

L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]

L.pop()

print(L)


# In[117]:


# remove()

L = ["Chemistry", "Biology", [1989, 2004], ("Oreily", "Pearson")]

L.remove("Chemistry")

print(L)


# In[118]:


# Sorting Lists

l = [32, 24, 65, 9]
l.sort()
print(l)


# In[119]:


l.sort(reverse= True)
print(l)


# #### Difference between sort and sorted
# 

# In[120]:


A = ["Orange", "Strawberry", "Mango"]
B = A.sort()
print(A)
print(B)


# In[121]:


A = ["Orange", "Strawberry", "Mango"]
C = sorted(A)

print(A)
print(C)


# #### Shadow Copying

# In[122]:


A = ["Orange", "Strawberry", "Mango"]
B = A

A[0] = "Apple"


# In[123]:


print(A)


# In[124]:


print(B)


# In[125]:


A = ["Orange", "Strawberry", "Mango"]
B = A[:]    #Note 'A[:]' is used to call all the values stored in A
#Now B has all the values of A but isn't a shadow copy of A
#Shadow copy is basically assigning multiple labels to a single reference point or memory location

A[0] = "Apple"


# In[126]:


print(A)


# In[127]:


print(B)


# Now that we know how to store multiple items together let us build another app that could store userid of our customers, which were randomly generated.
# 
# We have the following requirements from the app -
# 1. It shouldn't store duplicate values in the data-set
# 2. Since the userid are randomly generated the order doesn't matter
# 3. We need mutability i.e. if we want to delete a particular value we should be able to do so
# 4. We might frequently want to check whether a user-id is part of the existing data set so it should be able to perform this operation fast
# 
# For this use case, we could either use a unique function and maintain our list, but that would be comparatively time-consuming, so let's look at another data type provided by Python, which would be the best fit here.
# 

# <h1 style="color:Brown"> Sets </h1>
# 
# - Sets are a type of collection like lists and tuples, storing mixed data.
# - Sets are enclosed within curly brackets and elements are written as comma-separated.
# - Sets are unordered
# - Sets do not allow duplicates

# In[128]:


l = [1,3,2,4,5,5]

set_l = set(l)

print(set_l)


# In[129]:


print(len(set_l))


# In[130]:


a = {1, 2, 3, 4, 5}

a.add("Australia")

print(a)


# In[131]:


a = {1, 2, 3, 4, 5, "Australia"}
a.remove("Australia")
print(a)


# In[132]:


# Set Operations

A = {0, 2, 4, 6, 8}
B = {1, 2, 3, 4, 5}


# In[133]:


print(A | B) # Union
print (A.union(B))


# In[134]:


print(A & B) # Intersection
print(A.intersection(B))


# In[135]:


print(A - B) # Difference
print(A.difference(B))  # only A elements


# In[136]:


print(A ^ B) # Symmetric Difference


# In this session, we are going to talk about dictionaries. In Python, we use dictionaries to store key-value pairs.
# Here is an example -
# Let's take the example of a customer as seen earlier.
# Now a customer could have multiple attributes such as
# 1. Name
# 2. Email
# 3. Userid
# 4. Address
# 5. Phone number
# 
# Now each of these attributes would be our key and corresponding we'll have values against each on them
# 
# With a dictionary, we can store a bunch of key-value pairs, let us now see have a look at the 'Dictionary' data type in Python

# <h1 style="color:Brown"> Dictionaries </h1>
# 
# - A dictionary stores element as keys and values pairs.
# - The key is like an index, it's is always unique and immutable.
# - The values are the objects that contain information.
# - Values are accessed using their keys.
# - Each key is followed by a value separated by a colon.
# - The values can be immutable, mutable, and duplicates.
# - Each key and value pair is separated by a comma enclosed inside curly brackets.

# In[137]:


# Creating a dictionary

d = {"India" : "INR", "USA" : "USD", "France" : "Euros"}

# Access value using keys

print(d["USA"])


# In[138]:


# Replace the value for a key in a dictionary

d["USA"] = "$"

print(d)


# In[139]:


# Insert new key value pair into a dictionary

d["Japan"] = "Yen"

print(d)


# In[140]:


# Deleting a key value pair

del d["France"]

print(d)


# In[141]:


# Sorting a dictionary

print(sorted(d))


# In[142]:


# values() Method

d = {"India" : "INR", "USA" : "USD", "France" : "Euros"}

print(d.values())


# In[143]:


# Keys() Method

print(d.keys())


# In[144]:


# Update() Method

d.update({'USA':'$'})
print(d)


# <a id='OOP'>OOP</a>

# In[145]:


get_ipython().run_cell_magic('html', '', '<h1 style = "color : Blue"> i. Creating Classes and Objects </h1>\n\n<h2 style = "color : Brown">Defining a class</h2>\n\n- length and breadth as attributes\n- __init__() - constructor of class\n- self parameter - refers to the newly created instance of the class.\n- attributes length and breadth are associated with self-keyword to identify them as instance variables\n\nclass Rectangle :\n    def __init__(self):\n        self.length = 10\n        self.breadth = 5\n\n- create the object by calling name of the class followed by parenthesis.\n- print the values using dot operator\n\nrect = Rectangle()\nprint("Length = ",rect.length, "\\nBreadth = " ,rect.breadth)\n\n<h2 style = "color : Brown">Parametrised Constructor</h2>\n\n#- parametrised constructor - dynamically assign the attribute values during object creation\n\nclass Rectangle :\n    def __init__(self, length, breadth):\n        self.length = length\n        self.breadth = breadth\n\nrect = Rectangle(10, 5)\nprint("Length = ",rect.length, "\\nBreadth = " ,rect.breadth)\n\n<h1 style = "color : Blue">ii. Class Variable and Instance variables\n\nclass Circle :\n    pi = 3.14\n    def __init__(self, radius):\n        self.radius = radius\n\ncircle_1 = Circle(5)\nprint("Radius = {} \\t pi = {}".format(circle_1.radius,circle_1.pi))\n\ncircle_2 = Circle(2)\nprint("Radius = {} \\t pi = {}".format(circle_2.radius,circle_2.pi))\n\nCircle.pi = 3.1436\n\ncircle_1 = Circle(5)\nprint("Radius = {} \\t pi = {}".format(circle_1.radius,circle_1.pi))\n\ncircle_2 = Circle(2)\nprint("Radius = {} \\t pi = {}".format(circle_2.radius,circle_2.pi))\n\n<h1 style = "color : Blue">iii. Adding a method to class</h1>\n\n- calculate_area() - retutns the product of attributes length and breadth\n- self - identifies its association with the instance\n\nclass Rectangle :\n    def __init__(self, length, breadth):\n        self.length = length\n        self.breadth = breadth\n\n    def calculate_area(self):\n        return self.length * self.breadth\n\nrect = Rectangle(10, 5)\nprint("Length = ",rect.length, "\\nBreadth = " ,rect.breadth, "\\nArea = ", rect.calculate_area())\n\n<h2 style = "color : Brown"> Significance of self:</h2>\n\n- The attributes length and breadth are associated with an instance.\n- Self makes sure that each instance refers to its own copy of attributes\n\nnew_rect = Rectangle(15, 8)\nprint("Length = ",new_rect.length, "\\nBreadth = " ,new_rect.breadth, "\\nArea = ", new_rect.calculate_area())\n\nprint("Length = ",rect.length, "\\nBreadth = " ,rect.breadth, "\\nArea = ", rect.calculate_area())\n\n<h1 style = "color : Blue">iv. Class Method and Static Method\n\nclass Circle :\n    pi = 3.14\n    def __init__(self, radius):\n        self.radius = radius\n\n    # Instance Method\n    def calculate_area(self):\n        return Circle.pi * self.radius\n\n    # Class Method - I cannot access - radius\n    @classmethod\n    def access_pi(cls):\n        pi = 3.1436\n        return pi\n\n    # Static Method -  I cannot access - pi and radius\n    @staticmethod\n    def circle_static_method():\n        print("This is circle\'s static method")\n\ncir = Circle(5)\n\n# Calling methods\n\nprint(cir.calculate_area())\n\nprint(Circle.access_pi())\n\nCircle.circle_static_method()\n\n<h1 style = "color : Blue">v. Inheritance and Overriding\n\nclass Shape :\n\n\n    def set_color(self, color):\n        self.color = color\n\n    def calculate_area(self):\n        pass\n\n    def color_the_shape(self):\n        color_price = {"red" : 10, "blue" : 15, "green" : 5}\n        return self.calculate_area() * color_price[self.color]\n\nclass Circle(Shape) :\n    pi = 3.14\n    def __init__(self, radius):\n        self.radius = radius\n\n    # overriding\n    def calculate_area(self):\n        return Circle.pi * self.radius\n\nc = Circle(5)\nc.set_color("red")\nprint("Circle with radius =",c.radius ,"when colored", c.color,"costs $",c.color_the_shape())\n\nclass Rectangle(Shape) :\n    def __init__(self, length, breadth):\n        self.length = length\n        self.breadth = breadth\n\n     # Overriding user defined method\n    def calculate_area(self):\n        return self.length * self.breadth\n\n    # Overriding python default method\n    def __str__(self):\n        return "area of rectangle = " + str(self.calculate_area())\n\nr = Rectangle(5, 10)\nr.set_color("blue")\nprint("Rectangle with length =",r.length ," and breadth = ",r.breadth ,"when colored", r.color,"costs $",r.color_the_shape())\n\nprint(r)\n\nfrom distutils.log import debug\nimport numpy as np\nimport pandas as pd\nfrom tabulate import tabulate\nimport webbrowser\nfrom IPython.display import display,HTML\nimport subprocess\nimport sys\nimport matplotlib.pyplot as plt\nfrom flask import Flask\n\n\ndef install(package):\n    subprocess.check_call([sys.executable, "-m", "pip", "install", package])\ninstall("statsmodels")\n\n\nproduct_category = np.array([\'Furniture\', \'Technology\', \'Office Supplies\'])\nsales = np.array([4110451.90, 4744557.50, 3787492.52] )\n\nplt.bar(product_category, sales,color=\'green\')\nplt.show()\nplt.show()\n\ncars_per_cap = [809, 731, 588, 18, 200, 70, 45]\ncountry = [\'United States\', \'Australia\', \'Japan\', \'India\', \'Russia\', \'Morocco\', \'Egypt\']\ndrives_right = [True, False, False, False, True, True, True]\ndata = {"cars_per_cap": cars_per_cap, "country": country, "drives_right": drives_right}\ndisplay(pd.DataFrame(data))   # == print(pd.DataFrame(data)), in this case\n\nprint(tabulate(pd.DataFrame(data),headers=\'keys\', tablefmt=\'psql\'))\n#s = (pd.DataFrame(data).to_html())\n\ndef flask_app():\n    app = Flask(__name__)\n\n    @app.route(\'/\')\n    def scrape_and_reformat():\n        print(pd.DataFrame(data))\n        return (pd.DataFrame(data).to_string() + pd.DataFrame(data).to_html() + pd.DataFrame(data).to_html())\n\n    @app.route(\'/next\')\n    def next():\n        #return (pd.DataFrame(data))    # error - TypeError: The view function did not return a valid response. The return type must be a string, dict, list, tuple with headers or status, Response instance, or WSGI callable, but it was a DataFrame.\n        return (pd.DataFrame(data).to_string() + pd.DataFrame(data).to_string())\n\n    #if __name__ == \'__main__\':\n    webbrowser.open(\'http://127.0.0.1:5000\')\n    webbrowser.open(\'http://127.0.0.1:5000/next\')\n    app.run(debug = True, use_reloader=False)\nflask_app()\n')


# In[146]:


######################## START: Taking Input in Python  ########################

# # # Ice Cream Sundae - Ordering Menu

# # You are expected to build an interactive application to order ice cream Sundays in an ice-cream parlour.
# # You are expected to use the concepts you learnt from the Object-Oriented programming session. The complete building process is divided into questions given below after you solve all the questions you will the complete ordering application.

# # Let's get started!!


# # ### Question

# # Declare a class "ice_cream". It needs to have the following constants and instance methods defined.

# # 1. Radius of a small scoop (r_small = 1.5)
# # 2. Radius of large scoop (r_large = 2.5)
# # 3. Value of pie (pi = 3.14)
# # 4. An instance method "flavour" - it should print enter your flavour.

# # Given below is the example of a sample class and its methods and variables for reference:

# class chocolate:

#     chocolate_length=10
#     chocolate_breadth=2

#     def area(self):
#         print("Enter your favourite chocolate")

# # Declare class here:
# class ice_cream:
#     # Declare class variables:
#     r_small = 1.5
#     r_large = 2.5
#     pi = 3.14

#     # Declare instance methods:
#     def flavour(self):
#         print("Enter your flavour")

# # Call the class:
# order = ice_cream()

# order.flavour()

# ### Question

# # Modify the class "ice_cream" to add a method which can calculate the cost of the Ice cream based on its size.

# # Cost of ice cream is 0.5$ per unit volume. Take the input from the user about what size of ice cream scoops they want small or large. Based on that calculate the volume of the scoop and use the volume and the cost per volume to calculate the cost of ice cream.

# # Hint: Declare a class similar to example above. Add new function that calculates the cost depending on the scope size. Use If-else statement to calculate ice-cream cost.

# class ice_cream:
#     r_small = 1.75
#     r_large = 2.5
#     pi = 3.14

#     def flavour(self):
#         print("Enter your flavour")
#         size = input("Would you like a small scoop or a large scoop (enter s/l)")
#         cost = self.cost_ice_cream(size)
#         print ("The cost of ice cream is ", cost)

#     def cost_ice_cream(self, size):
#         if size=="s":
#             vol = 4/3 *(self.pi)* (self.r_small**3)
#             cost = vol * 0.5
#             return cost
#         elif size=="l":
#             vol = 4/3 *(self.pi)* (self.r_large**3)
#             cost = vol * 0.5
#             return cost
#         else:
#             print("Please enter a valid size")

# # Call the function:
# order = ice_cream()
# order.flavour()

# ### Question

# # In the above function find a way to round up the cost to the next integer value.

# # Hint: In the same class as above, add a additional element to the cost variable that rounds up the cost to the next nearest integer.
# # For example: if the cost is 10.41, the output should be 11$
# # ##### For rounding up an integer value you could use either the ceil() or round()

# import math

# class ice_cream:
#     r_small = 1.75
#     r_large = 2.5
#     pi = 3.14

#     def flavour(self):
#         print("Enter your flavour")
#         size = input("Would you like a small scoop or a large scoop (enter s/l)")
#         cost = self.cost_ice_cream(size)
#         print ("The cost of ice cream is ", math.ceil(cost))

#     def cost_ice_cream(self, size):
#         if size=="s":
#             vol = 4/3 *(self.pi)* (self.r_small**3)
#             cost = vol * 0.5
#             return cost
#         elif size=="l":
#             vol = 4/3 *(self.pi)* (self.r_large**3)
#             cost = vol * 0.5
#             return cost
#         else:
#             print("Please enter a valid size")

# # Call the function:
# order = ice_cream()
# order.flavour()

# ### Question

# # Modify the flavour function to give the options of available flavours and take as input the choice of the customer.

# # The available options are Vanilla, Chocolate, Butterscotch, Blue_berry.

# # Hint: Add a new function that asks the user to input the choice of flavour.

# import math

# class ice_cream:
#     r_small = 1.5
#     r_large = 2.5
#     pi = 3.14

#     def flavour(self):
#         print ("Available flavours of ice cream are Vanilla, Chocolate, Butterscotch, Blue_berry")
#         flv = input("Which flavour of ice cream would you like ")
#         size = input("Would you like a small scoop or a large scoop (enter s/l)")
#         i_cost = self.cost_ice_cream(size)
#         print ("The cost of ice cream is ", math.ceil(i_cost))

#     def cost_ice_cream(self, size):
#         if size=="s":
#             vol = 4/3 *(self.pi)* (self.r_small**3)
#             cost = vol * 0.5
#         elif size=="l":
#             vol = 4/3 *(self.pi)* (self.r_large**3)
#             cost = vol * 0.5
#         else:
#             print("Please enter a valid size")
#         return cost

# order = ice_cream()
# order.flavour()

# ### Question

# # Build a new class called "toppings". It should have all the functionality of the ice_cream class.

# # The toppings class will also have a method which will take as input the choice of toppings that the customer wants.
# # The available choices of toppings are: Hot_fudge, Sprinkles, Caramel, Oreos, Nuts

# # Hint: Create a new class that ask the user to choose one or more toppings

# class toppings(ice_cream):

#     def sel_toppings(self):
#         print ("Available toppings are Hot_fudge, Sprinkles, Caramel, Oreos, Nuts")
#         top = input ("Enter any number of toppings of your choice separated by a comma: ")
#         top_list = top.split(",")
#         print ("The toppings you selected are : ",top_list)

# sundae = toppings()
# sundae.sel_toppings()

# ### Question

# # Add a method to calculate the cost of selected toppings, given the cost of each of the topping is 2$.

# # Hint: Now in the class for toppings, add a function to calculate the cost per topping added.

# class toppings(ice_cream):

#     def sel_toppings(self):
#         print ("Available toppings are Hot_fudge, Sprinkles, Caramel, Oreos, Nuts")
#         top = input ("Enter any number of toppings of your choice separated by a comma: ")
#         top_list = top.split(",")
#         t_cost = self.top_cost(top_list)
#         print ("The cost for selected toppings is ",t_cost)

#     def top_cost(self, top_list):
#         cost = len(top_list) * 2
#         return cost

# sundae = toppings()
# sundae.sel_toppings()

# ### Question:
# # Now you have all the functionality needed to create the ordering menu.

# # 1. An order can be for simply Ice Cream or an Ice Cream sundae.
# # 2. There can be multiple items in an order.
# # 3. Calculate the cost of each order placed.

# # Hint: Club both the class you have created above and finally create an Ice-cream ordereing machine that display a welcome message: "Welcome to  Ice Cream parlour". Asks the user if he/she wants and ice cream or ice cream-sundae. Ask the choice of flavour and toppings and returns the total cost. Dont forget to ask if the user wants another item after he finishes ordering the first one!

# import math

# class ice_cream:
#     r_small = 1.5
#     r_large = 2.5
#     pi = 3.14

#     def flavour(self):
#         print ("Available flavours of ice cream are Vanilla, Chocolate, Butterscotch, Blue_berry")
#         flv = input("Which flavour of ice cream would you like ")
#         size = input("Would you like a small scoop or a large scoop (enter s/l)")
#         i_cost = self.cost_ice_cream(size)
#         if order_type == "i":
#             print ("The cost of the ice Cream is ", math.ceil(i_cost))
#         return math.ceil(i_cost)

#     def cost_ice_cream(self, size):
#         if size=="s":
#             vol = 4/3 *(self.pi)* (self.r_small**3)
#             cost = vol * 0.5
#         elif size=="l":
#             vol = 4/3 *(self.pi)* (self.r_large**3)
#             cost = vol * 0.5
#         else:
#             print("Please enter a valid size")
#         return cost


# class toppings(ice_cream):

#     def sel_toppings(self):
#         print ("Available toppings are Hot_fudge, Sprinkles, Caramel, Oreos, Nuts")
#         top = input ("Enter any number of toppings of your choice separated by a comma: ")
#         top_list = top.split(",")
#         i_cost = self.flavour()
#         t_cost = self.top_cost(top_list)
#         print ("The cost for sunday is ",t_cost+ i_cost)

#     def top_cost(self, top_list):
#         cost = len(top_list) * 2
#         return cost


# print ("Welcome to the upGrad Ice Cream parlour")

# while True:
#     print ("Would like an ice cream(i) or a Sundae(s)?")
#     order_type = input("Enter your response (i/s)")
#     if order_type == "i":
#         order = ice_cream()
#         order.flavour()
#     elif order_type == "s":
#         sundae = toppings()
#         sundae.sel_toppings()
#     else:
#         print ("Enter a valid choice")

#     print ("Would like to order anything else")
#     more = input ("Enter your response as (y/n)")

#     if more == "n":
#         break

# # Great job! You have learned the skills to create a real-world application, in case of any doubts please feel free to put in up on the discussion forum or reach out to your TA(teaching assistant).
# #####Happy Learning!!

######################## END: Taking Input in Python  ########################


# In[147]:


get_ipython().run_cell_magic('html', '', '<style type=\'text/css\'>\n.CodeMirror{\n    font-size: 12px;\n}\n\ndiv.output_area pre {\n    font-size: 12px;\n}\n</style>\n\n#!jt -l\n\'\'\'\nAvailable Themes:\n   chesterish\n   grade3\n   gruvboxd\n   gruvboxl\n   monokai\n   oceans16\n   onedork\n   solarizedd\n   solarizedl\n\'\'\'\n# https://github.com/dunovank/jupyter-themes\n# !jt -t gruvboxd -f roboto -fs 12\n!jt -r\n\n# Reference:\n    # https://stackoverflow.com/questions/42449814/running-jupyter-notebook-in-a-virtualenv-installed-sklearn-module-not-available\nimport sys\nprint(sys.version)\n\n#Classes overview (instance methods, class methods, static methods)\nfrom IPython.display import display,HTML\ndisplay(HTML(\'\'\'\n<h3>Classes</h3>\n<h3>Methods vs Functions</h3>\n<h4>Instance methods</h4>\n<h4>Class methods</h4>\n<h4>Static methods</h4>\n<h3>Inheritance and Overriding</h3>\n\'\'\'))\n\n\n\nimport subprocess\nimport sys\nprint(sys.version)\ndef install(package):\n    subprocess.check_call([sys.executable, "-m", "pip", "install", package])\n\ninstall("statsmodels")  # install "fancyimpute" from Anaconda Powershell prompt\n\n# Jupyter themes\n# https://towardsdatascience.com/customize-your-jupyter-notebook-theme-in-2-lines-of-code-fc726cea1513#:~:text=By%20default%2C%20Jupyter%20Notebook%20uses,the%20theme%20of%20the%20notebook.\n# Screens of the available themes are also available in this Github repository:\n#          https://github.com/dunovank/jupyter-themes/tree/master/screens\n# !jt -r    # to restore the default jupyter theme\n!jt -l\n#!jt -t gruvboxd\n\n######################## START: Taking Input in Python  ########################\n# sum_ = 0\n# while True:\n#     i = input("Enter a number, or press enter to exit: ")\n#     if not i:\n#         break\n#     sum_ += int(i)\n# print(sum_)\n######################## END: Taking Input in Python  ########################\n\n#Classes overview (instance methods, class methods, static methods)\n\n\ndef classes():\n    class rectangle:\n        \'\'\'\n        Parameterised constructor:\n        dynamically assign the attributes during object creation\n        \'\'\'\n        # class variables\n        length = 54\n        breadth = 43\n        pi = 3.14\n        def __init__(self,length,breadth):\n            self.length = length\n            self.breadth = breadth\n        print(length)\n        l = 40\n        b = 30\n        def area(self):\n            pi = 3.1234567\n            print("self.pi:",self.pi, "\\npi:",pi)\n            return self.b*self.l\n    print(rectangle(0,0).area())\n    # 0,0 above is to only fill up the parameters\n    # (number of parameters here should be equal to\n    # the number of parameters in init constructor of the class\n    # (other than self))\n    rect2 = (rectangle(56,67))\n    print(rect2.length,rect2.breadth)  # instance variables\n    print(rectangle.pi, rectangle(0,0).area())\n#classes()\n\ndef classes_2():\n    class Circle:\n        pi = 3.141592653589\n        def __init__(self,radius):\n            self.radius = radius\n\n        # instance method\n        def calculate_area(self):\n            return Circle.pi * (self.radius**2)\n        # class method (cannot access radius)\n        @classmethod\n        def access_pi(self):\n            pi = 3.14\n            return pi\n        # static method - cannot access - pi and radius\n        @staticmethod\n        def circle_static_method():\n            return ("This is circle\'s static method")\n    print(Circle(5).calculate_area()) # required parameter here\n    print(Circle.access_pi())\n    print(Circle.circle_static_method())\n#classes_2()\n\nclass A :\n    x = 10\n    def __init__(self, y,z):\n        self.y = y\n        self.z = z\n\n    def update_y(self):\n        self.y = self.y * self.z\n\nA1 = A(3,4)\nA2 = A(5,6)\nA.x = 30\nprint(A1.y + A2.x)\n\n# Overriding:\n\nclass Shape:\n    def set_color(self,color):\n        self.color = color\n\n    def calculate_area(self):\n        return 2**2\n\n    def color_the_shape(self):\n        color_price = {"red":10,"blue":15,"green":5}\n        return self.calculate_area() + color_price(self.color)\n\n\n\nclass Circle(Shape):\n    pi = 3.141592653589\n    def __init__(self,radius):\n        self.radius = radius\n\n    # Overriding (changing the functionality of this function in the parent class)\n    def calculate_area(self):\n        return Circle.pi * (self.radius**2)\n\nprint(Shape().calculate_area())\nprint(Circle(5).calculate_area())\n\nc = Circle(5)\n\n#Inheritance\nc.set_color("red")\nprint(c.pi,c.radius,c.color)\n\n\n#Inheritance (continued):\nclass Rectangle(Shape):\n\n    def __init__(self,length,breadth):\n        self.length = length\n        self.breadth = breadth\n\n    #Overriding\n    def calculate_area(self):\n        return self.length*self.breadth\n\n    #Overriding the default Python method\n    def __str__(self):\n        return "area of rectangle:" + str(self.calculate_area())\n\nr = Rectangle(3,6)\n\nr.set_color("blue")\nprint(r.length, r.breadth, r.color)\n\n')


# <a id='Control_structures_and_loops'>Control_structures_and_loops</a>

# In[148]:


# ######################## START: Taking Input in Python  ########################

# # Control_structures_and_loops

# ### Question


# # A word is said to belong to the red team if it has the letter r in it.
# # Write an 'if' statement to check whether the given word belongs to the red team or not.


# word = input("Enter a word in small caps ")
# is_red = 0
# for char in word:
#     if char == "r":
#         print ("The word belongs to red team")
#         is_red = 1
#         break

# if is_red == 0:
#     print ("The word does not belong to the red team")


# ### Question

# In continuation to previous example, if the word has the letter 'b', it belongs to the blue team. And if the word has both 'r' and 'b' then first letter gets a precedence. And if the word does not have the letter 'r' or 'b', it does not belong to any team.
# ######Example: (word,output)
# *   rabbit = red team
# *   brand =  blue team
# *   dog   =  no team


# Write a condition for this decision.


# word = input("Enter a word in small caps ")
# is_red = 0
# is_blue = 0
# for char in word:
#     if char == "r":
#         print ("The word belongs to red team")
#         is_red = 1
#         break
#     elif char == "b":
#         print ("The word belongs to blue team")
#         is_blue = 1
#         break

# if is_red == 0 and is_blue == 0:
#     print ("The word does not belong to any team")

# ### Question

# Write a program to print all the numbers which are divisible by 17 between 1 and a given number 'n'.

# num = int(input("Enter the upper limit "))

# i=1

# while i<= num:
#     if i% 17 == 0:
#         print(i)
#     i = i + 1

# ### Question

# Write a program to add 13 to all the elements of a given numeric list.
# ######Example -
# * Input list = [1,5,27]
# * Output list = [14,18,40]

# sample_list = [6, 98, 34, 45, 64, 7, 2, 5, 78, 90, 324, 111, 657, 438]

# # Write solution code here

# sol_list = []
# for i in sample_list:
#     sol_list.append(i+13)

# print (sol_list)

# ### Question

# In continuation to the example above, execute the same operation on the same sample list using list comprihension.

# sol_list = [x + 13 for x in sample_list]
# sol_list

# ### Question
# This question has two tasks:
# * Task 1 - Find the factors of all the numbers in a given list.
# * Task 2 - Add the numbers and their factors to a dictionary where the key will be the numbers from the list, and the values will be the factors of the respective number.

# Example -
# ##### Sample output :
# #####{6: [1, 2, 3, 6],
# 98: [1, 2, 7, 14, 49, 98],
# ##### 34: [1, 2, 17, 34],}

# sample_list = [6, 98, 34, 45, 64, 7, 2, 5, 78, 90, 324, 111, 657, 438]

# # write code here
# solu_dict = {}
# for num in sample_list:
#     fact_list = []
#     for i in range (1,num+1):
#         if num%i ==0:
#             fact_list.append(i)
#     solu_dict[num]=fact_list
# solu_dict

# ### Question

# Count the total number of unique characters in a given sentence.
# #####Example -
# * Input - 'total'
# * Output  - 'o', 'a', 'l'

# sample_sentence = "Pack my box with five dozen liquor jugs"

# # write code here

# unique_letter = set(sample_sentence)
# number = len(unique_letter)
# print(unique_letter)
# number

# ### Question

# Write a calculator application using functions, which will ask for the users input such as - arithmetic operation, number1, number 2, and return the output post applying the arithmetic operator

# #####Example -


# ```
# def calc(operation, x,y):
#     if operation == "add":
#         return x+y
# ```



# def calc(operation, x,y):
#     if operation == "add":
#         return x+y
#     if operation == "subs":
#         return x-y
#     if operation == "multi":
#         return x*y
#     if operation == "div":
#         return x/y
#     else:
#         print("Enter a valid operation")
#         return None

# print ("Enter the operation you want to execute")
# oper = input("The available operations are 'add', 'subs', 'multi', 'div' " )
# x = int (input ("Enter the first number"))
# y = int (input ("Enter the second number"))

# print("Answer is ", calc(oper, x, y))

# ### Question

# Write a code to ask the user to input three numbers and then print the largest one among the three.

# num = input("Enter three numbers :")

# num_list = num.split()

# a = int(num_list[0])
# b = int(num_list[1])
# c = int(num_list[2])

# print (a,b,c)

# if a>b:
#     if a>c:
#         print ("The largest number is ", a)
#     else:
#         print ("The largest number is ", c)
# else:
#     if b>c:
#         print ("The largest number is ", b)
#     else:
#         print ("The largest number is ", c)

# ### Question

# Check whether a number is perfect or not.

# A perfect number is a number that is sum of all of its positive divisors (excluding itself).
# For more info refer to this link - https://www.geeksforgeeks.org/perfect-number/

# n = int (input("Enter the number you want to check: "))

# sum = 0
# for x in range(1, n):
#     if n % x == 0:
#         sum += x

# if sum == n:
#     print ("The number is a perfect number")
# else:
#     print ("The number is not a perfect number")

# ### Question

# Add the elements in the two given lists using map and lambda functions.

# list1 = [5, 8, 9, 12]
# list2 = [10, 12, 8, 6]

# # write code here

# sol = list(map(lambda x, y : x+y, list1, list2))
# sol

# ### Question

# # Write a code to find the factorial of a number using reduce function.
# # #####Example -
# #  5!= 5 * 4 * 3 * 2 * 1=120.

# # n = int (input ("Enter a natural number "))

# # n_list = range(1,n+1)
# # from functools import reduce
# # factorial = reduce(lambda x,y: x*y, n_list)
# # factorial
# ######################## END: Taking Input in Python  ########################


# <a id='Python_basics__variables_data_types_functions'>Python_basics__variables_data_types_functions</a>

# In[149]:


from IPython.display import display,HTML
display(HTML('''
<h3>Python basics, variables, data types, functions</h3>
'''))

def trials_1():
    # tuples
    print(type(2))
    print(type('2'))
    x = 2,3,
    print(type(x))
    ###################################################
    # dictionaries
    # THESE ALL ARE DICTIONARIES
    d = {}
    d = {'a':1, 'b':2}
    for i in d:
        print(i)
    d = dict(a=1, b=2)
    print(type(d))
    print(d)
    ###################################################
    # exercise
    # List of order ID’s which are processed
    processed_orders = [1152, 1154, 1155, 1156, 1157, 1160, 1161, 1162, 1166, 1169, 1170, 1172, 1176, 1050, 1178, 1051, 1052, 1054, 1058, 1060, 1061, 1062, 1065, 1066, 1067, 1068, 1069, 1076, 1077, 1080, 1081, 1083, 1091, 1085, 1088, 1089, 1131, 1092, 1094, 1095, 1099, 1102, 1103, 1104, 1106, 1107, 1108, 1109, 1111, 1117, 1119, 1121, 1150, 1128, 1129, 1136, 1137, 1139, 1140, 1141, 1144, 1148, 1124]
    # List of order ID’s which are returned
    returned_orders = [1153, 1158, 1159, 1163, 1164, 1165, 1167, 1168, 1171, 1173, 1174, 1175, 1177, 1053, 1055, 1056, 1057, 1059, 1063, 1064, 1070, 1071, 1072, 1073, 1074, 1075, 1078, 1079, 1082, 1084, 1086, 1087, 1090, 1093, 1096, 1097, 1098, 1100, 1101, 1105, 1110, 1112, 1113, 1114, 1115, 1116, 1118, 1120, 1122, 1123, 1125, 1126, 1127, 1130, 1132, 1133, 1134, 1135, 1138, 1142, 1143, 1145, 1146, 1147, 1149, 1151]
    # Consider the information available in the above two lists and answer the question given below
    sum_orders = []
    for i in range(len(processed_orders)):
        sum_orders.append(processed_orders[i] + returned_orders[i])
    print(sum_orders[49])
    print(sorted(processed_orders)[:4])
    print(sorted(processed_orders + returned_orders)[:4])
    ###################################################
    # exercise - 2
    dic ={101: 43, 102: 25, 103: 43, 104: 31, 105: 26, 106: 28, 107: 29, 108: 43, 109: 25, 110: 22, 111: 22, 112: 25, 113: 30, 115: 45, 116: 23, 117: 29, 118: 28, 119: 30, 120: 28, 121: 42, 122: 39, 123: 29, 124: 42, 125: 43, 126: 42, 127: 40, 128: 27, 129: 23, 130: 30, 131: 37, 132: 20, 133: 36, 134: 27, 135: 27, 136: 22, 137: 28, 138: 23, 139: 45, 140: 39, 141: 29, 142: 33, 143: 39, 145: 34, 146: 26, 147: 30, 148: 38, 149: 29, 150: 24, 151: 28, 152: 34, 153: 42, 154: 29, 155: 23, 156: 31, 158: 25, 160: 45, 161: 42, 162: 27, 163: 24, 164: 20, 166: 24, 167: 28, 168: 20, 169: 33, 170: 34, 171: 37, 172: 45, 173: 35, 174: 23, 175: 44, 176: 27, 177: 30, 178: 26, 179: 27}
    print(sorted(list(dic.values()))[-1])
    print(sum(list(dic.values()))/len(dic))
    dic[104] = 27
    dic[140] = 27
    dic[164] = 27
    del dic[143]
    print(sum(list(dic.values()))/len(dic))
    ###################################################
trials_1()



def trials_2():
    ###################################################
    # comprehension:
        # list comprehension
        # dictionary comprehension
    ###################################################
    # exercise
    def func(x = 1 ,y = 2):
        z = x * y + x + y
        return z
    print(func(2, func(3)))
    print((101*99, 102*98))
    ###################################################
    #lambda, map, filter, reduce
    x = lambda x:x.upper()
    print(x("hello"))
    li = ([1,2,3,4,5,6,7])
    print([i() for i in (lambda x=x:x**2 for x in li)])
    # print(tuple([i() for i in (lambda x=x:x**2 for x in li)]))
    # print(set([i() for i in (lambda x=x:x**2 for x in li)]))
    # map
                #same above thing using map
    print(list(map(lambda x=x:x**2,li)))
    # filter
    print(list(filter(lambda x=x:x**2!=x,li)))
    print(list(map(lambda x=x:x**2!=x,li)))
    print(list(filter(lambda x=x:x**2,li)))
    #reduce
    from functools import reduce
    print(reduce(lambda x,y:x*y,li))
    ###################################################
trials_2()

import sys
print(sys.version)



# In[150]:


from datetime import datetime
import pytz

# Define the IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Get the current time in UTC
utc_now = datetime.now(pytz.utc)

# Convert the current time to IST
ist_now = utc_now.astimezone(ist)

# Print the current time in IST
print("Current Time in IST:", ist_now.strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




