#!/usr/bin/env python
# coding: utf-8

# ## Regular Expressions
# Regular expression is a set of characters, called as the pattern, which helps in finding substrings in a given string. The pattern is used to detect the substrings
# 
# For example, suppose you have a dataset of customer reviews about your restaurant. Say, you want to extract the emojis from the reviews because they are a good predictor os the sentiment of the review.
# 
# Take another example, the artificial assistants such as Siri, Google Now use information retrieval to give you better results. When you ask them for any query or ask them to search for something interesting on the screen, they look for common patterns such as emails, phone numbers, place names, date and time and so on. This is because then the assitant can automatically make a booking or ask you to call the resturant to make a booking.
# 
# Regular expressions are very powerful tool in text processing. It will help you to clean and handle your text in a much better way.
# 
# Online regex tool: https://regex101.com/
# 
# 
# ## Agenda:
# - Intro
# - Quantifiers (*,+,?,{})
# - Anchors (^, $)
# - Wildcard (.)
# - Character sets ([..])
# - Grouping (pipe operator)
# - Meta sequences
# - Greedy vs non-greedy / lazy regex (Greedy followed by "?")
# - re.compile()

# ### Let's import the regular expression library in python.

# In[ ]:


import re


# Let's do a quick search using a pattern.

# In[ ]:


for i in [
     re.search('Ravi', 'Ravi is an exceptional student!'),
     re.search('Ravis', 'Ravi is an exceptional student!'),
    re.search('Rr+avi+', 'Ravi is an exceptional student!'),
    re.search('Rr*avi+', 'Ravi is an exceptional student!'),
    bool(re.search('Ravi', 'Ravi is an exceptional student!')),
    bool(re.search('Ravis', 'Ravi is an exceptional student!')),
    re.search('Ravi', 'Ravi is an exceptional student!').group(),
]:
    print(i)


# In[ ]:


# print output of re.search()
match = re.search('Ravi', 'Ravi is an exceptional student!')
print(match.group())


# Let's define a function to match regular expression patterns

# In[ ]:


def find_pattern(text, patterns):
    return re.search(patterns, text) if re.search(patterns, text) else 'Not Found!'


# ### Quantifiers

# In[ ]:


# '*': Zero or more 
print(find_pattern("ac", "ab*"))
print(find_pattern("ac", "(ab)*"))
print(find_pattern("abc", "ab*"))
print(find_pattern("abbc", "ab*"))


# In[ ]:


# '?': Zero or one (tells whether a pattern is absent or present)
print(find_pattern("ac", "ab?"))
print(find_pattern("ac", "(ab)?"))
print(find_pattern("abc", "ab?"))
print(find_pattern("abc", "ab?"))
print(find_pattern("abc", "(ab)?"))
print(find_pattern("abbc", "ab?"))


# In[ ]:


# '+': One or more
print(find_pattern("ac", "ab+"))
print(find_pattern("abc", "ab+"))
print(find_pattern("abbc", "ab+"))


# In[ ]:


# {n}: Matches if a character is present exactly n number of times
print(find_pattern("abbc", "ab{2}"))
print(find_pattern("abbc", "(ab){2}"))


# In[ ]:


# {m,n}: Matches if a character is present from m to n number of times
print(find_pattern("aabbbbbbc", "ab{3,5}"))   # return true if 'b' is present 3-5 times
print(find_pattern("aabbbbbbc", "ab{7,10}"))  # return true if 'b' is present 7-10 times
print(find_pattern("aabbbbbbc", "ab{,10}"))   # return true if 'b' is present atmost 10 times
print(find_pattern("aabbbbbbc", "ab{10,}"))   # return true if 'b' is present from at least 10 times


# ### Anchors

# In[ ]:


# '^': Indicates start of a string
# '$': Indicates end of string

print(find_pattern("James", "^J"))   # return true if string starts with 'J' 
print(find_pattern("Pramod", "^J"))  # return true if string starts with 'J' 
print(find_pattern("India", "a$"))   # return true if string ends with 'c'
print(find_pattern("Japan", "a$"))   # return true if string ends with 'c'


# ### Wildcard

# In[ ]:


# '.': Matches any character
print(find_pattern("a", "."))
print(find_pattern("#", "."))
print(find_pattern("", "."))
print(find_pattern("(#$)$", "."))


# ### Character sets

# In[ ]:


# Now we will look at '[' and ']'.
# They're used for specifying a character class, which is a set of characters that you wish to match.
# Characters can be listed individually as follows
print(find_pattern("a", "[abc]"))

# Or a range of characters can be indicated by giving two characters and separating them by a '-'.
print(find_pattern("c", "[a-c]"))  # same as above


# In[ ]:


# '^' is used inside character set to indicate complementary set
print(find_pattern("a", "[^abc]"))  # return true if neither of these is present - a,b or c
print(find_pattern("a", "[^(bc)]")) 
print(find_pattern("a", "[^(abc)]")) 


# ### Character sets
# | Pattern  | Matches                                                                                    |
# |----------|--------------------------------------------------------------------------------------------|
# | [abc]    | Matches either an a, b or c character                                                      |
# | [abcABC] | Matches either an a, A, b, B, c or C character                                             |
# | [a-z]    | Matches any characters between a and z, including a and z                                  |
# | [A-Z]    | Matches any characters between A and Z, including A and Z                                  |
# | [a-zA-Z] | Matches any characters between a and z, including a and z ignoring cases of the characters |
# | [0-9]    | Matches any character which is a number between 0 and 9                                    |

# ### Meta sequences
# 
# | Pattern  | Equivalent to    |
# |----------|------------------|
# | \s       | [ \t\n\r\f\v]    |
# | \S       | [^ \t\n\r\f\v]   |
# | \d       | [0-9]            |
# | \D       | [^0-9]           |
# | \w       | [a-zA-Z0-9_]     |
# | \W       | [^a-zA-Z0-9_]    |

# In[ ]:


print(find_pattern("Upgrad", "\s+")) # true if whitespace is present
print(find_pattern("Upgrad", "\S+")) # true if whitespace is not present


# ### Greedy vs non-greedy  / lazy regex (Greedy followed by "?")

# In[ ]:


print(find_pattern("aabbbbbb", "ab{3,5}")) # return if a is followed by b 3-5 times GREEDY


# In[ ]:


print(find_pattern("aabbbbbb", "ab{3,5}?")) # return if a is followed by b 3-5 times GREEDY


# In[ ]:


# Example of HTML code
print(re.search("<.*>","<HTML><TITLE>My Page</TITLE></HTML>"))


# In[ ]:


# Example of HTML code
print(re.search("<.*?>","<HTML><TITLE>My Page</TITLE></HTML>"))


# ### The five most important re functions that you would be required to use most of the times are
# 
# match() Determine if the RE matches at the beginning of the string
# 
# search() Scan through a string, looking for any location where this RE matches
# 
# findall() Find all the substrings where the RE matches, and return them as a list
# 
# finditer() Find all substrings where RE matches and return them as an iterator
# 
# sub() Find all substrings where the RE matches and substitute them with the given string

# In[ ]:


# - this function uses the re.match() and let's see how it differs from re.search()
def match_pattern(text, patterns):
    if re.match(patterns, text):
        return re.match(patterns, text)
    else:
        return ('Not found!')


# In[ ]:


print(find_pattern("abbc", "b+"))


# In[ ]:


print(match_pattern("abbc", "b+"))


# In[ ]:


## Example usage of the sub() function. Replace Road with rd.

street = '21 Ramakrishna Road'
print(re.sub('Road', 'Rd', street))


# In[ ]:


print(re.sub('R\w+', 'Rd', street))


pattern = "\d"
replacement = "X"
string = "My address is 13B, Baker Street"

print(re.sub(pattern, replacement, string))


# In[ ]:


## Example usage of finditer(). Find all occurrences of word Festival in given sentence

text = 'Diwali is a festival of lights, Holi is a festival of colors!'
pattern = 'festival'
for match in re.finditer(pattern, text):
    print('START -', match.start(), end="")
    print('END -', match.end())


# In[ ]:


# Example usage of findall(). In the given URL find all dates
url = "http://www.telegraph.co.uk/formula-1/2017/10/28/mexican-grand-prix-2017-time-does-start-tv-channel-odds-lewisl/2017/05/12"
date_regex = '/(\d{4})/(\d{1,2})/(\d{1,2})/'
print(re.findall(date_regex, url))


# In[ ]:


## Exploring Groups
m1 = re.search(date_regex, url)
print(m1.group())  ## print the matched group


# In[ ]:


print(m1.group(1)) # - Print first group


# In[ ]:


print(m1.group(2)) # - Print second group


# In[ ]:


print(m1.group(3)) # - Print third group


# In[ ]:


print(m1.group(0)) # - Print zero or the default group


# ### re.compile()

# In[ ]:


# without re.compile() function
result = re.search("a+", "abc")
print(result)

# using the re.compile() function
result = re.compile("a+").search("abc")
print(result)


# ### Examples

# In[ ]:


# items contains all the files and folders of current directory
items = ['photos', 'documents', 'videos', 'image001.jpg','image002.jpg','image005.jpg', 'wallpaper.jpg',
         'flower.jpg', 'earth.jpg', 'monkey.jpg', 'image002.png']

# create an empty list to store resultant files
images = []

# regex pattern to extract files that end with '.jpg'
pattern = ".*\.jpg$"

for item in items:
    if re.search(pattern, item):
        images.append(item)

# print result
print(images)


# In[ ]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

