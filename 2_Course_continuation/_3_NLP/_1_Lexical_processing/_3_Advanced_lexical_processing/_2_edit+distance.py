#!/usr/bin/env python
# coding: utf-8

# ## Levenshtein Edit Distance
# The levenshtein distance calculates the number of steps (insertions, deletions or substitutions) required to go from source string to target string.

# In[1]:


def lev_distance(source='', target=''):
    """Make a Levenshtein Distances Matrix"""

    # get length of both strings
    n1, n2 = len(source), len(target)

    # create matrix using length of both strings - source string sits on columns, target string sits on rows
    matrix = [ [ 0 for i1 in range(n1 + 1) ] for i2 in range(n2 + 1) ]

    # fill the first row - (0 to n1-1)
    for i1 in range(1, n1 + 1):
        matrix[0][i1] = i1

    # fill the first column - (0 to n2-1)
    for i2 in range(1, n2 + 1):
        matrix[i2][0] = i2

    # fill the matrix
    for i2 in range(1, n2 + 1):
        for i1 in range(1, n1 + 1):

            # check whether letters being compared are same
            if (source[i1-1] == target[i2-1]):
                value = matrix[i2-1][i1-1]               # top-left cell value
            else:
                value = min(matrix[i2-1][i1]   + 1,      # left cell value     + 1
                            matrix[i2][i1-1]   + 1,      # top cell  value     + 1
                            matrix[i2-1][i1-1] + 1)      # top-left cell value + 1

            matrix[i2][i1] = value

    # return bottom-right cell value
    return matrix[-1][-1]


# In[2]:


lev_distance('cat', 'cta')


# ## Levenshtein distance in nltk library

# In[3]:


# import library
from nltk.metrics.distance import edit_distance


# In[4]:


print(
    edit_distance("apple", "appel"),"\n",
edit_distance("sparking", "parking"),"\n",
    edit_distance('perspective','prospective'),
)


# ## Damerau-Levenshtein Distance
# The Damerau-Levenshtein distance allows transpositions (swap of two letters which are adjacent to each other) as well.

# In[5]:


edit_distance("apple", "appel", transpositions=False, )


# In[6]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

