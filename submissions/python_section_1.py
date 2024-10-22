#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np


# ## Question 1: Reverse List by N Elements

# In[1]:


def reverse_by_n(lst, n):
    result = []
    
    # Iterate through the list in chunks of size 'n'
    for i in range(0, len(lst), n):
        # Get the current chunk
        chunk = lst[i:i + n]
        
        # Reverse the chunk manually
        for j in range(len(chunk) // 2):
            # Swap the elements
            chunk[j], chunk[len(chunk) - 1 - j] = chunk[len(chunk) - 1 - j], chunk[j]
        
        # Append the reversed chunk to the result
        result.extend(chunk)
    
    return result

# Test cases
print(reverse_by_n([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n([1, 2, 3, 4, 5], 2))           # Output: [2, 1, 4, 3, 5]
print(reverse_by_n([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]


# ## Question 2: Lists & Dictionaries

# In[2]:


def group_by_length(strings):
    length_dict = {}
    
    # Iterate through each string in the list
    for string in strings:
        length = len(string)
        
        # If the length is already a key in the dictionary, append the string
        if length in length_dict:
            length_dict[length].append(string)
        else:
            # Otherwise, create a new key and start a list
            length_dict[length] = [string]
    
    # Sort the dictionary by keys and return the result
    return dict(sorted(length_dict.items()))

# Test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}


# ## Question 3: Flatten a Nested Dictionary

# In[3]:


def flatten_dict(d, parent_key=''):
    items = []
    
    # Iterate through each key-value pair in the dictionary
    for k, v in d.items():
        # Create the new key by concatenating the parent key and the current key
        new_key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, dict):
            # If the value is a dictionary, recursively flatten it
            items.extend(flatten_dict(v, new_key).items())
        elif isinstance(v, list):
            # If the value is a list, iterate through each element with its index
            for i, item in enumerate(v):
                items.extend(flatten_dict({f"{k}[{i}]": item}, parent_key).items())
        else:
            # For other values (strings, numbers, etc.), add them to the result
            items.append((new_key, v))
    
    return dict(items)

# Test case
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

print(flatten_dict(nested_dict))

# Expected Output:
# {
#     "road.name": "Highway 1",
#     "road.length": 350,
#     "road.sections[0].id": 1,
#     "road.sections[0].condition.pavement": "good",
#     "road.sections[0].condition.traffic": "moderate"
# }


# ## Question 4: Generate Unique Permutations

# In[4]:


def unique_permutations(nums):
    def backtrack(path, used):
        # If the current path is a complete permutation, add it to the result
        if len(path) == len(nums):
            result.append(path[:])  # Add a copy of the path
            return
        
        # Iterate through the numbers
        for i in range(len(nums)):
            # Skip if the current number is the same as the previous one and the previous one hasn't been used
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            # Skip if the number has already been used in this path
            if used[i]:
                continue
            
            # Mark the number as used and add it to the current path
            used[i] = True
            path.append(nums[i])
            
            # Recurse to continue building the permutation
            backtrack(path, used)
            
            # Backtrack: remove the last number and mark it as unused
            path.pop()
            used[i] = False

    # Sort the input to handle duplicates
    nums.sort()
    result = []
    used = [False] * len(nums)
    
    # Start backtracking with an empty path
    backtrack([], used)
    
    return result

# Test case
print(unique_permutations([1, 1, 2]))

# Expected Output:
# [
#     [1, 1, 2],
#     [1, 2, 1],
#     [2, 1, 1]
# ]


# ## Question 5: Find All Dates in a Text

# In[5]:


import re

def find_all_dates(text):
    # Define the regex patterns for the different date formats
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]
    
    # List to store all found dates
    all_dates = []
    
    # Search for all matching patterns in the text
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        all_dates.extend(matches)
    
    return all_dates

# Test case
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))

# Expected Output:
# ["23-08-1994", "08/23/1994", "1994.08.23"]


# ## Question 7: Matrix Rotation and Transformation

# In[9]:


def rotate_and_transform_matrix(matrix):
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Transform the rotated matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate sum of the row and column excluding the element itself
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform_matrix(matrix)
print(result)


# ## Question 8: Time Check

# In[18]:


df = pd.read_csv('dataset-1.csv')


# In[19]:


df.head()


# In[20]:


import pandas as pd

def check_time_completeness(df):
    # Ensure that the timestamp columns are in datetime format
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'],errors='coerce')
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'],errors='coerce')
    
    # Create a multi-index from 'id' and 'id_2'
    df.set_index(['id', 'id_2'], inplace=True)

    # Group by (id, id_2)
    results = {}
    
    for (id_val, id_2_val), group in df.groupby(level=['id', 'id_2']):
        # Get the start and end times
        start_times = group['start']
        end_times = group['end']

        # Check if the timestamps cover a full 24-hour period
        time_coverage = start_times.min() < start_times.max() and end_times.min() < end_times.max()

        # Check for all days of the week
        all_days_covered = len(pd.Series(start_times.dt.dayofweek).unique()) == 6  # 0-6 for Mon-Sun

        # Check if both conditions are satisfied
        results[(id_val, id_2_val)] = not (time_coverage and all_days_covered)
    
    # Create a boolean series with multi-index
    boolean_series = pd.Series(results, dtype=bool)
    boolean_series.index = pd.MultiIndex.from_tuples(boolean_series.index, names=['id', 'id_2'])
    
    return boolean_series

# Check the completeness of time data
result = check_time_completeness(df)
print(result)


# In[21]:


result.reset_index()


# In[ ]:




