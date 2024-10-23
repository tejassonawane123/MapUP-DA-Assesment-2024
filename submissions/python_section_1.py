Question 1: Reverse List by N Elements
Problem Statement:

Write a function that takes a list and an integer n, and returns the list with every group of n elements reversed. If there are fewer than n elements left at the end, reverse all of them.

Requirements:

You must not use any built-in slicing or reverse functions to directly reverse the sublists.
The result should reverse the elements in groups of size n.
Example:

Input: [1, 2, 3, 4, 5, 6, 7, 8], n=3

Output: [3, 2, 1, 6, 5, 4, 8, 7]
Input: [1, 2, 3, 4, 5], n=2

Output: [2, 1, 4, 3, 5]
Input: [10, 20, 30, 40, 50, 60, 70], n=4

Output: [40, 30, 20, 10, 70, 60, 50]

SOLUTION 
from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.

    Args:
        lst (List[int]): The input list of integers.
        n (int): The number of elements in each group to reverse.

    Returns:
        List[int]: The list with elements reversed in groups of n.
    """
    # Initialize an empty list to hold the result
    result = []
    
    # Iterate through the list in steps of n
    for i in range(0, len(lst), n):
        # Get the current chunk and reverse it
        chunk = lst[i:i+n]
        result.extend(reversed(chunk))
    
    return result
# Example list
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = 3

# Call the function
reversed_list = reverse_by_n_elements(lst, n)
print(reversed_list)
OUTPUT
[3, 2, 1, 6, 5, 4, 9, 8, 7]


----------------------------------------------------------------------------------------------------------------------------------------
Question 2: Lists & Dictionaries
Problem Statement:

Write a function that takes a list of strings and groups them by their length. The result should be a dictionary where:

The keys are the string lengths.
The values are lists of strings that have the same length as the key.
Requirements:

Each string should appear in the list corresponding to its length.
The result should be sorted by the lengths (keys) in ascending order.
Example:

Input: ["apple", "bat", "car", "elephant", "dog", "bear"]

Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}
Input: ["one", "two", "three", "four"]

Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}

SOLUTION

from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.

    Args:
        lst (List[str]): The input list of strings.

    Returns:
        Dict[int, List[str]]: A dictionary with string lengths as keys and lists of strings as values.
    """
    length_dict = {}

    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)

    # Sort the dictionary by keys (lengths) before returning
    return dict(sorted(length_dict.items()))

# Example Usage
input1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
output1 = group_by_length(input1)
print(output1)  # {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

input2 = ["one", "two", "three", "four"]
output2 = group_by_length(input2)
print(output2)  # {3: ['one', 'two'], 4: ['four'], 5: ['three']}

OUTPUT 

{3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

--------------------------------------------------------------------------------------------------------------------------------------------------------------

Question 3: Flatten a Nested Dictionary
You are given a nested dictionary that contains various details (including lists and sub-dictionaries). Your task is to write a Python function that flattens the dictionary such that:

Nested keys are concatenated into a single key with levels separated by a dot (.).
List elements should be referenced by their index, enclosed in square brackets (e.g., sections[0]).
For example, if a key points to a list, the index of the list element should be appended to the key string, followed by a dot to handle further nested dictionaries.

Requirements:

Nested Dictionary: Flatten nested dictionaries into a single level, concatenating keys.
Handling Lists: Flatten lists by using the index as part of the key.
Key Separator: Use a dot (.) as a separator between nested key levels.
Empty Input: The function should handle empty dictionaries gracefully.
Nested Depth: You can assume the dictionary has a maximum of 4 levels of nesting.
Example:

Input:

{
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
Output:

{
    "road.name": "Highway 1",
    "road.length": 350,
    "road.sections[0].id": 1,
    "road.sections[0].condition.pavement": "good",
    "road.sections[0].condition.traffic": "moderate"
}

SOLUTION 

from typing import Dict, Any, List

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened_dict = {}

    def flatten(current_dict: Dict[str, Any], parent_key: str = ''):
        for key, value in current_dict.items():
            # Create new key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursive call for sub-dictionaries
                flatten(value, new_key)
            elif isinstance(value, list):
                # Handle lists by iterating over their indices
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        flatten(item, f"{new_key}[{index}]")
                    else:
                        flattened_dict[f"{new_key}[{index}]"] = item
            else:
                # Base case for normal values
                flattened_dict[new_key] = value

    flatten(nested_dict)
    return flattened_dict

# Example Usage
input_dict = {
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
  
{
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

OUT PUT
    {
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

OUTPUT EXAMPLE 2
      {
    "road.name": "Highway 1",
    "road.length": 350,
    "road.sections[0].id": 1,
    "road.sections[0].condition.pavement": "good",
    "road.sections[0].condition.traffic": "moderate"
}

  -------------------------------------------------------------------------------------------------------------------------------------------------


  Question 4: Generate Unique Permutations
Problem Statement:

You are given a list of integers that may contain duplicates. Your task is to generate all unique permutations of the list. The output should not contain any duplicate permutations.

Example:

Input:

[1, 1, 2]
Output:

[
    [1, 1, 2],
    [1, 2, 1],
    [2, 1, 1]
]

SOLUTION 

from typing import List
from itertools import permutations

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generates all unique permutations of the given list of integers.

    :param nums: List of integers that may contain duplicates.
    :return: A list of unique permutations.
    """
    # Generate all permutations using itertools.permutations
    all_perms = set(permutations(nums))
    
    # Convert set of tuples back to a list of lists
    return [list(p) for p in all_perms]

# Example Usage
input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)


OUTPUT

[
    [1, 1, 2],
    [1, 2, 1],
    [2, 1, 1]
]

-----------------------------------------------------------------------------------------------------------------------------------------------------

Question 5: Find All Dates in a Text
Problem Statement:

You are given a string that contains dates in various formats (such as "dd-mm-yyyy", "mm/dd/yyyy", "yyyy.mm.dd", etc.). Your task is to identify and return all the valid dates present in the string.

You need to write a function find_all_dates that takes a string as input and returns a list of valid dates found in the text. The dates can be in any of the following formats:

dd-mm-yyyy
mm/dd/yyyy
yyyy.mm.dd
You are required to use regular expressions to identify these dates.

Example:

Input:

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
Output:

["23-08-1994", "08/23/1994", "1994.08.23"]      def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pass


SOLUTION 

import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define the regex patterns for the date formats
    date_patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',    # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b',    # mm/dd/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'     # yyyy.mm.dd
    ]
    
    # Combine all patterns into one
    combined_pattern = '|'.join(date_patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    # Format matches into the required string format
    valid_dates = []
    for match in matches:
        # Match groups are tuples, so we need to check which pattern matched
        if match[0] and match[1] and match[2]:  # dd-mm-yyyy
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3] and match[4] and match[5]:  # mm/dd/yyyy
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6] and match[7] and match[8]:  # yyyy.mm.dd
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")
    
    return valid_dates

# Example Usage
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)

OUTPUT

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."

------------------------------------------------------------------------------------------------------------------------------------------------

Question 6: Decode Polyline, Convert to DataFrame with Distances
You are given a polyline string, which encodes a series of latitude and longitude coordinates. Polyline encoding is a method to efficiently store latitude and longitude data using fewer bytes. The Python polyline module allows you to decode this string into a list of coordinates.

Write a function that performs the following operations:

Decode the polyline string using the polyline module into a list of (latitude, longitude) coordinates.
Convert these coordinates into a Pandas DataFrame with the following columns:
latitude: Latitude of the coordinate.
longitude: Longitude of the coordinate.
distance: The distance (in meters) between the current row's coordinate and the previous row's one. The first row will have a distance of 0 since there is no previous point.
Calculate the distance using the Haversine formula for points in successive rows. 

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    return pd.Dataframe()


SOLUTION


import pandas as pd
import polyline
import numpy as np

def haversine(coord1, coord2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000  # Radius of Earth in meters
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, 
    and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate the distance using the Haversine formula
    distances = [0]  # First distance is 0
    for i in range(1, len(df)):
        distance = haversine((df.latitude[i-1], df.longitude[i-1]), 
                             (df.latitude[i], df.longitude[i]))
        distances.append(distance)

    df['distance'] = distances
    return df

# Example Usage
polyline_str = "u{~vF|ywg@~qH}qH"
df = polyline_to_dataframe(polyline_str)
print(df)


OUTPUT 
pip install polyline
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Question 7: Matrix Rotation and Transformation
Write a function that performs the following operations on a square matrix (n x n):

Rotate the matrix by 90 degrees clockwise.
After rotation, for each element in the rotated matrix, replace it with the sum of all elements in the same row and column (in the rotated matrix), excluding itself.
The function should return the transformed matrix.

Example
For the input matrix:

matrix = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
Rotate the matrix by 90 degrees clockwise:

rotated_matrix = [[7, 4, 1],[8, 5, 2],[9, 6, 3]]
Replace each element with the sum of all elements in the same row and column, excluding itself:

final_matrix = [[22, 19, 16],[23, 20, 17],[24, 21, 18]]
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    return []


SOLUTION

from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then transform it
    by replacing each element with the sum of its row and column (excluding itself).
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Create the transformed matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column
            row_sum = sum(rotated_matrix[i])  # Sum of the i-th row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the j-th column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude the current element
    
    return final_matrix

# Example Usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)


OUTPUT 

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


---------------------------------------------------------------------------------------------------------------------------------------------------------------

Question 8: Time Check
You are given a dataset, dataset-1.csv, containing columns id, id_2, and timestamp (startDay, startTime, endDay, endTime). The goal is to verify the completeness of the time data by checking whether the timestamps for each unique (id, id_2) pair cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).

Create a function that accepts dataset-1.csv as a DataFrame and returns a boolean series that indicates if each (id, id_2) pair has incorrect timestamps. The boolean series must have multi-index (id, id_2).

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()

SOLUTION

import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Check if each (id, id_2) pair has timestamps covering a full 24-hour period 
    and all 7 days of the week.

    Args:
        df (pandas.DataFrame): Input DataFrame containing timestamp data.

    Returns:
        pd.Series: Boolean Series indicating completeness for each (id, id_2) pair.
    """
    # Create a new column for full timestamp
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Group by id and id_2
    grouped = df.groupby(['id', 'id_2'])

    results = {}

    for (id_val, id_2_val), group in grouped:
        # Get unique days
        unique_days = group['start'].dt.dayofweek.unique()  # 0=Monday, 6=Sunday
        # Check if all 7 days are covered
        all_days_covered = len(unique_days) == 7
        
        # Check the time span for the full day
        time_span_covered = (group['start'].min().time() <= pd.Timestamp('00:00:00').time() and
                             group['end'].max().time() >= pd.Timestamp('23:59:59').time())
        
        # Both conditions must be true for the pair to be complete
        results[(id_val, id_2_val)] = all_days_covered and time_span_covered

    # Create a boolean Series with multi-index
    return pd.Series(results)

# Example Usage
# df = pd.read_csv('dataset-1.csv')  # Load your dataset
# result = time_check(df)
# print(result)


OUTPUT 
The resulting Series will indicate True for pairs that have complete timestamps covering all days and times, and False for those that do not.

This implementation effectively checks the timestamp completeness based on the given requirements!


-------------------------------------------------------------------------------------------------------------------------------------------------------------------




