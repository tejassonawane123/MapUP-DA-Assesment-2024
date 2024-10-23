from typing import Dict, List

import pandas as pd
import polyline
import math
import re 


def reverse_by_n_elements(lst: List[int], n: int):
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
       
        group = []
        for j in range(n):
            if i + j < length:
                group.append(lst[i + j])
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    
    return result


def group_by_length(lst: List[str]):
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
        
    sorted_dict = dict(sorted(length_dict.items()))
    
    return sorted_dict
    

def flatten_dict(nested_dict: Dict, sep: str = '.'):
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items = {}
    
    for key, value in nested_dict.items():
        
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
           
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
           
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{new_key}[{i}]", sep=sep))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
           
            items[new_key] = value

    return items

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:]) 
            return
        
        seen = set()  
        for i in range(start, len(nums)):
            if nums[i] not in seen:  
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  
                backtrack(start + 1)  
                nums[start], nums[i] = nums[i], nums[start]  
    
    nums.sort()
    result = []
    backtrack(0)
    return result
    


def find_all_dates(text: str):
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
      
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',
        r'\b\d{2}/\d{2}/\d{4}\b', 
        r'\b\d{4}\.\d{2}\.\d{2}\b'
    ]
    
 
    combined_pattern = '|'.join(patterns)
    
  
    dates = re.findall(combined_pattern, text)
    
    return dates
    
    
def haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


def polyline_to_dataframe(polyline_str: str):
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    df['distance'] = 0.0
    
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df
    
def rotate_matrix(matrix):
    n = len(matrix)
    
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    return rotated

def transform_matrix(matrix):
    n = len(matrix)
    
   
    transformed = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
           
            row_sum = sum(matrix[i]) - matrix[i][j]
            col_sum = sum(matrix[k][j] for k in range(n)) - matrix[i][j]
            transformed[i][j] = row_sum + col_sum
            
    return transformed


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
      
    rotated = rotate_matrix(matrix)
    

    final_matrix = transform_matrix(rotated)
    
    return final_matrix
    
    


def time_check(df):
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time
    grouped = df.groupby(['id', 'id_2'])

    results = pd.Series(dtype=bool)

    for name, group in grouped:
        if (group['startTime'].min() <= pd.to_datetime('00:00:00', format='%H:%M:%S').time() and
            group['endTime'].max() >= pd.to_datetime('23:59:59', format='%H:%M:%S').time() and
            set(group['startDay']).union(set(group['endDay'])) == set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])):
            results[name] = False
        else:
            results[name] = True

    return results
