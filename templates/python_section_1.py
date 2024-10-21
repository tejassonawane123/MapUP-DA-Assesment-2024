from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: list[int], n: int) -> list[int]:
    """
    Reverses the input list by groups of n elements.
    """
    ls = []
    
    # Traverse the list in steps of n
    for i in range(0, len(lst), n):
        
        group = []
        for j in range(min(n, len(lst) - i)):  
            group.insert(0, lst[i + j])  
        
        ls.extend(group)  
    
    return ls




def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    # Iterate through the list of strings
    for string in lst:
        length = len(string)
        
        # If the length is already a key in the dictionary, append the string to the list
        if length in length_dict:
            length_dict[length].append(string)
        # Otherwise, create a new list with the string
        else:
            length_dict[length] = [string]
    
    # Sort the dictionary by keys (lengths) in ascending order and return
    return dict(sorted(length_dict.items()))

input_list = ["apple", "bat", "car", "elephant", "dog", "bear"]
output = group_by_length(input_list)
print(output)




def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(current_dict: Any, parent_key: str = '') -> Dict[str, Any]:
        items = []
        
        
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        items.extend(_flatten(item, list_key).items())
                    else:
                        items.append((list_key, item))
            else:
                # Base case: the value is neither a dictionary nor a list, add it directly
                items.append((new_key, v))
        
        return dict(items)

    # Call the recursive function on the root dictionary
    return _flatten(nested_dict)

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

flattened = flatten_dict(nested_dict)
print(flattened)


from typing import List
from itertools import permutations

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Use itertools.permutations to generate all permutations
    all_permutations = permutations(nums)
    
    # Convert each permutation to a tuple and use a set to remove duplicates
    unique_perms = set(all_permutations)
    
    # Convert back to a list of lists for the output
    return [list(perm) for perm in unique_perms]

# Example usage
input_list = [1, 1, 2]
output = unique_permutations(input_list)
print(output)


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
    # Regular expression patterns for the specified date formats
    patterns = [
        r'\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})\b',  # dd-mm-yyyy
        r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])-(\d{4})\b',  # mm/dd/yyyy
        r'\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\b'   # yyyy.mm.dd
    ]
    
    # Combine patterns into one regex
    combined_pattern = '|'.join(patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    # Flatten and filter matches to create a list of valid dates
    valid_dates = []
    for match in matches:
        # Join the matched groups and format them correctly based on the pattern
        if match[0]:  # Match for dd-mm-yyyy
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  # Match for mm/dd/yyyy
            valid_dates.append(f"{match[3]}/{match[4]}-{match[5]}")
        elif match[6]:  # Match for yyyy.mm.dd
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")

    return valid_dates

# Example usage with the provided input
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output_dates = find_all_dates(text)
print(output_dates)


import polyline
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    
    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.
        
    Returns:
        float: Distance in meters.
    """
    R = 6371000  # Radius of the Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) * 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) * 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in meters

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate distances
    distances = [0.0]  # First point has a distance of 0
    for i in range(1, len(df)):
        distance = haversine(df.latitude[i-1], df.longitude[i-1], df.latitude[i], df.longitude[i])
        distances.append(distance)

    df['distance'] = distances
    
    return df

# Example usage
polyline_str = "u{~vF~yzxO~jA`x@_dB}fEwHq@bCmB"
df = polyline_to_dataframe(polyline_str)
print(df)



from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of all elements in the same row and column, excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]  # Create an empty n x n matrix
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Create a new matrix to store the transformed values
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate the sum of the current row and column, excluding the current element
            row_sum = sum(rotated_matrix[i])  # Sum of the i-th row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the j-th column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude itself

    return final_matrix

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)



import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the data by checking whether the timestamps for each unique (id, id_2) pair cover 
    a full 24-hour and 7-day period.
    
    Args:
        df (pandas.DataFrame): DataFrame containing columns id, id_2, startDay, startTime, endDay, endTime.
        
    Returns:
        pd.Series: A boolean Series indicating whether each (id, id_2) pair has incorrect timestamps.
    """
    
    # Combine date and time columns into a single datetime column for start and end times
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Set MultiIndex using id and id_2
    df.set_index(['id', 'id_2'], inplace=True)
    
    # Function to check each group
    def check_timestamps(group):
        # Check if the start and end cover the entire week (7 days) and 24 hours
        full_24_hours = (group['end'].max() - group['start'].min()) >= pd.Timedelta(days=7) \
                        and (group['end'].max().date() - group['start'].min().date()) == pd.Timedelta(days=6)
        
        # Get unique days in the group
        unique_days = group['start'].dt.date.unique()
        
        # Check if it spans all days of the week
        covers_all_days = len(unique_days) == 7
        
        # Return False if either condition fails
        return not (full_24_hours and covers_all_days)

    # Apply the function to each group and return as a boolean series
    result = df.groupby(level=[0, 1]).apply(check_timestamps)
    
    return result
