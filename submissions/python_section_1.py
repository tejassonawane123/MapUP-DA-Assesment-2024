from typing import Dict, List
import pandas as pd
from typing import List
from typing import List, Tuple
from itertools import permutations
import re
from typing import List
import pandas as pd
import polyline  # To decode polyline strings
from math import radians, sin, cos, sqrt, atan2
from datetime import time
df1 = pd.read_csv('C:/Users/admin/Downloads/MapUp-DA-Assessment-2024-main/MapUp-DA-Assessment-2024-main/datasets/dataset-1.csv')

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    
    # Iterate over the list in steps of size n
    for i in range(0, len(lst), n):
        group = []
        # Manually reverse the current group of size n
        for j in range(min(n, len(lst) - i)):  # Handle the last group that might have fewer than n elements
            group.insert(0, lst[i + j])  # Insert each element at the beginning of the group
        
        # Append the reversed group to the result list
        result.extend(group)
    
    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}

    # Iterate through the list of strings
    for string in lst:
        length = len(string)  # Get the length of the string

        # Add string to the corresponding length list in the dictionary
        if length in result:
            result[length].append(string)
        else:
            result[length] = [string]

    # Return the dictionary sorted by keys (lengths)
    return dict(sorted(result.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten.
    :param sep: The separator to use between parent and child keys (defaults to '.').
    :return: A flattened dictionary.
    """
    def _flatten(obj, parent_key=''):
        items = []

        # Iterate through dictionary items
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Concatenate key
                items.extend(_flatten(v, new_key).items())  # Recursively flatten

        # Iterate through list elements
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}[{i}]"  # Include index for list elements
                items.extend(_flatten(v, new_key).items())  # Recursively flatten

        # For any other type, just add the value
        else:
            items.append((parent_key, obj))

        return dict(items)

    # Start the flattening process
    return _flatten(nested_dict)

# Example usage
# Define the nested dictionary
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

# Call the flatten_dict function
flattened = flatten_dict(nested_dict)

# Print the result
print(flattened)

from itertools import permutations

def unique_permutations(lst):
    return [list(p) for p in set(permutations(lst))]

# Example usage
if __name__ == "__main__":
    print(unique_permutations([1, 1, 2]))




def find_all_dates(text):
    pattern = r'\b\d{2}-\d{2}-\d{4}|\b\d{2}/\d{2}/\d{4}|\b\d{4}\.\d{2}\.\d{2}'
    return re.findall(pattern, text)

# Example usage
if __name__ == "__main__":
    text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
    print(find_all_dates(text))


import pandas as pd
from geopy.distance import geodesic
import polyline

def decode_polyline_to_df(encoded_polyline):
    coordinates = polyline.decode(encoded_polyline)
    data = {'latitude': [], 'longitude': [], 'distance': []}
    
    for i, coord in enumerate(coordinates):
        data['latitude'].append(coord[0])
        data['longitude'].append(coord[1])
        if i == 0:
            data['distance'].append(0)
        else:
            data['distance'].append(geodesic(coordinates[i-1], coord).meters)
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    encoded_polyline = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    df = decode_polyline_to_df(encoded_polyline)
    print(df)



def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element
    with the sum of all elements in the same row and column, excluding itself.

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

    # Step 2: Replace each element with the sum of its row and column, excluding itself
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  # Sum of the row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude itself

    return final_matrix

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform_matrix(matrix)
print(result)



def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of timestamps for each unique (`id`, `id_2`) pair.
    Check whether the timestamps cover a full 24-hour period and span all 7 days of the week.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with timestamp information.

    Returns:
        pd.Series: A boolean series indicating if each (`id`, `id_2`) pair has incorrect timestamps.
    """
    # Combine start and end days with times into datetime
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Group by (`id`, `id_2`)
    grouped = df.groupby(['id', 'id_2'])

    def check_timestamps(group):
        # Get the start and end timestamps
        start_times = group['start_datetime'].min()
        end_times = group['end_datetime'].max()

        # Create a full range of expected timestamps for a week
        full_24_hours = pd.date_range(start=start_times.floor('D'), end=end_times.ceil('D'), freq='D')
        full_week = pd.date_range(start=start_times.floor('D'), periods=7, freq='D')

        # Check if we have all 7 days in the range
        has_full_week = all(day in full_week for day in full_24_hours)

        # Check if we have a full 24-hour period
        has_full_24_hours = (end_times - start_times) >= pd.Timedelta(hours=24)

        return not (has_full_week and has_full_24_hours)

    # Apply the check to each group and return a boolean Series
    result = grouped.apply(check_timestamps)

    return result
