from typing import Dict, List
import re
from typing import List
import pandas as pd
import polyline
import pandas as pd
import numpy as np



def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []
    length = len(lst)

    for i in range(0, length, n):
        end = min(i + n, length)        
        for j in range(end - 1, i - 1, -1):
            result.append(lst[j])
    
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}

    for string in lst:
        length = len(string)
        if length in length_dict:
            length_dict[length].append(string)
        else:
            length_dict[length] = [string]
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    flattened_dict = {}

    def flatten(current_dict: Dict, parent_key: str = ''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                flatten(value, new_key)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    item_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):
                        flatten(item, item_key)
                    else:
                        flattened_dict[item_key] = item
            else:
                flattened_dict[new_key] = value

    flatten(nested_dict)
    return flattened_dict
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    result = []
    nums.sort()
    backtrack(0)
    return result
    pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b', 
        r'\b(\d{2})/(\d{2})/(\d{4})\b',
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b' 
    ]
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if pattern == patterns[0]:
                dates.append(f"{match[0]}-{match[1]}-{match[2]}")
            elif pattern == patterns[1]:
                dates.append(f"{match[0]}/{match[1]}/{match[2]}")
            elif pattern == patterns[2]:
                dates.append(f"{match[0]}.{match[1]}.{match[2]}")
    
    return dates
    pass


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371000
    return r * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    distances = [0]
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)

    df['distance'] = distances
    
    return df

    return pd.Dataframe()


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
    n = len(matrix)
    
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix
    return []

def time_check(df: pd.DataFrame) -> pd.Series:
    required_cols = {'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'}
    if not required_cols.issubset(df.columns):
        raise ValueError("Missing required columns")

    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['startDay_num'] = df['startDay'].map(day_map)
    df['endDay_num'] = df['endDay'].map(day_map)

    def spans_full_day(start, end):
        return start == '00:00:00' and end == '23:59:59'

    results = []
    for (id, id_2), group in df.groupby(['id', 'id_2']):
        days = set()
        complete = True
        for _, row in group.iterrows():
            days.update(range(row['startDay_num'], row['endDay_num'] + 1))
            if row['startDay_num'] == row['endDay_num'] and not spans_full_day(row['startTime'], row['endTime']):
                complete = False
        results.append((id, id_2, days != set(range(7)) or not complete))
    return pd.Series([r[2] for r in results], index=pd.MultiIndex.from_tuples([(r[0], r[1]) for r in results]))

# Test Cases
if __name__ == "__main__":
    # Test for Question 1: Reverse List by N Elements
    print("Reverse by N Elements:")
    print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))
    print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))
    print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))

    # Test for Question 2: Lists & Dictionaries
    print("\nGroup by Length:")
    print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))

    # Test for Question 3: Flatten a Nested Dictionary
    print("\nFlatten a Nested Dictionary:")
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

    # Test for Question 4: Generate Unique Permutations
    print("\nUnique Permutations:")
    print(unique_permutations([1, 1, 2]))

    # Test for Question 5: Find All Dates in a Text
    print("\nFind All Dates:")
    text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
    print(find_all_dates(text))

    # Test for Question 6: Decode Polyline, Convert to DataFrame with Distances
    print("\nDecode Polyline and Create DataFrame:")
    polyline_str = "a~lHouC"
    df = polyline_to_dataframe(polyline_str)
    print(df)

    # Test for Question 7: Matrix Rotation and Transformation
    print("\nRotate and Multiply Matrix:")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transformed_matrix = rotate_and_multiply_matrix(matrix)
    print(transformed_matrix)

    print("\nTime Check:")

    df = pd.read_csv('datasets\dataset-1.csv')

    result = time_check(df)
    print(result)
