Question 9: Distance Matrix Calculation
Create a function named calculate_distance_matrix that takes the dataset-2.csv as input and generates a DataFrame representing distances between IDs.

The resulting DataFrame should have cumulative distances along known routes, with diagonal values set to 0. If distances between toll locations A to B and B to C are known, then the distance from A to C should be the sum of these distances. Ensure the matrix is symmetric, accounting for bidirectional distances between toll locations (i.e. A to B is equal to B to A).

Sample result dataframe:
Section 2 Question 9


SOLUTION 
import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): DataFrame containing columns ['id_start', 'id_end', 'distance'].

    Returns:
        pd.DataFrame: Distance matrix
    """
    # Extract unique IDs from both columns
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    num_ids = len(unique_ids)

    # Create an empty distance matrix initialized with np.inf
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    # Fill the diagonal with zeros
    np.fill_diagonal(distance_matrix.values, 0)

    # Populate the matrix with known distances
    for _, row in df.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']

        distance_matrix.at[start_id, end_id] = distance
        distance_matrix.at[end_id, start_id] = distance  # Ensure symmetry

    # Apply Floyd-Warshall algorithm to calculate all pairs shortest paths
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                # Update the distance matrix if a shorter path is found
                if distance_matrix.at[i, k] + distance_matrix.at[k, j] < distance_matrix.at[i, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix

# Example Usage
# df = pd.read_csv('dataset-2.csv')  # Load your dataset
# distance_matrix = calculate_distance_matrix(df)
# print(distance_matrix)


--------------------------------------------------------------------------------------------------------------------------------------------------------


Question 10: Unroll Distance Matrix
Create a function unroll_distance_matrix that takes the DataFrame created in Question 9. The resulting DataFrame should have three columns: columns id_start, id_end, and distance.

All the combinations except for same id_start to id_end must be present in the rows with their distance values from the input DataFrame.


  def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    return df

SOLUTION


import pandas as pd

def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_matrix (pandas.DataFrame): The distance matrix with IDs as index and columns.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []

    # Iterate through the matrix to extract id_start, id_end, and distance
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude self-references
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the collected data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Sample distance matrix for demonstration
data = {
    'A': [0.0, 5.0, 15.0],
    'B': [5.0, 0.0, 10.0],
    'C': [15.0, 10.0, 0.0]
}
distance_matrix = pd.DataFrame(data, index=['A', 'B', 'C'])

# Unroll the distance matrix

unrolled_df = unroll_distance_matrix(distance_matrix)

# Show the output
print(unrolled_df)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

Question 11: Finding IDs within Percentage Threshold
Create a function find_ids_within_ten_percentage_threshold that takes the DataFrame created in Question 10 and a reference value from the id_start column as an integer.

Calculate average distance for the reference value given as an input and return a sorted list of values from id_start column which lie within 10% (including ceiling and floor) of the reference value's average.

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    return df

SOLUTION

import pandas as pd

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): The ID for which to calculate the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate average distance for the reference ID
    avg_distance_reference = df[df['id_start'] == reference_id]['distance'].mean()

    # Calculate bounds for the 10% threshold
    lower_bound = avg_distance_reference * 0.9
    upper_bound = avg_distance_reference * 1.1

    # Filter the DataFrame for IDs within the specified distance range
    ids_within_threshold = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_ids = ids_within_threshold[
        (ids_within_threshold['distance'] >= lower_bound) & 
        (ids_within_threshold['distance'] <= upper_bound)
    ]

    # Return a sorted DataFrame of IDs
    return filtered_ids.sort_values(by='id_start')

# Example Usage
# Assuming unrolled_df is the DataFrame created in Question 10
# unrolled_df = pd.DataFrame({
#     'id_start': [1, 1, 2, 2, 3, 3],
#     'id_end': [2, 3, 1, 3, 1, 2],
#     'distance': [5.0, 10.0, 5.0, 15.0, 10.0, 15.0]
# })

# reference_id = 1
# result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
# print(result_df)


-------------------------------------------------------------------------------------------------------------------------------------------------------------

Question 12: Calculate Toll Rate
Create a function calculate_toll_rate that takes the DataFrame created in Question 10 as input and calculates toll rates based on vehicle types.

The resulting DataFrame should add 5 columns to the input DataFrame: moto, car, rv, bus, and truck with their respective rate coefficients. The toll rates should be calculated by multiplying the distance with the given rate coefficients for each vehicle type:

0.8 for moto
1.2 for car
1.5 for rv
2.2 for bus
3.6 for truck
    def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    return df

SOLUTION

import pandas as pd

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: The original DataFrame with additional columns for toll rates.
    """
    # Define the rate coefficients
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate in rates.items():
        df[vehicle_type] = df['distance'] * rate

    return df

# Example Usage
# Assuming unrolled_df is the DataFrame created in Question 10
# unrolled_df = pd.DataFrame({
#     'id_start': [1, 1, 2, 2, 3, 3],
#     'id_end': [2, 3, 1, 3, 1, 2],
#     'distance': [5.0, 10.0, 5.0, 15.0, 10.0, 15.0]
# })

# result_df = calculate_toll_rate(unrolled_df)
# print(result_df)


----------------------------------------------------------------------------------------------------------------------------------------------------------------

Question 13: Calculate Time-Based Toll Rates
Create a function named calculate_time_based_toll_rates that takes the DataFrame created in Question 12 as input and calculates toll rates for different time intervals within a day.

The resulting DataFrame should have these five columns added to the input: start_day, start_time, end_day, and end_time.

start_day, end_day must be strings with day values (from Monday to Sunday in proper case)
start_time and end_time must be of type datetime.time() with the values from time range given below.
Modify the values of vehicle columns according to the following time ranges:

Weekdays (Monday - Friday):

From 00:00:00 to 10:00:00: Apply a discount factor of 0.8
From 10:00:00 to 18:00:00: Apply a discount factor of 1.2
From 18:00:00 to 23:59:59: Apply a discount factor of 0.8
Weekends (Saturday and Sunday):

Apply a constant discount factor of 0.7 for all times.
For each unique (id_start, id_end) pair, cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).
                                                                                                                                                                                                                                        
SOLUTION

import pandas as pd
import numpy as np
from datetime import time

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame containing toll rates for different vehicle types.

    Returns:
        pandas.DataFrame: Updated DataFrame with time-based toll rates and additional time columns.
    """
    # Add start_day, start_time, end_day, and end_time columns
    # For demonstration, we can assume all entries are on the same day
    # You may want to customize this based on your actual use case
    df['start_day'] = 'Monday'  # Example, can vary based on your logic
    df['end_day'] = 'Monday'     # Example, can vary based on your logic
    df['start_time'] = time(0, 0)  # 00:00:00
    df['end_time'] = time(23, 59)   # 23:59:59

    # Define the discount factors
    weekday_discount = {
        'morning': 0.8,  # 00:00 to 10:00
        'day': 1.2,      # 10:00 to 18:00
        'evening': 0.8   # 18:00 to 23:59
    }
    weekend_discount = 0.7

    # Apply discounts based on day and time
    def apply_discount(row):
        start_day = row['start_day']
        # Apply weekend discount for Saturday and Sunday
        if start_day in ['Saturday', 'Sunday']:
            for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                row[vehicle] *= weekend_discount
        else:  # Apply weekday discounts
            start_hour = row['start_time'].hour
            if 0 <= start_hour < 10:  # 00:00 to 10:00
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    row[vehicle] *= weekday_discount['morning']
            elif 10 <= start_hour < 18:  # 10:00 to 18:00
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    row[vehicle] *= weekday_discount['day']
            else:  # 18:00 to 23:59
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    row[vehicle] *= weekday_discount['evening']
        return row

    # Apply the discount function to each row
    df = df.apply(apply_discount, axis=1)

    return df




---------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            



