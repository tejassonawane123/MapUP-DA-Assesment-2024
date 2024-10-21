import pandas as pd


import pandas as pd
import numpy as np

def calculate_distance_matrix(df) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Initialize distance matrix as a DataFrame with same index and columns as df
    distance_matrix = pd.DataFrame(index=df.index, columns=df.columns, data=np.inf)

    # Fill the diagonal with 0s since the distance from a point to itself is zero
    np.fill_diagonal(distance_matrix.values, 0)

    # Fill the known distances from the input DataFrame
    for i in df.index:
        for j in df.columns:
            if not pd.isna(df.loc[i, j]):
                distance_matrix.loc[i, j] = df.loc[i, j]

    # Compute cumulative distances using Floyd-Warshall algorithm to ensure symmetry and find shortest paths
    for k in df.index:
        for i in df.index:
            for j in df.columns:
                # Update the distance with the minimum of the current or the new path through k
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], distance_matrix.loc[i, k] + distance_matrix.loc[k, j])

    # Ensure symmetry (A to B == B to A)
    distance_matrix = distance_matrix.where(np.triu(np.ones(distance_matrix.shape)).astype(bool)).T + distance_matrix.where(np.tril(np.ones(distance_matrix.shape)).astype(bool))

    return distance_matrix


import pandas as pd

def unroll_distance_matrix(df) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize an empty list to collect the rows
    rows = []

    # Loop over all rows and columns of the DataFrame
    for id_start in df.index:
        for id_end in df.columns:
            # Avoid adding rows where id_start == id_end
            if id_start != id_end:
                # Append a dictionary for each pair (id_start, id_end, distance)
                rows.append({'id_start': id_start, 'id_end': id_end, 'distance': df.loc[id_start, id_end]})

    # Create the unrolled DataFrame from the list of rows
    unrolled_df = pd.DataFrame(rows)

    return unrolled_df


import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter rows where id_start matches the reference_id
    reference_distances = df[df['id_start'] == reference_id]['distance']

    # Calculate the average distance for the reference_id
    reference_avg = reference_distances.mean()

    # Define the 10% threshold range
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1

    # Calculate average distances for all ids
    avg_distances = df.groupby('id_start')['distance'].mean()

    # Filter IDs whose average distance is within the 10% threshold
    ids_within_threshold = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index

    # Return the filtered IDs as a sorted list
    return sorted(ids_within_threshold)


def calculate_toll_rate(df) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Create new columns for each vehicle type, calculating the toll based on distance
    df['moto'] = df['distance'] * rate_coefficients['moto']
    df['car'] = df['distance'] * rate_coefficients['car']
    df['rv'] = df['distance'] * rate_coefficients['rv']
    df['bus'] = df['distance'] * rate_coefficients['bus']
    df['truck'] = df['distance'] * rate_coefficients['truck']
    
    return df


import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define the time intervals and discount factors
    weekday_discount_factors = [
        {'start_time': time(0, 0), 'end_time': time(10, 0), 'factor': 0.8},
        {'start_time': time(10, 0), 'end_time': time(18, 0), 'factor': 1.2},
        {'start_time': time(18, 0), 'end_time': time(23, 59, 59), 'factor': 0.8}
    ]
    
    weekend_discount_factor = 0.7

    # List of days from Monday to Sunday
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Prepare the result DataFrame
    time_based_df = pd.DataFrame()

    # Iterate through all rows of the input DataFrame
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        moto = row['moto']
        car = row['car']
        rv = row['rv']
        bus = row['bus']
        truck = row['truck']
        
        # For each day of the week
        for day in days_of_week:
            # If it's a weekend, apply the constant discount factor
            if day in ['Saturday', 'Sunday']:
                discounted_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': time(0, 0),
                    'end_day': day,
                    'end_time': time(23, 59, 59),
                    'moto': moto * weekend_discount_factor,
                    'car': car * weekend_discount_factor,
                    'rv': rv * weekend_discount_factor,
                    'bus': bus * weekend_discount_factor,
                    'truck': truck * weekend_discount_factor
                }
                time_based_df = time_based_df.append(discounted_row, ignore_index=True)
            # If it's a weekday, apply different factors based on time ranges
            else:
                for interval in weekday_discount_factors:
                    discounted_row = {
                        'id_start': id_start,
                        'id_end': id_end,
                        'start_day': day,
                        'start_time': interval['start_time'],
                        'end_day': day,
                        'end_time': interval['end_time'],
                        'moto': moto * interval['factor'],
                        'car': car * interval['factor'],
                        'rv': rv * interval['factor'],
                        'bus': bus * interval['factor'],
                        'truck': truck * interval['factor']
                    }
                    time_based_df = time_based_df.append(discounted_row, ignore_index=True)

    return time_based_df
