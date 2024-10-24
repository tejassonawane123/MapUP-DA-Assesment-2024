
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/admin/Downloads/MapUp-DA-Assessment-2024-main/MapUp-DA-Assessment-2024-main/datasets/dataset-2.csv')


def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): DataFrame containing distance information.

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Get unique IDs from the DataFrame
    unique_ids = pd.concat([df['id_A'], df['id_B']]).unique()

    # Create a square DataFrame initialized to zero
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    # Fill the distance matrix with the known distances
    for _, row in df.iterrows():
        id_A = row['id_A']
        id_B = row['id_B']
        distance = row['distance']

        # Update the matrix for both directions
        distance_matrix.loc[id_A, id_B] = distance
        distance_matrix.loc[id_B, id_A] = distance

    # Calculate cumulative distances
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] > 0 and distance_matrix.at[k, j] > 0:
                    new_distance = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    if distance_matrix.at[i, j] == 0 or new_distance < distance_matrix.at[i, j]:
                        distance_matrix.at[i, j] = new_distance

    return distance_matrix




def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_matrix (pandas.DataFrame): The distance matrix with IDs as index and columns.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create an empty list to hold the rows
    rows = []

    # Iterate through the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude same id_start and id_end
                distance = distance_matrix.at[id_start, id_end]
                rows.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Convert the list of rows to a DataFrame
    unrolled_df = pd.DataFrame(rows)

    return unrolled_df



def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'id_start', 'id_end', and 'distance'.
        reference_id (int): The reference ID to compare against.

    Returns:
        pd.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                      of the reference ID's average distance.
    """
    # Calculate the average distance for the reference ID
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    # Calculate the 10% threshold
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1

    # Calculate average distances for all IDs
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    avg_distances.columns = ['id_start', 'avg_distance']

    # Find IDs within the threshold
    within_threshold = avg_distances[(avg_distances['avg_distance'] >= lower_bound) &
                                      (avg_distances['avg_distance'] <= upper_bound)]

    # Sort the result by id_start
    sorted_result = within_threshold.sort_values(by='id_start')

    return sorted_result



def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for each vehicle type's toll rate.
    """
    # Define the rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate the toll rates by multiplying the distance by the respective rate coefficients
    for vehicle, coefficient in rate_coefficients.items():
        df[vehicle] = df['distance'] * coefficient

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df


def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): Input DataFrame containing vehicle toll rates.

    Returns:
        pandas.DataFrame: DataFrame with updated toll rates based on time intervals.
    """
    # Define the days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Define time intervals and corresponding discount factors
    time_intervals = {
        'Weekdays': [
            (time(0, 0), time(10, 0), 0.8),
            (time(10, 0), time(18, 0), 1.2),
            (time(18, 0), time(23, 59, 59), 0.8)
        ],
        'Weekends': [
            (time(0, 0), time(23, 59, 59), 0.7)
        ]
    }

    # Create columns for start and end times/days
    df['start_day'] = np.random.choice(days_of_week, len(df))
    df['end_day'] = np.random.choice(days_of_week, len(df))
    df['start_time'] = time(0, 0)  # Starting from midnight
    df['end_time'] = time(23, 59, 59)  # Ending at just before midnight

    # Function to calculate the adjusted toll rates based on day and time
    def adjust_toll_rates(row):
        # Check the start day
        start_day = row['start_day']
        end_day = row['end_day']
        # Set initial discount factor
        discount_factor = 1.0

        # Adjust discount factor based on weekdays or weekends
        if start_day in days_of_week[:5]:  # Monday to Friday
            for start_time, end_time, factor in time_intervals['Weekdays']:
                if start_time <= row['start_time'] <= end_time:
                    discount_factor = factor
                    break
        else:  # Saturday and Sunday
            discount_factor = time_intervals['Weekends'][0][2]

        # Adjust toll rates for each vehicle type
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row[vehicle] *= discount_factor

        return row

    # Apply the toll rate adjustment
    df = df.apply(adjust_toll_rates, axis=1)

    return df
